"""Entry point for Groq-based experiments (llama-3.3-70b, qwen/qwen3-32b).

Usage:
    python main_groq.py                                  # all models
    python main_groq.py --model llama-3.3-70b-versatile  # single model
    python main_groq.py --model qwen/qwen3-32b
"""

import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, "src")

from data_loader import load_dataset, DATASET_CONFIG
from fair_data import prepare_all_datasets
from metrics import consolidate_results
from groq_classifier import (
    run_fewshot_groq,
    run_zs_cot_groq,
    run_fs_cot_groq,
    run_d4r_groq,
    run_zs_d1_groq,
    run_zs_d5_groq,
    run_decontam_groq,
)


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

DATASETS   = ["adult", "compas", "german"]
TEST_SIZE  = 500
N_EXAMPLES = 10
SEED       = 42

MODELS = [
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
]

RESULTS_DIR = Path("results")
DATA_DIR    = Path("data")
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# STEP 1 — DATA
# ---------------------------------------------------------------------------

def generate_data():
    for dataset_name in DATASETS:
        cfg = DATASET_CONFIG[dataset_name]
        d1 = DATA_DIR / dataset_name / "D1_original.csv"
        d2 = DATA_DIR / dataset_name / "D2_fair_causal.csv"
        d3 = DATA_DIR / dataset_name / "D3_resampled.csv"

        if d1.exists() and d2.exists() and d3.exists():
            print(f"[{dataset_name}] Data already exists, skipping generation.")
            continue

        print(f"\n[{dataset_name}] Generating D1, D2, D3...")
        df, _ = load_dataset(dataset_name)
        prepare_all_datasets(
            df=df,
            dataset_name=dataset_name,
            target_col=cfg["target"],
            sensitive_features=cfg["protected"],
            privileged_values=cfg["privileged"],
            output_dir=DATA_DIR,
        )
    print("\nData ready.")


# ---------------------------------------------------------------------------
# STEP 2 — CHECK IF RESULTS EXIST
# ---------------------------------------------------------------------------

def _get_train_test(dataset_name: str, condition: str, test_size: int, seed: int):
    cfg    = DATASET_CONFIG[dataset_name]
    target = cfg["target"]
    df_d1  = pd.read_csv(DATA_DIR / dataset_name / "D1_original.csv")
    _, df_test = train_test_split(df_d1, test_size=test_size, random_state=seed, stratify=df_d1[target])
    df_cond = pd.read_csv(DATA_DIR / dataset_name / f"{condition}.csv")
    df_train, _ = train_test_split(df_cond, test_size=test_size, random_state=seed, stratify=df_cond[target])
    return df_train, df_test.reset_index(drop=True)


_D4R_MODELS = {"qwen/qwen3-32b"}


def _get_d4_test_sets(dataset_name: str, test_size: int, seed: int) -> dict:
    """Return {'D1': df_test_d1, 'D2': df_test_d2} for D4* dual-test evaluation."""
    cfg    = DATASET_CONFIG[dataset_name]
    target = cfg["target"]
    df_d1 = pd.read_csv(DATA_DIR / dataset_name / "D1_original.csv")
    _, df_test_d1 = train_test_split(df_d1, test_size=test_size, random_state=seed, stratify=df_d1[target])
    df_d2 = pd.read_csv(DATA_DIR / dataset_name / "D2_fair_causal.csv")
    _, df_test_d2 = train_test_split(df_d2, test_size=test_size, random_state=seed, stratify=df_d2[target])
    return {"D1": df_test_d1.reset_index(drop=True),
            "D2": df_test_d2.reset_index(drop=True)}


def _all_results_exist(model_tag: str, model_id: str) -> bool:
    """Return True iff every expected output CSV exists for this model."""
    for dataset_name in DATASETS:
        results_out = RESULTS_DIR / dataset_name / model_tag
        expected = []
        for condition in ["D1_original", "D2_fair_causal", "D3_resampled"]:
            expected.append(f"{dataset_name}_{condition}_LLM_FewShot_{model_tag}_table_performance.csv")
        for test_tag in ("D1", "D2"):
            expected.append(f"{dataset_name}_D4a_LLM_ZSCoT_D4a_{model_tag}_test{test_tag}_table_performance.csv")
            expected.append(f"{dataset_name}_D4b_D1_LLM_FSCoT_D4b_{model_tag}_test{test_tag}_table_performance.csv")
            expected.append(f"{dataset_name}_D4b_D2_LLM_FSCoT_D4b_{model_tag}_test{test_tag}_table_performance.csv")
            if model_id in _D4R_MODELS:
                for d4r in ["D4r_0", "D4r_D1", "D4r_D2"]:
                    expected.append(f"{dataset_name}_{d4r}_LLM_D4r_high_{model_tag}_test{test_tag}_table_performance.csv")
        for zs in ["ZS_D1", "ZS_D5"]:
            expected.append(f"{dataset_name}_{zs}_LLM_ZeroShot_{model_tag}_table_performance.csv")
        expected.append(f"{dataset_name}_D5_decontam_LLM_FewShot_{model_tag}_table_performance.csv")
        for fname in expected:
            if not (results_out / fname).exists():
                return False
    return True


def _skip(path: Path, label: str) -> bool:
    if path.exists():
        print(f"  [SKIP] {label} already exists.")
        return True
    return False


# ---------------------------------------------------------------------------
# STEP 3 — EXPERIMENTS
# ---------------------------------------------------------------------------

def run_experiments():
    for model_id in MODELS:
        model_tag = model_id.replace("/", "-").replace(".", "-")
        print(f"\n{'='*70}")
        print(f"  MODEL: {model_id}")
        print(f"{'='*70}")

        if _all_results_exist(model_tag, model_id):
            print(f"  [SKIP] All results already exist for {model_id}.")
            continue

        for dataset_name in DATASETS:
            results_out = RESULTS_DIR / dataset_name / model_tag
            results_out.mkdir(parents=True, exist_ok=True)
            print(f"\n  Dataset: {dataset_name.upper()}")

            cfg = DATASET_CONFIG[dataset_name]
            target_col = cfg["target"]
            df_d1 = pd.read_csv(DATA_DIR / dataset_name / "D1_original.csv")
            df_d1_train, _ = train_test_split(df_d1, test_size=TEST_SIZE, random_state=SEED, stratify=df_d1[target_col])
            df_d2 = pd.read_csv(DATA_DIR / dataset_name / "D2_fair_causal.csv")
            df_d2_train, _ = train_test_split(df_d2, test_size=TEST_SIZE, random_state=SEED, stratify=df_d2[target_col])

            d4_test_sets = _get_d4_test_sets(dataset_name, TEST_SIZE, SEED)

            for condition in ["D1_original", "D2_fair_causal", "D3_resampled"]:
                prefix = f"{dataset_name}_{condition}_LLM_FewShot_{model_tag}"
                if _skip(results_out / f"{prefix}_table_performance.csv", f"ICL {condition}"):
                    continue
                print(f"  [ICL FewShot] {condition}")
                df_train, df_test = _get_train_test(dataset_name, condition, TEST_SIZE, SEED)
                run_fewshot_groq(
                    df_train=df_train, df_test=df_test,
                    dataset_name=dataset_name, data_condition=condition,
                    model_id=model_id, n_examples=N_EXAMPLES, output_dir=results_out,
                )

            for test_tag, df_test in d4_test_sets.items():
                d4a_prefix = f"{dataset_name}_D4a_LLM_ZSCoT_D4a_{model_tag}_test{test_tag}"
                if not _skip(results_out / f"{d4a_prefix}_table_performance.csv", f"D4a test={test_tag}"):
                    print(f"  [D4a ZS-CoT] D4a (test={test_tag})")
                    run_zs_cot_groq(
                        df_test=df_test, dataset_name=dataset_name,
                        data_condition="D4a", model_id=model_id,
                        output_dir=results_out, test_set=test_tag,
                    )

                d4b_d1_prefix = f"{dataset_name}_D4b_D1_LLM_FSCoT_D4b_{model_tag}_test{test_tag}"
                if not _skip(results_out / f"{d4b_d1_prefix}_table_performance.csv", f"D4b_D1 test={test_tag}"):
                    print(f"  [D4b FS-CoT] D4b_D1 (demos from D1, test={test_tag})")
                    run_fs_cot_groq(
                        df_train=df_d1_train, df_test=df_test,
                        dataset_name=dataset_name, data_condition="D4b_D1",
                        model_id=model_id, output_dir=results_out, test_set=test_tag,
                    )

                d4b_d2_prefix = f"{dataset_name}_D4b_D2_LLM_FSCoT_D4b_{model_tag}_test{test_tag}"
                if not _skip(results_out / f"{d4b_d2_prefix}_table_performance.csv", f"D4b_D2 test={test_tag}"):
                    print(f"  [D4b FS-CoT] D4b_D2 (demos from D2, test={test_tag})")
                    run_fs_cot_groq(
                        df_train=df_d2_train, df_test=df_test,
                        dataset_name=dataset_name, data_condition="D4b_D2",
                        model_id=model_id, output_dir=results_out, test_set=test_tag,
                    )

                if model_id in _D4R_MODELS:
                    for df_train_local, cond_label, desc in [
                        (None,        "D4r_0",  "D4r_0 (no demos)"),
                        (df_d1_train, "D4r_D1", "D4r_D1 (demos from D1)"),
                        (df_d2_train, "D4r_D2", "D4r_D2 (demos from D2)"),
                    ]:
                        d4r_prefix = f"{dataset_name}_{cond_label}_LLM_D4r_high_{model_tag}_test{test_tag}"
                        if _skip(results_out / f"{d4r_prefix}_table_performance.csv", f"D4r {cond_label} test={test_tag}"):
                            continue
                        print(f"  [D4r Native Reasoning] {desc} (test={test_tag})")
                        run_d4r_groq(
                            df_test=df_test, dataset_name=dataset_name,
                            data_condition=cond_label, model_id=model_id,
                            df_train=df_train_local, reasoning_effort="high",
                            output_dir=results_out, test_set=test_tag,
                        )

            prefix_zs1 = f"{dataset_name}_ZS_D1_LLM_ZeroShot_{model_tag}"
            if not _skip(results_out / f"{prefix_zs1}_table_performance.csv", "ZS_D1"):
                print(f"  [ZeroShot] ZS_D1")
                run_zs_d1_groq(
                    dataset_name=dataset_name, model_id=model_id,
                    data_dir=DATA_DIR, test_size=TEST_SIZE, seed=SEED,
                    output_dir=results_out,
                )

            prefix_zs5 = f"{dataset_name}_ZS_D5_LLM_ZeroShot_{model_tag}"
            if not _skip(results_out / f"{prefix_zs5}_table_performance.csv", "ZS_D5"):
                print(f"  [ZeroShot] ZS_D5")
                run_zs_d5_groq(
                    dataset_name=dataset_name, model_id=model_id,
                    data_dir=DATA_DIR, test_size=TEST_SIZE, seed=SEED,
                    output_dir=results_out,
                )

            prefix_d5 = f"{dataset_name}_D5_decontam_LLM_FewShot_{model_tag}"
            if not _skip(results_out / f"{prefix_d5}_table_performance.csv", "D5"):
                print(f"  [D5] Decontamination")
                df_train, df_test = _get_train_test(dataset_name, "D1_original", TEST_SIZE, SEED)
                run_decontam_groq(
                    df_train=df_train, df_test=df_test,
                    dataset_name=dataset_name, model_id=model_id,
                    n_examples=N_EXAMPLES, seed=SEED, output_dir=results_out,
                )

    print("\n" + "="*70)
    print("  ALL MODELS COMPLETED")
    print("="*70)


# ---------------------------------------------------------------------------
# STEP 4 — EXPORT
# ---------------------------------------------------------------------------

def export_results():
    t1_all, t2_all = consolidate_results(RESULTS_DIR)
    if t1_all is None:
        print("[export] No results found.")
        return

    print(f"Results loaded: {len(t2_all)} rows in Table 2")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Groq experiment runner")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Run only this model (e.g. 'qwen/qwen3-32b'). "
             "If omitted, runs all models in MODELS list sequentially."
    )
    args = parser.parse_args()

    if args.model:
        if args.model not in MODELS:
            print(f"[main_groq] WARNING: '{args.model}' not in MODELS list. Running anyway.")
        MODELS = [args.model]
        print(f"[main_groq] Running single model: {args.model}")

    generate_data()
    run_experiments()
    export_results()
