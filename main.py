"""Main entry point for the HuggingFace experiment runner."""

import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable] + sys.argv, {**os.environ, "PYTHONHASHSEED": "0"})

import argparse
import gc
import traceback
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

import torch

sys.path.insert(0, "src")

print(f"[main] CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"[main] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[main] VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("[main] WARNING: No GPU detected — running on CPU (will be very slow)")

from data_loader import load_dataset, DATASET_CONFIG
from fair_data import prepare_all_datasets
from hf_classifier import load_model, MODEL_CONFIGS
from metrics import consolidate_results
from experiment_runner import run_all_experiments, release_model, HF_D4R_MODELS
from xgboost_baselines import run_all_baselines


# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------

DATASETS   = ["adult", "compas", "german"]
TEST_SIZE  = 500
N_EXAMPLES = 10
SEED       = 42

MODELS = [
    "meta-llama/Llama-3.1-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "google/gemma-4-E4B-it",
    "Qwen/Qwen2.5-14B-Instruct",
    "openai/gpt-oss-20b",
    "microsoft/phi-4",
    "Qwen/Qwen3-32B",
    "google/gemma-4-31B-it",
]

RESULTS_DIR = Path("results")
DATA_DIR    = Path("data")
RESULTS_DIR.mkdir(exist_ok=True)
DATA_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# STEP 1 — DATA GENERATION
# ---------------------------------------------------------------------------

def generate_data():
    for dataset_name in DATASETS:
        cfg = DATASET_CONFIG[dataset_name]
        expected = [
            DATA_DIR / dataset_name / fname
            for fname in ["D1_train.csv", "D1_test.csv", "D2_train.csv", "D2_test.csv", "D3_train.csv"]
        ]

        if all(p.exists() for p in expected):
            print(f"[{dataset_name}] Data already exists, skipping generation.")
            continue

        print(f"\n[{dataset_name}] Generating canonical split + D2/D3...")
        df, _ = load_dataset(dataset_name)
        prepare_all_datasets(
            df=df,
            dataset_name=dataset_name,
            target_col=cfg["target"],
            sensitive_features=cfg["protected"],
            privileged_values=cfg["privileged"],
            output_dir=DATA_DIR,
            test_size=TEST_SIZE,
            seed=SEED,
        )
    print("\nData ready.")


# ---------------------------------------------------------------------------
# STEP 2 — RETRAINABLE BASELINES
# ---------------------------------------------------------------------------

def run_baselines():
    run_all_baselines(
        datasets=DATASETS, data_dir=DATA_DIR, results_dir=RESULTS_DIR, seed=SEED,
    )


# ---------------------------------------------------------------------------
# STEP 3 — EXPERIMENTS
# ---------------------------------------------------------------------------

def _all_results_exist(model_tag: str, model_id: str, datasets: list) -> bool:
    """Return True iff every expected output CSV exists for this model."""
    for dataset_name in datasets:
        results_out = RESULTS_DIR / dataset_name / model_tag
        expected = []
        for condition in ["D1_original", "D2_fair_causal", "D3_resampled"]:
            for suffix in ("", "_testD2"):
                expected.append(f"{dataset_name}_{condition}_LLM_FewShot_{model_tag}{suffix}_table_performance.csv")
        for test_tag in ("D1", "D2"):
            expected.append(f"{dataset_name}_D4a_LLM_ZSCoT_D4a_{model_tag}_test{test_tag}_table_performance.csv")
            expected.append(f"{dataset_name}_D4b_D1_LLM_FSCoT_D4b_{model_tag}_test{test_tag}_table_performance.csv")
            expected.append(f"{dataset_name}_D4b_D2_LLM_FSCoT_D4b_{model_tag}_test{test_tag}_table_performance.csv")
            expected.append(f"{dataset_name}_D5_decontam_LLM_FewShot_{model_tag}_test{test_tag}_table_performance.csv")
            expected.append(f"{dataset_name}_D6_fairprompt_LLM_FewShot_{model_tag}_test{test_tag}_table_performance.csv")
            if model_id in HF_D4R_MODELS:
                for d4r in ["D4r_0", "D4r_D1", "D4r_D2"]:
                    expected.append(f"{dataset_name}_{d4r}_LLM_D4r_high_{model_tag}_test{test_tag}_table_performance.csv")
        for zs in ["ZS_D1", "ZS_D5"]:
            for suffix in ("", "_testD2"):
                expected.append(f"{dataset_name}_{zs}_LLM_ZeroShot_{model_tag}{suffix}_table_performance.csv")
        for fname in expected:
            if not (results_out / fname).exists():
                return False
    return True


def run_experiments(models: list, datasets: list):
    demo_seed = int(os.environ.get("DEMO_SEED", SEED))
    failed = []
    for model_id in models:
        model_tag = model_id.replace("/", "-").replace(".", "-")
        print(f"\n{'='*70}")
        print(f"  MODEL: {MODEL_CONFIGS[model_id]['desc']}")
        print(f"{'='*70}")

        if demo_seed == SEED and _all_results_exist(model_tag, model_id, datasets):
            print(f"  [SKIP] All results already exist for {model_id}.")
            continue

        model = tokenizer = None
        try:
            model, tokenizer = load_model(model_id)

            run_all_experiments(
                model_id=model_id, model=model, tokenizer=tokenizer,
                datasets=datasets, results_dir=RESULTS_DIR, data_dir=DATA_DIR,
                n_examples=N_EXAMPLES,
            )

            # Fairness summary for this model
            t1_all, t2_all = consolidate_results(RESULTS_DIR)
            if t1_all is not None:
                t2_model = t2_all[t2_all["Algorithm"].str.contains(model_tag, na=False)]
                if not t2_model.empty:
                    print("\n--- Table 2 (Fairness) — this model ---")
                    print(t2_model.to_string(index=False))
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            failed.append((model_id, f"{type(exc).__name__}: {exc}"))
            print(f"\n[error] {model_id} stopped: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        finally:
            if model is not None:
                try:
                    release_model(model_id, model, tokenizer)
                except BaseException as release_exc:
                    print(f"[mem] Could not release {model_id}: {release_exc}")
            model = tokenizer = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print("\n" + "="*70)
    if failed:
        print(f"  COMPLETED WITH {len(failed)} MODEL(S) FAILED")
        for model_id, message in failed:
            print(f"    - {model_id}: {message}")
        print("  Rerun the same command to resume: existing results are skipped.")
    else:
        print("  ALL MODELS COMPLETED")
    print("="*70)
    return failed


# ---------------------------------------------------------------------------
# STEP 4 — EXPORT
# ---------------------------------------------------------------------------

def export_results():
    t1_all, t2_all = consolidate_results(RESULTS_DIR)
    if t1_all is None:
        print("[export] No results found.")
        return

    print(f"Results loaded: {len(t2_all)} rows in Table 2")
    print("[export] For figures and LaTeX tables, run analysis.ipynb.")


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="HuggingFace experiment runner.")
    parser.add_argument(
        "--data-only", action="store_true",
        help="Generate the canonical splits and run the XGBoost baselines, "
             "then stop before loading any LLM.",
    )
    parser.add_argument(
        "--models", nargs="+", metavar="MODEL_ID", default=MODELS,
        help="HuggingFace ids to run, in order (default: every model in MODELS). "
             "Results live under results/<dataset>/<model>, so distinct models "
             "can run as separate jobs at the same time.",
    )
    parser.add_argument(
        "--datasets", nargs="+", choices=DATASETS, default=DATASETS,
        help="Datasets to run (default: all three).",
    )
    args = parser.parse_args()
    unknown = [m for m in args.models if m not in MODEL_CONFIGS]
    if unknown:
        parser.error(f"unknown model id(s): {unknown}; known: {list(MODEL_CONFIGS)}")
    return args


if __name__ == "__main__":
    args = parse_args()

    generate_data()
    run_baselines()

    if args.data_only:
        print("\n[main] --data-only: stopping before the LLM experiments.")
    else:
        failed = run_experiments(args.models, args.datasets)
        export_results()
        if failed:
            sys.exit(1)
