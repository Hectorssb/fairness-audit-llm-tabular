"""Main entry point for the HuggingFace experiment runner."""

import sys
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
    "Qwen/Qwen2.5-14B-Instruct",
    "openai/gpt-oss-20b",
    "google/gemma-4-E4B-it",
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
# STEP 2 — EXPERIMENTS
# ---------------------------------------------------------------------------

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
            if model_id in HF_D4R_MODELS:
                for d4r in ["D4r_0", "D4r_D1", "D4r_D2"]:
                    expected.append(f"{dataset_name}_{d4r}_LLM_D4r_high_{model_tag}_test{test_tag}_table_performance.csv")
        for zs in ["ZS_D1", "ZS_D5"]:
            expected.append(f"{dataset_name}_{zs}_LLM_ZeroShot_{model_tag}_table_performance.csv")
        expected.append(f"{dataset_name}_D5_decontam_LLM_FewShot_{model_tag}_table_performance.csv")
        for fname in expected:
            if not (results_out / fname).exists():
                return False
    return True


def run_experiments():
    for model_id in MODELS:
        model_tag = model_id.replace("/", "-").replace(".", "-")
        print(f"\n{'='*70}")
        print(f"  MODEL: {MODEL_CONFIGS[model_id]['desc']}")
        print(f"{'='*70}")

        if _all_results_exist(model_tag, model_id):
            print(f"  [SKIP] All results already exist for {model_id}.")
            continue

        model, tokenizer = load_model(model_id)

        run_all_experiments(
            model_id=model_id, model=model, tokenizer=tokenizer,
            datasets=DATASETS, results_dir=RESULTS_DIR, data_dir=DATA_DIR,
            n_examples=N_EXAMPLES, test_size=TEST_SIZE, seed=SEED,
        )

        # Fairness summary for this model
        t1_all, t2_all = consolidate_results(RESULTS_DIR)
        if t1_all is not None:
            t2_model = t2_all[t2_all["Algorithm"].str.contains(model_tag, na=False)]
            if not t2_model.empty:
                print("\n--- Table 2 (Fairness) — this model ---")
                print(t2_model.to_string(index=False))

        release_model(model_id, model, tokenizer)

    print("\n" + "="*70)
    print("  ALL MODELS COMPLETED")
    print("="*70)


# ---------------------------------------------------------------------------
# STEP 3 — EXPORT
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

if __name__ == "__main__":
    generate_data()
    run_experiments()
    export_results()
