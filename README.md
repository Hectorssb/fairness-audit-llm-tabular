# fairness-audit-llm-tabular

[![DOI](https://zenodo.org/badge/1284963024.svg)](https://doi.org/10.5281/zenodo.21068577)

Code and experimental outputs accompanying the paper *"A Non-Degenerate Fairness Auditing Framework for LLM-Based Tabular Decision Systems"*.

The framework evaluates bias-mitigation interventions for tabular classification with Large Language Models (LLMs) under in-context learning, combining data-level interventions (causal fair data via FLAI, resampling, semantic decontamination) and prompt-level interventions (zero-shot and few-shot Chain-of-Thought, native reasoning). Every prompt-level intervention is reported on both the original and the causally corrected test distribution, and fairness is scored by joint EOD+DI compliance after discounting group-level prediction collapse.

## What it does

The framework runs a structured set of experiments on three fairness benchmark datasets (Adult Income, COMPAS Recidivism, German Credit), measuring accuracy and fairness metrics (EOD, DI, SPD, OD) under three experimental blocks:

| Condition | Description |
|-----------|-------------|
| D1 | Few-shot ICL with original data |
| D2 | Few-shot ICL with FLAI fair-causal data |
| D3 | Few-shot ICL with resampled data |
| D4a | Zero-shot Chain-of-Thought, evaluated on both D1 and D2 test sets |
| D4b_D1 | Few-shot Chain-of-Thought with demonstrations sampled from D1, evaluated on both D1 and D2 test sets |
| D4b_D2 | Few-shot Chain-of-Thought with demonstrations sampled from D2, evaluated on both D1 and D2 test sets |
| D4r_0 | Native reasoning without demonstrations, evaluated on both D1 and D2 test sets |
| D4r_D1 | Native reasoning with demonstrations sampled from D1, evaluated on both D1 and D2 test sets |
| D4r_D2 | Native reasoning with demonstrations sampled from D2, evaluated on both D1 and D2 test sets |
| ZS_D1 | Zero-shot with original column names |
| ZS_D5 | Zero-shot with anonymized column names |
| D5 | Few-shot with anonymized column names (decontamination) |

Models are evaluated via two backends:
- **HuggingFace** (local inference on GPU): Llama-3.1-8B, Qwen2.5-7B, Qwen2.5-14B, GPT-OSS-20B, Gemma-4-E4B, Gemma-4-31B
- **Groq API**: Llama-3.3-70B, Qwen3-32B

## Requirements

- Python 3.10
- CUDA 12.x with an A100 40GB GPU (for HuggingFace models)
- A [Groq API key](https://console.groq.com/) (for Groq models)
- A [HuggingFace token](https://huggingface.co/settings/tokens) with access to gated models

Install dependencies:

```bash
pip install -r requirements.txt
```

Set up environment variables in a `.env` file:

```
HF_TOKEN=your_huggingface_token
GROQ_API_KEY_1=your_groq_key
# Optional: add multiple Groq keys to rotate (GROQ_API_KEY_2, etc.)
```

## Usage

**HuggingFace models** (runs all models sequentially, one at a time):

```bash
python main.py
# or in background:
nohup python main.py > logs/experiment.log 2>&1 &
```

**Groq models** (can run in parallel, one terminal per model):

```bash
nohup python main_groq.py --model llama-3.3-70b-versatile   > logs/groq_llama.log  2>&1 &
nohup python main_groq.py --model qwen/qwen3-32b            > logs/groq_qwen3.log  2>&1 &
```

Both entry points skip already-completed experiments automatically, so runs can be safely interrupted and resumed.

## Project structure

```
fairness-audit-llm-tabular/
├── main.py                  # HuggingFace experiment entry point
├── main_groq.py             # Groq API experiment entry point
├── analysis.ipynb           # Results analysis and figures
├── requirements.txt
├── src/
│   ├── data_loader.py           # Dataset loading and preprocessing
│   ├── fair_data.py             # FLAI wrapper: causal fair data generation
│   ├── hf_classifier.py         # HuggingFace few-shot classifier
│   ├── hf_cot_classifier.py     # HuggingFace CoT classifier (D4a, D4b)
│   ├── hf_zeroshot_classifier.py# HuggingFace zero-shot classifier
│   ├── groq_classifier.py       # Groq API classifier (all conditions)
│   ├── experiment_runner.py     # Experiment orchestration (HF)
│   ├── metrics.py               # Fairness metrics: EOD, DI, SPD, OD
│   ├── reasoning_generator.py   # Deterministic reasoning templates for CoT prompts
│   └── prompts/                 # System prompts and few-shot examples (JSON)
├── results.zip              # Experiment outputs (per-condition CSVs; unzip to results/)
├── data/                    # Generated datasets (created on run, gitignored)
└── logs/                    # Experiment logs (gitignored)
```

The raw per-condition outputs are bundled in `results.zip`. Unzip it to a
`results/` directory before running `analysis.ipynb`:

```bash
unzip results.zip
```

## Datasets

Downloaded automatically on first run via `ucimlrepo`:

- **Adult Income** — Kohavi (1996). Protected: sex, race.
- **COMPAS Recidivism** — Angwin et al. (2016). Protected: sex, race.
- **German Credit** — Dua & Graff (2017). Protected: sex, age.

## Reporting protocol

Block 1 (D1, D2, D3), Block 3 (ZS_D1, ZS_D5, D5), and the decontaminated few-shot condition are reported on the D1 test split. Block 2 prompt-level conditions (D4a, D4b_D1, D4b_D2, D4r_0, D4r_D1, D4r_D2) are each evaluated twice: once on the D1 test split and once on the D2 test split. The `_testD1` / `_testD2` filename suffix identifies the evaluation distribution.

## FLAI

Fair data generation (D2, D3) is based on the **FLAI** library (González-Sendino et al., 2024), which learns a causal graph from the original data and generates a new dataset where the influence of protected attributes on the target variable is mitigated through causal intervention.

> González-Sendino, R., Serrano, E., Bajo, J. (2024). *FLAI: Fairness-aware causal data generation for machine learning*.
> GitHub: [https://github.com/rugonzs/FLAI](https://github.com/rugonzs/FLAI)
> Package: `pip install flai-causal`

## How to cite

If you use this code or data, please cite the accompanying article and the
archived software (DOI: [10.5281/zenodo.21068577](https://doi.org/10.5281/zenodo.21068577)).
Machine-readable metadata is provided in [`CITATION.cff`](CITATION.cff).

```bibtex
@software{sanchez2026fairnessaudit_software,
  author    = {S{\'a}nchez San-Blas, H{\'e}ctor and Serrano, Emilio and
               Gonz{\'a}lez-Sendino, Rub{\'e}n and Lozano Murciego, {\'A}lvaro and Bajo, Javier},
  title     = {fairness-audit-llm-tabular: A Non-Degenerate Fairness Auditing
               Framework for LLM-Based Tabular Decision Systems},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21068577},
  url       = {https://doi.org/10.5281/zenodo.21068577}
}
```

## License

This project is released under the BSD 3-Clause License. See [`LICENSE`](LICENSE).
