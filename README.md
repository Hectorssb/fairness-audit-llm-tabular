# fairness-audit-llm-tabular

[![DOI](https://zenodo.org/badge/1284963024.svg)](https://doi.org/10.5281/zenodo.21068577)

Code and experimental outputs for an audit of fairness evaluation practice in tabular classification with Large Language Models.

It evaluates Large Language Models (LLMs) under in-context learning, using thirteen data-level and prompt-level intervention conditions as a testbed: data-level interventions (causal fair data via FLAI, resampling, semantic decontamination) and prompt-level interventions (zero-shot and few-shot Chain-of-Thought, native reasoning, debiasing instruction). Every condition is evaluated on both the original and the causally corrected test distribution, and fairness is scored by joint EOD+DI compliance after discounting group-level prediction collapse, together with a task-accuracy check against the majority-class baseline.

## What it does

The pipeline runs a structured set of experiments on three fairness benchmark datasets (Adult Income, COMPAS Recidivism, German Credit), measuring accuracy and fairness metrics (EOD, DI, SPD, OD) under three experimental blocks:

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
| D5 | Few-shot with anonymized column names (decontamination) |
| D6 | Few-shot with an explicit debiasing instruction |
| ZS_D1 | Zero-shot with original column names |
| ZS_D5 | Zero-shot with anonymized column names |

All eight models run locally on GPU through HuggingFace, with greedy decoding
and fixed seeds against a stable open-weights checkpoint, which is what makes
the predictions reproducible from this code:

Llama-3.1-8B, Qwen2.5-7B, Qwen2.5-14B, GPT-OSS-20B, Gemma-4-E4B, Gemma-4-31B,
Phi-4 and Qwen3-32B.

Four of them (GPT-OSS-20B, Gemma-4-E4B, Gemma-4-31B, Qwen3-32B) expose a native
reasoning mode, and the D4r block uses those four to contrast internal reasoning
with an explicit chain of thought within the same model.

Alongside the LLM conditions, `src/xgboost_baselines.py` runs the retrainable
paradigm on the same canonical splits, covering the three mitigation families:
pre-processing (training on D1, D2 and D3), in-processing (a reduction with an
equalized-odds constraint) and post-processing (per-group thresholds).

## Requirements

- Python 3.10
- CUDA 12.x with at least 40 GB of GPU memory
- A [HuggingFace token](https://huggingface.co/settings/tokens) with access to gated models

Install dependencies:

```bash
pip install -r requirements.txt
```

Set up the environment variable in a `.env` file:

```
HF_TOKEN=your_huggingface_token
```

## Usage

Run all models sequentially, one at a time:

```bash
python main.py
# or in background:
nohup python main.py > logs/experiment.log 2>&1 &
```

`main.py` generates the canonical splits (or reuses those shipped in `data/`),
runs the XGBoost baselines, and then
runs the LLM conditions. It skips already-completed experiments automatically,
so runs can be safely interrupted and resumed. The baselines can also be run on
their own once the data step has completed:

```bash
PYTHONPATH=src python -m xgboost_baselines
```

## Analysis

`analysis.ipynb` computes every quantity the experiment produces, from the raw
prediction files to the audit levels. The audit has three levels, each stricter
than the last:

1. **Raw GF** — jointly compliant on equal-opportunity difference (`|EOD| <= 0.10`)
   and disparate impact (`0.80 <= DI <= 1.20`).
2. **Non-collapsed GF** — raw GF that also survives the collapse filter, which
   discounts cells whose classifier answers one label almost always or whose
   group rates saturate.
3. **Non-collapsed GF + utility** — the cell must also reach the majority-class
   predictor, scored with the same group-unweighted estimator.

Neither is a normative claim of fairness: the gates are descriptors used to
audit a protocol, not a validated fairness standard.

The notebook also covers transfer between the two canonical test splits with
paired bootstrap intervals, the sensitivity of the collapse criterion to both of
its margins, the XGBoost baselines, and the fairness–utility frontier within
each regime.

Every gate is defined once, in `analysis/cells.py`, and imported from there, so
a number cannot be computed two ways.

## Tests

```bash
python -m pytest tests/
```

The suite checks the parts of the protocol that a reader cannot verify by
inspection: that the canonical split is fixed before any derived artifact is
generated and that FLAI never sees a test row; that the German column mapping
follows the official UCI codebook and that a cache written by an earlier layout
is rejected; that the COMPAS loader never passes `score_text` to the model; that
no generated reasoning chain mentions a protected attribute; that the
demonstrations are drawn balanced and reproducibly; and that the notebook reads
its gates from `analysis/cells.py` rather than redefining them.

The suite runs on CPU in a few seconds and downloads nothing: the checks that
need a GPU library are skipped when it is absent, and the two that read
`results/` are skipped until it is unzipped.

## Project structure

```
fairness-audit-llm-tabular/
├── main.py                  # Experiment entry point
├── analysis.ipynb           # Results analysis: the three audit levels, transfer and frontier
├── requirements.txt
├── src/
│   ├── data_loader.py           # Dataset loading and preprocessing
│   ├── fair_data.py             # FLAI wrapper: causal fair data generation
│   ├── hf_classifier.py         # Few-shot classifier
│   ├── hf_cot_classifier.py     # CoT classifier (D4a, D4b)
│   ├── hf_zeroshot_classifier.py# Zero-shot classifier
│   ├── experiment_runner.py     # Experiment orchestration
│   ├── metrics.py               # Fairness metrics: EOD, DI, SPD, OD
│   ├── xgboost_baselines.py     # Retrainable baselines (pre/in/post-processing)
│   ├── reasoning_generator.py   # Deterministic reasoning templates for CoT prompts
│   └── prompts/                 # System prompts and few-shot examples (JSON)
├── analysis/
│   └── cells.py                 # Audit gates and result loading
├── tests/                   # Protocol, loader, metric and notebook tests
├── results.zip              # Experiment outputs (per-condition CSVs; unzip to results/)
├── data/                    # Canonical splits used in the paper (raw downloads gitignored)
└── logs/                    # Experiment logs (gitignored)
```

The raw per-condition outputs are bundled in `results.zip`. Unzip it to a
`results/` directory before running `analysis.ipynb`:

```bash
unzip results.zip
```

The result tables carry the collapse flag (`insufficient_support`) and the group
sizes (`N`) the audit reads, so they must come from this release: an earlier
`results/` directory is rejected with an explicit message rather than silently
producing different numbers.

## Datasets

Downloaded automatically on first run via `ucimlrepo`:

- **Adult Income** — Kohavi (1996). Protected: sex, race.
- **COMPAS Recidivism** — Angwin et al. (2016). Protected: sex, race.
- **German Credit** — Dua & Graff (2017). Protected: sex, age.

Column mappings follow the official UCI codebooks. For German Credit in
particular, `Attribute1` is checking-account status, `Attribute3` credit
history, `Attribute15` housing and `Attribute17` job; `data_loader.py` records
the column layout each loader produces and rejects any cache that does not match
it, rather than letting a stale file shadow the loader.

## Reporting protocol

A single stratified train/test split of the original data is fixed first, and
every derived artifact — the FLAI causal model, D2 and D3 — is generated from
the training half alone, so no evaluation row participates in fitting,
discretisation or mitigation.

Every condition is evaluated twice: once on the D1 test split and once on the
D2 test split. The `_testD1` / `_testD2` filename suffix identifies the
evaluation distribution; the D1-test outputs of D1, D2, D3, ZS_D1 and ZS_D5
carry no suffix.

D1 and D2 test sets are not paired: D2 is an independent synthetic sample from
the mitigated causal model, not a row-wise counterfactual of D1. The transfer
measure is read with that limit in mind.

## FLAI

Fair data generation (D2, D3) is based on the **FLAI** library (González-Sendino et al., 2024), which learns a causal graph from the original data and generates a new dataset where the influence of protected attributes on the target variable is mitigated through causal intervention.

> González-Sendino, R., Serrano, E., Bajo, J. (2024). *FLAI: Fairness-aware causal data generation for machine learning*.
> GitHub: [https://github.com/rugonzs/FLAI](https://github.com/rugonzs/FLAI)
> Package: `pip install flai-causal`

## How to cite

If you use this code or data, please cite the archived software
(DOI: [10.5281/zenodo.21068577](https://doi.org/10.5281/zenodo.21068577)).
Machine-readable metadata is provided in [`CITATION.cff`](CITATION.cff).

```bibtex
@software{sanchez2026fairnessaudit_software,
  author    = {S{\'a}nchez San-Blas, H{\'e}ctor and Serrano, Emilio and
               Gonz{\'a}lez-Sendino, Rub{\'e}n and Lozano Murciego, {\'A}lvaro and Bajo, Javier},
  title     = {fairness-audit-llm-tabular: When Fairness Metrics Fail: Auditing
               Collapse and Transfer Risks in LLM Decision Systems},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21068577},
  url       = {https://doi.org/10.5281/zenodo.21068577}
}
```

## License

This project is released under the BSD 3-Clause License. See [`LICENSE`](LICENSE).
