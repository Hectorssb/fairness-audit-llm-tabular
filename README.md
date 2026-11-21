# fairness-audit-llm-tabular

[![DOI](https://zenodo.org/badge/1284963024.svg)](https://doi.org/10.5281/zenodo.21068577)

Code and experimental outputs for an audit of fairness evaluation practice in tabular classification with Large Language Models.

It evaluates Large Language Models (LLMs) under in-context learning, using thirteen data-level and prompt-level intervention conditions as a testbed: data-level interventions (causal fair data via FLAI, resampling, semantic decontamination) and prompt-level interventions (zero-shot and few-shot Chain-of-Thought, native reasoning, debiasing instruction). Every condition is evaluated on both the original and the causally corrected test distribution, and fairness is scored by joint EOD+DI compliance after discounting group-level prediction collapse, together with a task-accuracy check against the majority-class baseline. The conditions that receive demonstrations are run under three draws of their ten demonstrations, and every count of compliant cells is reported as the median over the three.

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

The nine conditions that receive demonstrations (D1, D2, D3, D4b_D1, D4b_D2,
D4r_D1, D4r_D2, D5 and D6) are run under three draws of their ten
demonstrations, with seeds 42, 1 and 8, on the same test instances and prompts.
The paper labels the draws A, B and C. The four conditions without
demonstrations (D4a, D4r_0, ZS_D1 and ZS_D5) do not depend on the draw.

The eight models run locally on GPU through HuggingFace with greedy decoding
and fixed seeds. They are Llama-3.1-8B, Qwen2.5-7B, Qwen2.5-14B, GPT-OSS-20B,
Gemma-4-E4B, Gemma-4-31B, Phi-4 and Qwen3-32B. The analysis reads the archived
predictions, so its results are exactly reproducible from this release. A
validation set of zero-shot responses collected with the same prompts and
inference code (`results_checks.zip`) agrees with the archived predictions on
every instance for seven of the eight models, and on 85% to 100% of the
instances of each GPT-OSS-20B pass.

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

`python main.py` runs every condition with the demonstrations of draw A
(seed 42). The environment variable `DEMO_SEED` selects the draw, and draws B
and C are run with:

```bash
DEMO_SEED=1 python main.py
DEMO_SEED=8 python main.py
```

With a seed other than 42, only the LLM conditions that receive demonstrations
run, on the same test splits and prompts. Their output files are written to
`results/` with a `_seed<n>` tag before the suffix, so no draw overwrites
another. The analysis reads `results/` as draw A and looks for the other draws
in `results_seed<n>/`, so move the tagged files there before running it, once
per seed:

```bash
n=1
for f in results/*/*/*_seed${n}_*; do
  mkdir -p "results_seed${n}/$(dirname "${f#results/}")" && mv "$f" "results_seed${n}/${f#results/}"
done
```

## Analysis

`analysis.ipynb` computes the audit levels from the archived results. The
bootstrap uses `REPLICATES = 200` by default; set it to 2000 to reproduce the
count intervals in the paper. The audit has three levels:

1. **Raw GF** — jointly compliant on equal-opportunity difference (`|EOD| <= 0.10`)
   and disparate impact (`0.80 <= DI <= 1.20`).
2. **Non-collapsed GF** — raw GF that also survives the collapse filter, which
   discounts cells whose classifier answers one label almost always or whose
   group rates saturate.
3. **Non-collapsed GF + utility** — the cell must also reach the majority-class
   predictor, scored with the same group-unweighted estimator.

None of the three is a normative claim of fairness: the gates are descriptors
used to audit a protocol, not a validated fairness standard.

The notebook also covers the difference between the two canonical test splits
with paired bootstrap intervals, the sensitivity of the collapse criterion to
both of its margins, the XGBoost baselines, and the fairness–utility frontier
within each regime. The bootstrap resamples the test sets in a fixed order
(`RESAMPLE_ORDER` in `analysis/cells.py`), so the intervals do not depend on
the order in which the result files are read.

Those sections read one draw of the demonstrations, set by the notebook
variable `DEMO_SEED` in the setup cell (the environment variable of the same
name only affects `main.py`): seed 42 is read from `results/`, and seeds 1 and
8 read the conditions that receive demonstrations from `results_seed1/` and
`results_seed8/`. A further section takes the three draws together: the count
of every condition under each draw with its median and range, the status every
cell has under at least two of the three draws and the median of its metrics,
the agreement between draws, and intervals that pool the bootstrap replicates
of the three. It is skipped when the two directories are absent. The closing
summary gives the three audit levels under draws A, B and C with their median.

DI is undefined or infinite in a draw where the privileged group receives no
positive prediction. Across the three draws it is summarised as the median of
the finite ratios when at least two draws are finite, as infinite when at least
two are infinite, and as undefined otherwise.

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
flagged mentions of protected attributes in the archived D4b chains average at
most one per 1,000 chains after filtering out input echoes and explicit
instructions to ignore an attribute; that the
demonstrations are drawn balanced and reproducibly; and that the notebook reads
its gates from `analysis/cells.py` rather than redefining them.

The suite runs on CPU in a few seconds and downloads nothing: the checks that
need a GPU library are skipped when it is absent, and the two that read
`results/` are skipped until it is unzipped.

## Project structure

```
fairness-audit-llm-tabular/
├── main.py                  # Experiment entry point
├── analysis.ipynb           # Results analysis: the three audit levels, the difference between test sets and the frontier
├── requirements.txt
├── src/
│   ├── data_loader.py           # Dataset loading and preprocessing
│   ├── fair_data.py             # D2 (FLAI causal generation) and D3 (resampling)
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
├── results_seed1.zip        # Conditions with demonstrations, drawn with seed 1 (unzip to results_seed1/)
├── results_seed8.zip        # Conditions with demonstrations, drawn with seed 8 (unzip to results_seed8/)
├── results_summary.zip      # Cell tables of the three draws, their summary and the demonstrations (unzip to results_summary/)
├── results_checks.zip       # Validation responses of the zero-shot conditions and batch-size check (unzip to results_checks/)
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

`results.zip` holds draw A (seed 42) and the conditions without demonstrations.
`results_seed1.zip` and `results_seed8.zip` hold draws B (seed 1) and C
(seed 8) of the conditions that receive demonstrations, 384 passes each. They
unzip to `results_seed1/` and `results_seed8/`, apart from `results/`, with the
same layout and file schema and a `_seed<n>` tag in every filename. The three
draws share no demonstration within a dataset and demonstration source. The
test instances are the same in the three draws, so the predictions of a pass
can be compared row by row across draws. The section of the notebook on the
three draws needs both archives:

```bash
unzip results_seed1.zip
unzip results_seed8.zip
```

`results_summary.zip` holds the cell tables behind the counts of the paper:
one per draw, from which every count of cells is taken as the median of its
three per-draw values, and one for the three draws together, with the median of every
metric and the status every cell has under at least two of the three draws.
It also holds `demonstrations.csv`, the ten training rows of every dataset,
training split and draw. The result files do not store those rows:
`demos_sha256` in `_generation_stats.csv` identifies them for the few-shot
conditions (D1, D2, D3, D5 and D6), while for D4b_D1, D4b_D2, D4r_D1 and D4r_D2
it is the hash of an empty input and identifies nothing. The table is
reconstructed from the splits in `data/` with `select_fewshot_examples`
(`src/hf_classifier.py`), the function every condition calls with the seed of
the draw, and it reproduces the archived hash of every few-shot pass.

`results_checks.zip` holds a validation set of zero-shot responses collected
with the same prompts and inference code and kept as text, the request for the
label alone made to Phi-4 under `ZS_D5`, and two checks of GPT-OSS-20B: its
sensitivity to batch size, and its German Credit D1 passes under the three
draws in one software environment. Each archive carries its own `README.md`.

The paper counts 622 responses scored as the negative class by default, out of
660,000. They are the sum of two counts: `parse_failures_total` in the
`_generation_stats.csv` of every pass other than ZS_D1 and ZS_D5, over the
three draws (47), and `unread` in
`results_checks/zero_shot_responses/summary.json` (575). The ZS_D1 and ZS_D5
passes archive the predicted labels and not the responses, so their count is
read from the validation responses and not from the archived counter, which
records 235.

## Datasets

Downloaded automatically on first run: Adult Income and German Credit from the
UCI repository through `ucimlrepo`, and COMPAS from the ProPublica repository.

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
the training split alone, so no evaluation row participates in fitting,
discretisation or mitigation.

Every condition is evaluated twice: once on the D1 test split and once on the
D2 test split. The `_testD1` / `_testD2` filename suffix identifies the
evaluation distribution; the D1-test outputs of D1, D2, D3, ZS_D1 and ZS_D5
carry no suffix.

D1 and D2 test sets are not paired: D2 is an independent synthetic sample from
the mitigated causal model, not a row-wise counterfactual of D1. The difference
between the two test sets is read with that limit in mind.

## FLAI

D2 is generated with the **FLAI** library (González-Sendino et al., 2024), which learns a causal graph from the original data and generates a new dataset where the influence of protected attributes on the target variable is mitigated through causal intervention. D3 does not use FLAI: it oversamples the smaller of the two sex groups of the D1 training split until both have the same size, the resampling baseline of Liu et al. (2024).

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
               Collapse and Sensitivity to the Test Distribution in LLM Decision Systems},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.21068577},
  url       = {https://doi.org/10.5281/zenodo.21068577}
}
```

## License

This project is released under the BSD 3-Clause License. See [`LICENSE`](LICENSE).
