# CI speech outcome prediction from routine clinical data

This repository contains the clean, paper-facing analysis code CI outcome prediction study.

The analyses address three questions:

1. **Q1 — Predictive performance:** Can postoperative CI speech understanding be predicted from routinely available clinical data?
2. **Q2 — Predictors and interpretability:** Which variables have the highest predictive value, and in which direction?
3. **Q3 — Hoppe benchmark:** How do the models compare with the previously published model by Hoppe on the 6-month endpoint and on the 12/24-month study endpoint?

The raw clinical dataset is **not included** for data protection reasons.

## Repository structure

```text
ci-speech-outcome-prediction/
├── README.md
├── requirements.txt
├── config.yaml
├── src/
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── metrics.py
│   ├── models.py
│   ├── evaluation.py
│   ├── q1_predictive_performance.py
│   ├── q1_ablation.py
│   ├── q2_feature_importance.py
│   ├── q3_hoppe_comparison.py
│   ├── reporting.py
│   └── make_all.py
├── data/
│   └── README.md
├── results/
│   └── README.md

```

## Setup

Create a fresh Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

Place the local clinical Excel file at the path specified in `config.yaml`, for example:

```text
data/raw/dataset.xlsx
```

Then run all analyses:

```bash
python src/make_all.py --config config.yaml --out runs/essen_primary
```

The command creates a complete results bundle:

```text
runs/essen_primary/
├── results_summary.md
├── run_manifest.json
├── config_used.yaml
├── environment.txt
├── tables/
├── figures/
└── predictions/
```

Zip this folder for sharing and review:

```bash
zip -r essen_primary_results.zip runs/essen_primary
```

## Repeated-CV sensitivity analysis

To examine how strongly the regression and ablation results depend on the
five-fold partition, repeat the analyses across split seeds 42 through 61:

```bash
python src/repeated_cv_sensitivity.py \
  --config config.yaml \
  --out runs/essen_primary_repeated_cv \
  --seeds 42:61
```

By default, the script evaluates ElasticNet and MLP. The outer split seed
changes between repetitions, while the model seed remains fixed at the
`random_seed` in the config. This isolates sensitivity to fold allocation as
far as possible. The output bundle contains a human-readable
`results_summary.md`, the fold MAEs, one result per seed, an across-repetition
CSV summary, the exact config, a manifest, and the Python environment.

Alternative seeds or models can be supplied explicitly, for example:

```bash
python src/repeated_cv_sensitivity.py --seeds 1,7,21 --models ElasticNet MLP
```


## Data availability

The clinical raw data are not included in this repository due to patient privacy and institutional data protection requirements.

The analysis code expects a local Excel file with the same column structure as described in `data/README.md`. The data path can be configured in `config.yaml`.

Generated results, prediction files, and run bundles are also not tracked in Git. They should be stored in a protected institutional project folder.


## Main outputs

| Output | Description |
|---|---|
| `results_summary.md` | Compact auto-generated summary of the run, core tables, and best models |
| `tables/table_1_cohort.csv` | Cohort and endpoint availability |
| `tables/table_2_q1_regression.csv` | Q1 regression model performance |
| `tables/table_3_q1_classification.csv` | Q1 responder/non-responder classification |
| `tables/table_4_q1_ablation.csv` | Complete-case and Tragezeit ablations |
| `tables/table_5_q3_hoppe_comparison.csv` | Hoppe comparison for 6-month and 12/24-month endpoints |
| `tables/table_s3_feature_importance.csv` | Q2 variable-level importance |
| `tables/table_s4_elasticnet_feature_coefficients.csv` | Q2 feature-level ElasticNet coefficients |
| `tables/table_s5_feature_importance_with_wearing.csv` | Q2 variable-level importance with wearing time |
| `figures/` | Manuscript and supplement figures |
| `predictions/` | Fold-wise predictions for internal checking |
