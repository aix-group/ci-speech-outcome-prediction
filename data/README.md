# Data description

This repository does **not** contain clinical raw data.

The analyses require a local Excel file containing cochlear implant outcome data with the same column structure as the study dataset. The raw clinical data cannot be shared publicly due to patient privacy and institutional data protection requirements.

The code is provided to reproduce the analyses from a local, access-controlled copy of the dataset.

---

## Required input file

The analysis expects an Excel file with one row per implanted case.

The local file path and sheet name are configured in `config.yaml`, for example:

```yaml
data_path: data/raw/dataset.xlsx
sheet_name: Sheet1
```

The folder `data/raw/` is intentionally excluded from Git tracking and must not be committed.

---

## Cohort selection

Only the Essen subset is analysed.

The code filters the dataset using the `Klinik` column:

```python
df = df_raw[
    df_raw["Klinik"].astype(str).str.strip().str.casefold() == "essen"
].copy()
```

General exclusions for the main analyses:

- age below 18 years
- documented language barrier, if available/configured in the dataset

Additional Hoppe-specific inclusion and exclusion criteria are applied in the Hoppe comparison analysis according to the published Hoppe model and the available Essen variables.

---

## Primary endpoints

### Regression endpoint

```text
EV65CI_24_alt_12
```

This is the primary regression target.

It represents postoperative Freiburg monosyllable word recognition at 65 dB with CI:

- preferably at 24 months, if available
- otherwise at 12 months

Predictions are clipped to the clinically possible range of 0–100% and rounded to 5% steps.

### Classification endpoint

```text
EV65_CI_größer_gleich_25
```

This is the primary classification target.

It defines responder status using the 25% threshold:

- `0`: non-responder
- `1`: responder

The classification analysis uses the existing dataset variable `EV65_CI_größer_gleich_25`.

### Hoppe comparison endpoints

The Hoppe comparison uses two endpoints:

```text
EV65CI_6_Monate
EV65CI_24_alt_12
```

`EV65CI_6_Monate` is used for direct comparison with the published Hoppe model.  
`EV65CI_24_alt_12` is used to evaluate transfer of the Hoppe approach to the primary 12/24-month endpoint.

---

## Published Hoppe model variables

The published Hoppe model uses three preoperative variables.

| Published Hoppe variable | Essen variable used here |
|---|---|
| age at implantation | `Alter_OP` |
| preoperative maximum word recognition, WRSmax | `EVmaxLL_prä` |
| aided word recognition at 65 dB, WRS65(HA) | `EV65HG_prä` |

The fixed published formula is implemented as:

```text
prediction = 100 / (1 + exp(-(β0 + β1 * WRSmax + β2 * age + β3 * WRS65(HA))))
```

with:

```text
β0 = 0.84
β1 = 0.012
β2 = -0.0094
β3 = 0.0059
```

---

## Main predictor set

The main prediction models use routinely available preoperative or perioperative clinical variables.

The final main predictor set is:

```text
Alter_OP
Geschlecht
Seite
Zeitpunkt_HV
Beginn_HHV
Beginn_HV
Ursache_Transformiert
SSD
Versorgung_Gegenohr
HG_Nutzung
Elektrodenform
PTA4_LL_prä
EVmaxLL_prä
EV65HG_prä
```

### Predictor groups

| Group | Variables |
|---|---|
| Demographics | `Alter_OP`, `Geschlecht` |
| Side and symptoms | `Seite`|
| Hearing history | `Zeitpunkt_HV`, `Beginn_HV`, `Beginn_HHV`|
| Aetiology | `Ursache_Transformiert` |
| Contralateral-ear situation | `SSD`, `Versorgung_Gegenohr`, `HG_Nutzung` |
| Surgical/perioperative variables | `Elektrodenform` |
| Preoperative audiometry and speech audiometry | `PTA4_LL_prä`, `EVmaxLL_prä`, `EV65HG_prä` |

---

## Variables excluded from the main model

The following variables are not part of the primary preoperative/perioperative model:

```text
Mittlere_Tragezeit
Mittlere_Tragezeit_größer_7
Mittlere_Tragezeit_größer_12
Hersteller
Implantattyp
Hörminderung_Gegenohr
Klickbera
Bera_4kHz
Tinnitus
Schwindel
OAE
Cochleazugang
Radikalhöhlenanlage
```

Reasons:

- `Mittlere_Tragezeit` is postoperative and is therefore used only in a separate ablation analysis.
- `Hersteller` and `Implantattyp` may encode centre-specific treatment patterns and are not part of the main generalisable model.
- `Hörminderung_Gegenohr` is not used because the contralateral-ear situation is already represented by `SSD`, `Versorgung_Gegenohr`, and `HG_Nutzung`.

---

## Ablation variable

The postoperative wearing-time variable is used only in the dedicated ablation analysis:

```text
Mittlere_Tragezeit
```

This analysis evaluates how much postoperative usage information improves prediction. It is not interpreted as part of a purely preoperative prognostic model.

---

## Expected core columns

The required structure is documented in `data_schema.csv`.

The schema file lists:

- column name
- role in the analysis
- approximate data type
- short description
- whether the variable is part of the main model

Other dataset columns may be present, but are not necessarily used in the final analyses.

---

## Missing data handling

The main analysis uses imputation within the cross-validation pipeline.

- numeric variables: median imputation
- categorical variables: most frequent category imputation
- categorical encoding: one-hot encoding
- numeric scaling: applied for ElasticNet and MLP

A complete-case analysis is performed as an ablation.

---

## Data privacy and Git tracking

Do **not** commit any raw, processed, or patient-level data to GitHub.

The following files and folders must remain local or on a protected institutional project drive:

```text
data/raw/
data/processed/
runs/
results/
*.xlsx
*.xls
*.zip
```

The repository should contain only:

- analysis code
- configuration templates
- documentation
- this data description
- `data_schema.csv`, which contains column-level metadata only

Generated result bundles, prediction files, and manuscript result tables should be stored outside GitHub in a protected project folder.

---

## Reproducibility note

To reproduce the analyses, place the local clinical dataset in the configured path, verify the column names, and run:

```bash
python src/make_all.py --config config.yaml --out runs/essen_primary
```

The command creates a run folder with generated tables, figures, predictions, a run manifest, and `results_summary.md`.

Because the clinical dataset is not public, full numerical reproduction is only possible for authorised users with access to the protected raw data.
