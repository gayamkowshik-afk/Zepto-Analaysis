# Module 2 — Analytics Pipeline (`/analytics`)

## What this does

One cohesive pipeline over the Titanic dataset: profile it, clean it, tell a
visual data story about it (Part A), then build and rigorously evaluate a
full predictive-modeling pipeline on the same cleaned data (Part B).

## Install & run

```bash
pip install seaborn pandas numpy matplotlib scikit-learn imbalanced-learn joblib
python 01_eda.py        # loads data once, cleans it, produces the EDA story
python 02_modeling.py   # reads the cleaned data 01_eda.py produced, runs modeling
```

Run `01_eda.py` first — `02_modeling.py` depends on the `titanic_clean.csv`
it produces and does not load or clean data on its own.

## Data flow (load once, clean once)

- `01_eda.py` calls `sns.load_dataset('titanic')` **exactly once**, in the
  whole module. Immediately after loading, it saves the raw, as-loaded
  DataFrame to `titanic.csv` — an offline fallback so that if
  `sns.load_dataset` can't reach the network at grading time, the script
  transparently falls back to `pd.read_csv("titanic.csv")` and continues
  identically.
- `01_eda.py` then cleans that data once (Task 2) and performs the entire
  EDA story on the cleaned DataFrame, saving it to **`titanic_clean.csv`**
  at the end.
- `02_modeling.py` reads `titanic_clean.csv` directly. It never calls
  `sns.load_dataset` and never repeats the cleaning step — everything in
  Part B builds on that one cleaned dataset.

## Outputs

| File | Description |
|---|---|
| `titanic.csv` | Raw fallback, saved immediately after the one load |
| `titanic_clean.csv` | Cleaned data (produced once by `01_eda.py`) |
| `EDA_REPORT.md` | Profiling, missing-value decisions, outlier counts, skew analysis, bivariate breakdowns, correlation heatmap, 4 data-story charts with interpretations, standardization check |
| `MODELING_REPORT.md` | Split justification, all classifier/regression metrics, imbalance comparison, tuning results, final comparison table + recommendation |
| `figures/*.png` | All charts from both notebooks |
| `best_pipeline.joblib` | Complete fitted pipeline (preprocessing + tuned Random Forest), reloadable end-to-end on raw input |

## Key design decisions

- **Missing values** (threshold rule): `deck` (~77% missing) is dropped as a
  column — too sparse to impute reliably and not used downstream. `age`
  (~20% missing) is median-imputed. `embarked`/`embark_town` (~0.2% missing)
  have their few missing rows dropped.
- **Correlation matrix** uses exactly the 6 specified numeric columns
  (`survived, pclass, age, sibsp, parch, fare`), excluding the derived
  boolean flags `adult_male` and `alone`.
- **Modeling features**: `pclass, sex, age, sibsp, parch, fare, embarked`.
  Columns that leak the target or duplicate another feature
  (`alive`, `who`, `class`, `adult_male`, `alone`) are excluded.
- **Preprocessing** is a `ColumnTransformer` (StandardScaler on numeric
  columns, OneHotEncoder on categoricals) wrapped in a `Pipeline` with the
  estimator, fit only on the training split and applied transform-only to
  the test split — enforced structurally, not by hand.
- **Imbalance handling** and **hyperparameter tuning** are both run
  specifically on Random Forest, per the task spec, even though — on this
  particular train/test split — Logistic Regression turns out to edge it
  out on F1/AUC among the three baseline classifiers. The final written
  recommendation in `MODELING_REPORT.md` is generated from whichever model
  actually wins on the metrics, rather than being hardcoded.
- **Saved artifact**: `best_pipeline.joblib` is the tuned Random Forest's
  complete pipeline (preprocessor + estimator together), per Task 15's
  requirement — confirmed reloadable via `joblib.load` and correct on raw,
  unpreprocessed input.
