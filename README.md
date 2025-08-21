# Network Anomaly Detection (Flow Metadata)

A practical, notebook-driven workflow for exploring, benchmarking, and exporting classical ML models on a labeled network flow dataset (~126k rows, 44 columns). We focus on lightweight feature engineering, statistical validation, supervised benchmarks, and unsupervised baselines—no claims beyond what you can reproduce from the included notebooks.

## Overview

- **Data**: `Network_anomaly_data.csv` (≈125,973 rows, 44 columns)
- **Labels**: `attack` → derived `attack_binary` (Normal vs Attack) and `attack_family` (DoS, Probe, R2L, U2R, OtherAttack, Normal)
- **Feature engineering (light)**: log transforms for bytes/duration, simple totals/ratios
- **EDA**: distributions, categorical breakdowns, correlations, outliers, time parsing attempts
- **Statistical tests**: t-tests on log bytes, chi-square for protocol/service, logistic regression for `flag` and `urgent`
- **Supervised models**: Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, HistGradientBoosting, Linear SVC, MLP
- **Unsupervised (context)**: PCA/t-SNE visualization, KMeans/DBSCAN clustering, IsolationForest/LOF outlier scoring
- **Artifacts**: CV summaries, confusion matrix, pickled pipelines/models, cluster labels, metadata

> We did not build a streaming system, deep models on payloads, or drift monitoring in this project.

---

## Repo Structure

- `EDA_Network_Anomaly_Detection.ipynb` — EDA, distributions, categorical summaries, correlations, outliers, and statistical tests
- `Training_Network_Anomaly_Detection.ipynb` — preprocessing, supervised CV benchmarks, small grid searches, stacking, unsupervised exploration, artifact saving
- `requirements_u.txt` — main Python dependencies
- `models/` — saved pipelines and models (created by the training notebook)
- `ml_outputs/` — exported CSV summaries (created by the training notebook)

---

## Quickstart

### 1) Environment

```bash
# (Option A) venv
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements_u.txt
```

### 2) Data

- Place `Network_anomaly_data.csv` in the project root (or update `DATA_PATH` in both notebooks).

### 3) Reproduce

- Open and run the notebooks in order:
  1. `EDA_Network_Anomaly_Detection.ipynb`
  2. `Training_Network_Anomaly_Detection.ipynb`

Outputs will be written to:
- `ml_outputs/` (e.g., `supervised_cv_summary.csv`, `confusion_matrix__<Model>.csv`)
- `models/` (timestamped `.joblib` pipelines and unsupervised objects)

---

## What the EDA Notebook Does

- Normalizes column names to lowercase
- Identifies key columns if present: `srcbytes/src_bytes`, `dstbytes/dst_bytes`, `duration`, `protocol`, `service`, `flag`, `urgent`, `attack`
- Missing values overview
- Distributions:
  - `srcbytes`, `dstbytes`, `duration` (log-scaled due to heavy tails)
  - Additional numeric columns (capped number of plots)
- Categorical breakdowns: `protocol`, `service`, `flag`, and `attack`
- Correlation heatmap for numeric columns + top correlated pairs
- Outlier analysis via IQR (top columns by outlier %)
- Time parsing attempts for any timestamp-like columns; optional daily summaries

### Statistical Tests Implemented
- Two-sample t-tests on `log1p(srcbytes)` and `log1p(dstbytes)` for Attack vs Normal (with Cohen’s d)
- Chi-square tests with Cramér’s V:
  - `protocol` × `attack_binary`
  - Top-N `service` × `attack_binary`
- Logistic regression:
  - `flag` one-hot → binary attack (AUC + classification report)
  - `urgent`: univariate vs multivariate (controlling for log bytes/duration)

> Insert your observed statistics (t, p, Cohen’s d, Cramér’s V, AUCs) into your documentation or report as needed.

---

## What the Training Notebook Does

### Preprocessing
- `ColumnTransformer`:
  - Numeric: `SimpleImputer(strategy="median")` + `StandardScaler`
  - Categorical: `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`
- Stratified train/test split (75/25)
- Target mode: binary (Normal vs Attack)

### Supervised Benchmarks (5-fold CV on train)
- Models:
  - LogisticRegression (balanced)
  - DecisionTree (balanced)
  - RandomForest (balanced)
  - GradientBoosting
  - HistGradientBoosting
  - Linear SVC (probability=True, balanced) — optional subsample for speed
  - MLP (small network)
- Metrics: accuracy, precision_macro, recall_macro, f1_macro, ROC-AUC (binary)
- Produces `ml_outputs/supervised_cv_summary.csv`

### Hyperparameter Tuning (small grids)
- RandomForest: `n_estimators`, `max_depth`
- LogisticRegression: `C`

### Simple Stacking Ensemble
- Base: RandomForest, GradientBoosting, LogisticRegression
- Final: LogisticRegression
- Cross-validated ROC-AUC reported

### Unsupervised Exploration (contextual)
- PCA (2D/20D) and optional t-SNE visualization
- Clustering: KMeans and DBSCAN on PCA-20 (silhouette, DB-index, ARI/NMI vs labels)
- Outlier detection: IsolationForest, LocalOutlierFactor (AUC vs `attack_binary`)

### Artifact Export
- Best supervised pipeline (from CV ranking)
- All benchmarked supervised models
- Tuned RF/LR best estimators
- Stacking pipeline
- Unsupervised objects: preprocessing, PCA-20, KMeans, DBSCAN, IsolationForest, LOF
- `ml_outputs/`:
  - CV summary
  - Confusion matrix for selected best model on holdout test
  - Unsupervised cluster labels (if generated)
- `models/`:
  - Timestamped `.joblib` files for all the above
- Metadata JSON with feature lists and label distribution

---

## Using a Saved Model (Example)

```python
import joblib
from pathlib import Path
import pandas as pd

# Load latest "best" supervised pipeline
models_dir = Path("models")
best = sorted(models_dir.glob("*__supervised__best__*.joblib"))[-1]
pipe = joblib.load(best)
print("Loaded:", best.name)

# Predict on new data with the same schema as training X
X_new = pd.read_csv("Network_anomaly_data.csv")  # or your own data, same columns
y_pred = pipe.predict(X_new)
y_proba = pipe.predict_proba(X_new)[:, 1] if hasattr(pipe, "predict_proba") else None
```

---

## Configuration Knobs (in notebooks)

- `DATA_PATH` — CSV path
- `TARGET_MODE` — `"binary"`
- `TEST_SIZE` — `0.25`
- `N_SPLITS` — `5` for CV
- `SAMPLE_FOR_SVC` — optional subsampling to speed up linear SVC
- `TSNE_SAMPLE` — optional subsampling for t-SNE

---

## Results (Fill With Your Numbers)

- Supervised CV (top models): add mean ROC-AUC, F1, etc.
- Best RF/LR grid scores + best params
- Stacking ROC-AUC (CV)
- Unsupervised: KMeans/DBSCAN ARI/NMI; IsolationForest/LOF AUCs
- Holdout confusion matrix for selected best model

> Replace with metrics printed in the notebooks and CSVs saved in `ml_outputs/`.

---

## Troubleshooting

- numexpr warning during pandas ops: upgrade `numexpr` or ignore (benign for our workflow).
- t-SNE/plots run slow or OOM: reduce `TSNE_SAMPLE`; downsample for plotting.
- Sparse to dense conversion: only used for specific models (e.g., HistGradientBoosting/MLP) via a picklable `FunctionTransformer`.

---

## License

Add your license here (e.g., MIT).

## Citation

If you use this repo, please cite or reference the project page/Medium article.

## Acknowledgments

This project builds on standard scikit-learn workflows and classic KDD-style flow labeling conventions.
