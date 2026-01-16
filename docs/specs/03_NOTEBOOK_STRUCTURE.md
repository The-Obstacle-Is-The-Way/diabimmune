# 03: Notebook Structure

## Goal

Define a minimal, reproducible notebook that:
- loads prepared 16S artifacts,
- trains a simple baseline (logistic regression),
- evaluates without subject leakage,
- optionally evaluates “predict by month m” without time leakage.

Notebook file: `notebooks/01_food_allergy_baseline.ipynb`

---

## Inputs (prepared artifacts)

From `scripts/prepare_16s_dataset.py`:
- `data/processed/16s/samples_food_allergy.csv`
- `data/processed/16s/otu_counts.npz`

---

## Notebook Outline

1. Setup & configuration (seeds, paths, versions)
2. Load data + integrity checks
3. Feature transforms (relative abundance, optional log1p)
4. Subject-level tables for time horizons (optional)
5. Model + evaluation (StratifiedGroupKFold)
6. Country generalization (leave-one-country-out)
7. Exports (metrics table + plots) + reproducibility footer

---

## Cell-by-Cell Skeleton (Key Cells)

### 0) Setup

- Print versions: Python, NumPy, pandas, scikit-learn
- Set `RANDOM_SEED = 42`
- Define:
  - `SAMPLES_CSV = data/processed/16s/samples_food_allergy.csv`
  - `COUNTS_NPZ = data/processed/16s/otu_counts.npz`

### 1) Load

- Load `samples = pd.read_csv(SAMPLES_CSV)`
- Load `npz = np.load(COUNTS_NPZ)`
- Assert:
  - `samples.shape[0] == npz["counts"].shape[0]`
  - `np.all(samples["sample_id"].to_numpy() == npz["sample_id"])`
  - `npz["counts"].shape[1] == 2005`
  - `samples["subject_id"].nunique()` matches expected (203 currently)

### 2) Features

- `X_counts = npz["counts"].astype(np.float32)`
- Convert to relative abundance:
  - `X = X_counts / X_counts.sum(axis=1, keepdims=True)`
  - Guard division by zero (should not happen, but assert).
- Optional: `X = np.log1p(X_counts)` or CLR transform (defer unless needed).

### 3) Subject-level aggregation by horizon (avoid time leakage)

If reporting “predict by month m”:
- For each horizon `m`, keep samples with `collection_month <= m`.
- Aggregate to **one row per subject**:
  - `mean` of feature vectors across samples up to `m` OR `last` available sample.

This prevents:
- time leakage (using future months),
- subject overweighting (many samples per infant).

### 4) Model

Pipeline:
- `StandardScaler(with_mean=True, with_std=True)`
- `LogisticRegression(class_weight="balanced", solver="liblinear", max_iter=2000)`

### 5) Evaluation

Use `StratifiedGroupKFold` with:
- `groups = subject_id`
- `y = label_food`

Compute:
- AUROC (primary)
- Precision/Recall/F1 at threshold 0.5 (secondary)
- Confusion matrix (normalized)

### 6) Country generalization

Leave-one-country-out:
- Train on two countries, test on held-out country.
- Report AUROC per held-out country per horizon.

---

## Reproducibility Requirements

- No network calls inside the notebook.
- All data inputs are read from `data/processed/16s/`.
- All randomness seeded (`random_state=RANDOM_SEED`).
- Notebook runs top-to-bottom without edits.
