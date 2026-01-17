# 03: Notebook Structure

## Goal

Define a minimal, reproducible baseline notebook that:
- loads Track A data (HF embeddings + Ludo's corrected metadata),
- trains LogReg with fixed hyperparameters,
- evaluates with StratifiedGroupKFold (no subject leakage),
- runs **cumulative horizon analyses** (≤3, ≤6, ≤12 months) using only samples collected up to that horizon,
- runs LOCO analysis (country confounding check),
- is structured to support future hyperparameter tuning without restructuring.

Notebook file: `notebooks/01_food_allergy_baseline.ipynb`

Scientific hypotheses and the horizon set are defined in `docs/specs/00_HYPOTHESIS.md`.

---

## Notebook TDD: Assertion-Driven Development

Traditional “tests-first” TDD does not map cleanly onto notebooks. For this project, the notebook itself is the primary runnable artifact, so we treat **inline assertions** as the notebook’s “tests”.

Requirements:
- Keep the notebook **self-contained** (no project-specific helper modules required).
- Put **configurable paths** and constants at the top.
- Use only standard scientific Python packages (`numpy`, `pandas`, `h5py`, `scikit-learn`, `pathlib`).
- Add `assert` checks at every critical step (shape, key alignment, no duplicates, label consistency, etc.).
- The notebook must run **top-to-bottom without edits**; if an invariant breaks, it should fail fast with a clear assertion message.

Transplantability target:
- Copy `notebooks/01_food_allergy_baseline.ipynb`
- Copy `data/processed/longitudinal_wgs_subset/`
- Copy `data/processed/hf_legacy/microbiome_embeddings_100d.h5`

That bundle should be sufficient to reproduce the baseline run on another machine with the same Python deps installed.

## Inputs

### Track A (Current Focus)

**Metadata:**
- `data/processed/longitudinal_wgs_subset/Month_*.csv`
- Columns: `sid, patient_id, country, label, allergen_class`

**Embeddings:**
- `data/processed/hf_legacy/microbiome_embeddings_100d.h5`
- Keys: SRS IDs (e.g., `SRS1719259`)
- Values: 100-dim float32 vectors

**Join key:** `sid` in CSVs = keys in H5 file

---

## Notebook Outline

```
0. Setup & Configuration
1. Load Data
2. Integrity Checks
3. ANALYSIS 1: Association Baseline (All Samples)
4. ANALYSIS 2: Horizon ≤3 months (Exploratory)
5. ANALYSIS 3: Horizon ≤6 months
6. ANALYSIS 4: Horizon ≤12 months
7. LOCO Analysis (per viable horizon)
8. Results Summary & Export
9. Reproducibility Footer
```

---

## Cell-by-Cell Specification

### 0) Setup & Configuration

```python
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import sys

# Configuration
RANDOM_SEED = 42
N_SPLITS_OUTER = 5
HORIZONS = [None, 3, 6, 12]  # None = all samples (association baseline)

# Paths
# Note: `jupyter nbconvert --execute` runs with cwd set to the notebook's directory,
# so auto-detect the repo root by searching for `data/processed/`.
def find_repo_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "data" / "processed").exists():
            return p
    raise FileNotFoundError(
        f"Could not find repo root from {start.resolve()} (expected data/processed/)"
    )

REPO_ROOT = find_repo_root(Path.cwd())
METADATA_DIR = REPO_ROOT / "data" / "processed" / "longitudinal_wgs_subset"
EMBEDDINGS_PATH = REPO_ROOT / "data" / "processed" / "hf_legacy" / "microbiome_embeddings_100d.h5"
RESULTS_DIR = REPO_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Set seeds for reproducibility
np.random.seed(RANDOM_SEED)

# Print versions
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"pandas: {pd.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"Random seed: {RANDOM_SEED}")
```

### 1) Load Data

```python
import h5py
import pandas as pd
from pathlib import Path

# Load all Month CSVs
dfs = []
for csv_path in sorted(METADATA_DIR.glob("Month_*.csv")):
    month = int(csv_path.stem.split("_")[1])
    df = pd.read_csv(csv_path)
    df["month"] = month
    dfs.append(df)
metadata = pd.concat(dfs, ignore_index=True)

# Load embeddings
embeddings_dict = {}
with h5py.File(EMBEDDINGS_PATH, "r") as f:
    for key in f.keys():
        embeddings_dict[key] = f[key][:]

# Merge
metadata["embedding"] = metadata["sid"].map(embeddings_dict)
df = metadata.dropna(subset=["embedding"])  # Should be 0 drops if aligned

# Extract arrays
X = np.stack(df["embedding"].values)
y = df["label"].values
patient_ids = df["patient_id"].values
countries = df["country"].values
```

### 2) Integrity Checks

```python
# Counts
print(f"Samples: {len(df)}")
print(f"Unique patients: {df['patient_id'].nunique()}")
print(f"Label distribution: {df['label'].value_counts().to_dict()}")
print(f"Country distribution: {df['country'].value_counts().to_dict()}")

# Assertions
assert X.shape == (785, 100), f"Expected (785, 100), got {X.shape}"
assert len(y) == 785
assert df["patient_id"].nunique() == 212

# Check: each sample in exactly one month
assert df["sid"].duplicated().sum() == 0, "Duplicate SRS IDs found!"

# Check: labels consistent per patient
labels_per_patient = df.groupby("patient_id")["label"].nunique()
assert (labels_per_patient == 1).all(), "Inconsistent labels within patient!"

# Horizon sanity checks (Track A; month derived from file name)
def horizon_counts(m: int):
    d = df[df["month"] <= m]
    return len(d), d["patient_id"].nunique(), d["label"].value_counts().to_dict(), d["country"].value_counts().to_dict()

print("Horizon counts:")
for m in [3, 6, 12]:
    n_samp, n_pat, labels, countries_m = horizon_counts(m)
    print(f"  month<={m}: samples={n_samp}, patients={n_pat}, labels={labels}, countries={countries_m}")

assert horizon_counts(3)[0] == 45
assert horizon_counts(6)[0] == 110
assert horizon_counts(12)[0] == 307
```

### 3) ANALYSES 1–4: Association Baseline + Cumulative Horizons (Subject-Level)

```python
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

outer_cv = StratifiedGroupKFold(n_splits=N_SPLITS_OUTER, shuffle=True, random_state=RANDOM_SEED)

def mean_embedding(vectors: pd.Series) -> np.ndarray:
    return np.mean(np.stack(vectors.to_list(), axis=0), axis=0)

def build_subject_table(samples_df: pd.DataFrame) -> pd.DataFrame:
    subj = samples_df.groupby("patient_id").agg(
        label=("label", "first"),
        country=("country", "first"),
        n_samples=("sid", "count"),
    )
    subj["embedding"] = samples_df.groupby("patient_id")["embedding"].apply(mean_embedding)
    return subj.reset_index()

def run_cv(subj_df: pd.DataFrame, horizon_label: str) -> list[dict]:
    X_subj = np.stack(subj_df["embedding"].to_list())
    y_subj = subj_df["label"].to_numpy()
    groups = subj_df["patient_id"].to_numpy()

    fold_rows = []
    for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X_subj, y_subj, groups=groups)):
        X_train, X_test = X_subj[train_idx], X_subj[test_idx]
        y_train, y_test = y_subj[train_idx], y_subj[test_idx]

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                C=1.0,
                class_weight='balanced',
                solver='lbfgs',
                max_iter=2000,
                random_state=RANDOM_SEED
            ))
        ])

        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)

        fold_rows.append({
            'horizon': horizon_label,
            'fold': fold_idx,
            'n_train_subjects': len(y_train),
            'n_test_subjects': len(y_test),
            'auroc': roc_auc_score(y_test, y_pred_proba),
            'auprc': average_precision_score(y_test, y_pred_proba),
            'f1': f1_score(y_test, y_pred),
        })
    return fold_rows

cv_results = []

for m in HORIZONS:
    if m is None:
        horizon_label = "all"
        df_h = df.copy()
    else:
        horizon_label = f"<= {m}mo"
        df_h = df[df["month"] <= m].copy()

    subj = build_subject_table(df_h)
    cv_results.extend(run_cv(subj, horizon_label=horizon_label))

cv_df = pd.DataFrame(cv_results)
print(cv_df)
```

### 4) LOCO: Leave-One-Country-Out (Per Horizon, Where Meaningful)

```python
loco_results = []

for m in HORIZONS:
    if m is None:
        horizon_label = "all"
        df_h = df.copy()
    else:
        horizon_label = f"<= {m}mo"
        df_h = df[df["month"] <= m].copy()

    subj = build_subject_table(df_h)
    X_subj = np.stack(subj["embedding"].to_list())
    y_subj = subj["label"].to_numpy()
    countries_subj = subj["country"].to_numpy()

    for held_out in ['FIN', 'EST', 'RUS']:
        train_mask = countries_subj != held_out
        test_mask = countries_subj == held_out

        y_test = y_subj[test_mask]
        # AUROC is undefined if held-out set has only one class
        if len(set(y_test)) < 2:
            continue

        X_train, y_train = X_subj[train_mask], y_subj[train_mask]
        X_test = X_subj[test_mask]

        model = Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(
                C=1.0,
                class_weight='balanced',
                solver='lbfgs',
                max_iter=2000,
                random_state=RANDOM_SEED
            ))
        ])

        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        loco_results.append({
            'horizon': horizon_label,
            'held_out': held_out,
            'n_train_subjects': len(y_train),
            'n_test_subjects': len(y_test),
            'auroc': roc_auc_score(y_test, y_pred_proba),
            'auprc': average_precision_score(y_test, y_pred_proba)
        })

loco_df = pd.DataFrame(loco_results)
print(loco_df)
```

### 5) Results Summary & Export

```python
# Save results
cv_df.to_csv(RESULTS_DIR / "cv_metrics.csv", index=False)
loco_df.to_csv(RESULTS_DIR / "loco_metrics.csv", index=False)

# Summary by horizon
summary_rows = []
for horizon_label, g in cv_df.groupby("horizon"):
    summary_rows.append({
        "horizon": horizon_label,
        "auroc_mean": g["auroc"].mean(),
        "auroc_std": g["auroc"].std(),
        "auprc_mean": g["auprc"].mean(),
        "auprc_std": g["auprc"].std(),
        "f1_mean": g["f1"].mean(),
        "f1_std": g["f1"].std(),
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(RESULTS_DIR / "cv_summary.csv", index=False)

print("Results saved to results/")
```

### 6) Reproducibility Footer

```python
from datetime import datetime

print("=" * 50)
print("REPRODUCIBILITY INFO")
print("=" * 50)
print(f"Random seed: {RANDOM_SEED}")
print(f"Outer CV splits: {N_SPLITS_OUTER}")
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"pandas: {pd.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"Run completed: {datetime.now().isoformat()}")
```

---

## Optional: Nested CV for Hyperparameter Tuning

**This is OPTIONAL. The baseline does NOT need this.**

If you later want to tune hyperparameters, use nested CV (inner loop inside outer loop). This is the only correct way to both tune and get unbiased performance estimates.

```python
from sklearn.model_selection import GridSearchCV

# Inside outer loop, on X_train, y_train:
inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
groups_train = patient_ids[train_idx]

param_grid = {'clf__C': [0.01, 0.1, 1.0, 10.0]}

grid_search = GridSearchCV(model, param_grid, cv=inner_cv, scoring='roc_auc')
grid_search.fit(X_train, y_train, groups=groups_train)  # groups= is CRITICAL

# Evaluate on outer test fold
y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
```

See `docs/specs/04_EVALUATION.md` for detailed explanation of nested CV.

---

## Reproducibility Requirements

- ✅ No network calls inside the notebook
- ✅ All data from `data/processed/`
- ✅ All randomness seeded (`RANDOM_SEED = 42`)
- ✅ Notebook runs top-to-bottom without edits
- ✅ Version info printed at end
