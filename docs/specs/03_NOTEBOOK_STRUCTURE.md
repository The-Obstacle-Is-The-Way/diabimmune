# 03: Notebook Structure

## Goal

Define a minimal, reproducible baseline notebook that:
- loads Track A data (HF embeddings + Ludo's corrected metadata),
- trains LogReg with fixed hyperparameters,
- evaluates with StratifiedGroupKFold (no subject leakage),
- runs LOCO analysis (country confounding check),
- is structured to support future hyperparameter tuning without restructuring.

Notebook file: `notebooks/01_food_allergy_baseline.ipynb`

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
3. Outer CV: StratifiedGroupKFold Evaluation
4. LOCO: Leave-One-Country-Out Analysis
5. Results Summary & Export
6. Reproducibility Footer
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

# Paths
METADATA_DIR = Path("data/processed/longitudinal_wgs_subset")
EMBEDDINGS_PATH = Path("data/processed/hf_legacy/microbiome_embeddings_100d.h5")
RESULTS_DIR = Path("results")
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
```

### 3) Outer CV: StratifiedGroupKFold Evaluation

```python
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

outer_cv = StratifiedGroupKFold(n_splits=N_SPLITS_OUTER, shuffle=True, random_state=RANDOM_SEED)

cv_results = []

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=patient_ids)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Fixed hyperparameters - no tuning, no inner loop needed
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

    cv_results.append({
        'fold': fold_idx,
        'n_train': len(y_train),
        'n_test': len(y_test),
        'auroc': roc_auc_score(y_test, y_pred_proba),
        'auprc': average_precision_score(y_test, y_pred_proba),
        'f1': f1_score(y_test, y_pred)
    })

cv_df = pd.DataFrame(cv_results)
print(cv_df)
print(f"\nMean AUROC: {cv_df['auroc'].mean():.3f} ± {cv_df['auroc'].std():.3f}")
print(f"Mean AUPRC: {cv_df['auprc'].mean():.3f} ± {cv_df['auprc'].std():.3f}")
print(f"Mean F1:    {cv_df['f1'].mean():.3f} ± {cv_df['f1'].std():.3f}")
```

### 4) LOCO: Leave-One-Country-Out Analysis

```python
loco_results = []

for held_out in ['FIN', 'EST', 'RUS']:
    train_mask = countries != held_out
    test_mask = countries == held_out

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

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
        'held_out': held_out,
        'n_train': len(y_train),
        'n_test': len(y_test),
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

# Summary
summary = {
    'metric': ['AUROC', 'AUPRC', 'F1'],
    'mean': [cv_df['auroc'].mean(), cv_df['auprc'].mean(), cv_df['f1'].mean()],
    'std': [cv_df['auroc'].std(), cv_df['auprc'].std(), cv_df['f1'].std()]
}
summary_df = pd.DataFrame(summary)
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
