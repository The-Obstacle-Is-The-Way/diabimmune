# 03: Notebook Structure

## Overview

Define the exact structure of `01_baseline_classifier.ipynb` with cell-by-cell breakdown, ensuring reproducibility and clean organization.

---

## Notebook Principles

1. **Linear execution**: Run top-to-bottom without errors
2. **Idempotent**: Can re-run any cell safely
3. **Self-documenting**: Markdown explains each section
4. **Reproducible**: All random seeds set, versions logged
5. **Type-hinted**: All functions have type annotations

---

## Notebook Outline

```
01_baseline_classifier.ipynb
├── 0. Setup & Configuration
├── 1. Data Loading
│   ├── 1.1 Download from HuggingFace
│   ├── 1.2 Load Metadata (RData)
│   └── 1.3 Explore Metadata Structure
├── 2. Data Processing
│   ├── 2.1 Load Embeddings & Labels
│   ├── 2.2 Create Unified Dataset
│   └── 2.3 Validate Data
├── 3. Model Training & Evaluation
│   ├── 3.1 Define Model Pipeline
│   ├── 3.2 Run Cross-Validation
│   └── 3.3 Aggregate Results
├── 4. Visualization
│   ├── 4.1 ROC Curves
│   ├── 4.2 Confusion Matrices
│   └── 4.3 Per-Month Analysis
├── 5. Results Summary
└── 6. Export Results
```

---

## Cell-by-Cell Specification

### Section 0: Setup & Configuration

**Cell 0.1: Imports** (Code)
```python
"""Imports and setup."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyreadr
import seaborn as sns
from huggingface_hub import snapshot_download
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)
```

**Cell 0.2: Configuration** (Code)
```python
"""Configuration constants."""

# Paths
PROJECT_ROOT = Path.cwd().parent  # Assumes notebook in notebooks/
DATA_DIR = PROJECT_ROOT / "data" / "huggingface" / "AI4FA-Diabimmune"
RDATA_PATH = PROJECT_ROOT / "DIABIMMUNE_Karelia_metadata.RData"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories
DATA_DIR.parent.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Experiment settings
RANDOM_SEED = 42
N_SPLITS = 5
MONTHS_TO_EVALUATE = [1, 2, 3, 6, 12, 24, 36]  # Key timepoints

# Reproducibility
np.random.seed(RANDOM_SEED)

print(f"Project root: {PROJECT_ROOT}")
print(f"Data dir: {DATA_DIR}")
print(f"RData path: {RDATA_PATH} (exists: {RDATA_PATH.exists()})")
```

**Cell 0.3: Version Logging** (Code)
```python
"""Log package versions for reproducibility."""
import sklearn
import platform

print(f"Python: {platform.python_version()}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")
print(f"Scikit-learn: {sklearn.__version__}")
print(f"Platform: {platform.platform()}")
```

---

### Section 1: Data Loading

**Cell 1.0: Markdown Header**
```markdown
# 1. Data Loading

Load embeddings from HuggingFace and subject metadata from RData file.
```

**Cell 1.1: Download from HuggingFace** (Code)
```python
"""Download dataset from HuggingFace (skip if already exists)."""

if not DATA_DIR.exists() or not any(DATA_DIR.iterdir()):
    print("Downloading from HuggingFace...")
    snapshot_download(
        repo_id="hugging-science/AI4FA-Diabimmune",
        repo_type="dataset",
        local_dir=DATA_DIR,
        local_dir_use_symlinks=False,
    )
    print("Download complete!")
else:
    print(f"Dataset already exists at {DATA_DIR}")

# Verify structure
print("\nDataset contents:")
for p in sorted(DATA_DIR.rglob("*"))[:20]:
    print(f"  {p.relative_to(DATA_DIR)}")
```

**Cell 1.2: Load Metadata** (Code)
```python
"""Load subject metadata from RData file."""

result = pyreadr.read_r(RDATA_PATH)

print(f"Objects in RData: {list(result.keys())}")

# Get the dataframe
metadata_key = list(result.keys())[0]
metadata_df = result[metadata_key]

print(f"\nLoaded '{metadata_key}' with shape {metadata_df.shape}")
```

**Cell 1.3: Explore Metadata** (Code)
```python
"""Explore metadata structure to find required columns."""

print("=== Column Summary ===\n")
for col in metadata_df.columns:
    dtype = metadata_df[col].dtype
    n_unique = metadata_df[col].nunique()
    n_missing = metadata_df[col].isna().sum()
    sample = metadata_df[col].dropna().iloc[0] if n_missing < len(metadata_df) else "N/A"
    print(f"{col}:")
    print(f"  dtype={dtype}, unique={n_unique}, missing={n_missing}")
    print(f"  sample: {sample}\n")
```

**Cell 1.4: Identify Key Columns** (Code)
```python
"""Identify column names for mapping.

After running Cell 1.3, update these based on actual column names.
"""

# TODO: Update these after exploring metadata
SAMPLE_ID_COL = "sampleID"        # Column containing SRS IDs
SUBJECT_ID_COL = "host_subject_id"  # Column containing infant IDs
COUNTRY_COL = "country"           # Column containing country

# Verify columns exist
for col_name, col in [
    ("SAMPLE_ID_COL", SAMPLE_ID_COL),
    ("SUBJECT_ID_COL", SUBJECT_ID_COL),
    ("COUNTRY_COL", COUNTRY_COL),
]:
    if col in metadata_df.columns:
        print(f"✅ {col_name}='{col}' found")
    else:
        print(f"❌ {col_name}='{col}' NOT FOUND - update variable!")
```

---

### Section 2: Data Processing

**Cell 2.0: Markdown Header**
```markdown
# 2. Data Processing

Load embeddings, merge with metadata, create training datasets.
```

**Cell 2.1: Define Data Classes** (Code)
```python
"""Data structures for the pipeline."""


@dataclass
class MonthDataset:
    """Dataset for a single month."""

    month: int
    X: np.ndarray  # (n_samples, 100)
    y: np.ndarray  # (n_samples,)
    sample_ids: list[str]
    subject_ids: list[str]
    countries: list[str]

    def __repr__(self) -> str:
        n_allergic = int(self.y.sum())
        n_healthy = len(self.y) - n_allergic
        return (
            f"MonthDataset(month={self.month}, "
            f"samples={len(self.y)}, "
            f"subjects={len(set(self.subject_ids))}, "
            f"allergic={n_allergic}, healthy={n_healthy})"
        )
```

**Cell 2.2: Data Loading Functions** (Code)
```python
"""Functions to load embeddings and labels."""


def load_embeddings(month: int) -> dict[str, np.ndarray]:
    """Load microbiome embeddings for a month."""
    h5_path = DATA_DIR / f"processed/microbiome_embeddings/month_{month}/microbiome_embeddings.h5"

    if not h5_path.exists():
        raise FileNotFoundError(f"No embeddings for month {month}: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        return {sid: np.array(f[sid]) for sid in f.keys()}


def load_labels(month: int) -> pd.DataFrame:
    """Load labels for a month."""
    csv_path = DATA_DIR / f"metadata/Month_{month}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"No labels for month {month}: {csv_path}")

    return pd.read_csv(csv_path)
```

**Cell 2.3: Create Unified Dataset** (Code)
```python
"""Create unified dataset with embeddings + metadata."""


def create_dataset(month: int) -> MonthDataset | None:
    """Create dataset for a specific month."""
    try:
        embeddings = load_embeddings(month)
        labels_df = load_labels(month)
    except FileNotFoundError as e:
        print(f"⚠️ Skipping month {month}: {e}")
        return None

    # Merge labels with metadata
    merged = labels_df.merge(
        metadata_df[[SAMPLE_ID_COL, SUBJECT_ID_COL, COUNTRY_COL]],
        left_on="sid",
        right_on=SAMPLE_ID_COL,
        how="inner",
    )

    # Filter to samples with embeddings
    merged = merged[merged["sid"].isin(embeddings.keys())]

    if len(merged) == 0:
        print(f"⚠️ No matching samples for month {month}")
        return None

    return MonthDataset(
        month=month,
        X=np.array([embeddings[sid] for sid in merged["sid"]]),
        y=merged["label"].values,
        sample_ids=merged["sid"].tolist(),
        subject_ids=merged[SUBJECT_ID_COL].tolist(),
        countries=merged[COUNTRY_COL].tolist(),
    )
```

**Cell 2.4: Load All Months** (Code)
```python
"""Load datasets for all target months."""

datasets: dict[int, MonthDataset] = {}

for month in MONTHS_TO_EVALUATE:
    ds = create_dataset(month)
    if ds is not None:
        datasets[month] = ds
        print(f"✅ {ds}")

print(f"\nLoaded {len(datasets)} months: {list(datasets.keys())}")
```

**Cell 2.5: Data Validation** (Code)
```python
"""Validate all datasets."""


def validate_dataset(ds: MonthDataset) -> bool:
    """Validate a dataset, return True if valid."""
    errors = []

    if ds.X.shape[1] != 100:
        errors.append(f"Expected 100-dim, got {ds.X.shape[1]}")

    if np.isnan(ds.X).any():
        errors.append("NaN in embeddings")

    if set(ds.y) - {0, 1}:
        errors.append(f"Non-binary labels: {set(ds.y)}")

    if any(s is None for s in ds.subject_ids):
        errors.append("Missing subject IDs")

    if errors:
        print(f"❌ Month {ds.month}: {errors}")
        return False

    print(f"✅ Month {ds.month} valid")
    return True


all_valid = all(validate_dataset(ds) for ds in datasets.values())
print(f"\nAll datasets valid: {all_valid}")
```

---

### Section 3: Model Training & Evaluation

**Cell 3.0: Markdown Header**
```markdown
# 3. Model Training & Evaluation

Train LogisticRegression with StratifiedGroupKFold cross-validation.
```

**Cell 3.1: Define Model** (Code)
```python
"""Define the classifier pipeline."""


def create_model() -> Pipeline:
    """Create a fresh model instance."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_SEED,
            solver="lbfgs",
        )),
    ])
```

**Cell 3.2: Evaluation Function** (Code)
```python
"""Cross-validation evaluation function."""


@dataclass
class EvalResult:
    """Results from cross-validation."""

    month: int
    auroc_mean: float
    auroc_std: float
    f1_mean: float
    f1_std: float
    fold_aurocs: list[float]
    fold_f1s: list[float]
    y_true_all: np.ndarray
    y_prob_all: np.ndarray


def evaluate_month(ds: MonthDataset) -> EvalResult:
    """Run StratifiedGroupKFold CV on a month's data."""
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

    fold_aurocs: list[float] = []
    fold_f1s: list[float] = []
    y_true_all: list[np.ndarray] = []
    y_prob_all: list[np.ndarray] = []

    groups = np.array(ds.subject_ids)

    for fold, (train_idx, test_idx) in enumerate(cv.split(ds.X, ds.y, groups=groups)):
        X_train, X_test = ds.X[train_idx], ds.X[test_idx]
        y_train, y_test = ds.y[train_idx], ds.y[test_idx]

        model = create_model()
        model.fit(X_train, y_train)

        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        auroc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)

        fold_aurocs.append(auroc)
        fold_f1s.append(f1)
        y_true_all.append(y_test)
        y_prob_all.append(y_prob)

    return EvalResult(
        month=ds.month,
        auroc_mean=np.mean(fold_aurocs),
        auroc_std=np.std(fold_aurocs),
        f1_mean=np.mean(fold_f1s),
        f1_std=np.std(fold_f1s),
        fold_aurocs=fold_aurocs,
        fold_f1s=fold_f1s,
        y_true_all=np.concatenate(y_true_all),
        y_prob_all=np.concatenate(y_prob_all),
    )
```

**Cell 3.3: Run Evaluation** (Code)
```python
"""Evaluate all months."""

results: dict[int, EvalResult] = {}

for month, ds in datasets.items():
    print(f"Evaluating month {month}...", end=" ")
    result = evaluate_month(ds)
    results[month] = result
    print(f"AUROC={result.auroc_mean:.3f}±{result.auroc_std:.3f}")

print("\n✅ Evaluation complete")
```

---

### Section 4: Visualization

**Cell 4.0: Markdown Header**
```markdown
# 4. Visualization

ROC curves, confusion matrices, and per-month analysis.
```

**Cell 4.1: Results Table** (Code)
```python
"""Create summary table."""

summary_df = pd.DataFrame([
    {
        "Month": r.month,
        "AUROC": f"{r.auroc_mean:.3f} ± {r.auroc_std:.3f}",
        "F1": f"{r.f1_mean:.3f} ± {r.f1_std:.3f}",
        "Samples": len(datasets[r.month].y),
        "Subjects": len(set(datasets[r.month].subject_ids)),
    }
    for r in results.values()
])

summary_df = summary_df.sort_values("Month")
display(summary_df)
```

**Cell 4.2: AUROC Over Time** (Code)
```python
"""Plot AUROC across months."""

fig, ax = plt.subplots(figsize=(10, 5))

months = sorted(results.keys())
aurocs = [results[m].auroc_mean for m in months]
stds = [results[m].auroc_std for m in months]

ax.errorbar(months, aurocs, yerr=stds, marker="o", capsize=5, linewidth=2)
ax.axhline(0.5, color="red", linestyle="--", label="Random (0.5)")
ax.set_xlabel("Month")
ax.set_ylabel("AUROC")
ax.set_title("Food Allergy Prediction: AUROC Over Time")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "auroc_over_time.png", dpi=150)
plt.show()
```

**Cell 4.3: ROC Curves** (Code)
```python
"""Plot ROC curves for each month."""

n_months = len(results)
fig, axes = plt.subplots(2, (n_months + 1) // 2, figsize=(15, 10))
axes = axes.flatten()

for ax, month in zip(axes, sorted(results.keys())):
    r = results[month]
    RocCurveDisplay.from_predictions(
        r.y_true_all,
        r.y_prob_all,
        ax=ax,
        name=f"Month {month}",
    )
    ax.set_title(f"Month {month}\nAUROC={r.auroc_mean:.3f}")
    ax.grid(True, alpha=0.3)

# Hide unused axes
for ax in axes[n_months:]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "roc_curves.png", dpi=150)
plt.show()
```

**Cell 4.4: Confusion Matrices** (Code)
```python
"""Plot confusion matrices for each month."""

fig, axes = plt.subplots(2, (n_months + 1) // 2, figsize=(15, 10))
axes = axes.flatten()

for ax, month in zip(axes, sorted(results.keys())):
    r = results[month]
    y_pred = (r.y_prob_all >= 0.5).astype(int)

    ConfusionMatrixDisplay.from_predictions(
        r.y_true_all,
        y_pred,
        ax=ax,
        cmap="Blues",
        normalize="true",
    )
    ax.set_title(f"Month {month}")

for ax in axes[n_months:]:
    ax.set_visible(False)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrices.png", dpi=150)
plt.show()
```

---

### Section 5: Results Summary

**Cell 5.0: Markdown Header**
```markdown
# 5. Results Summary

Key findings and limitations.
```

**Cell 5.1: Key Findings** (Code)
```python
"""Summarize key findings."""

best_month = max(results.keys(), key=lambda m: results[m].auroc_mean)
best_result = results[best_month]

print("=" * 50)
print("RESULTS SUMMARY")
print("=" * 50)
print(f"\nBest performing month: {best_month}")
print(f"  AUROC: {best_result.auroc_mean:.3f} ± {best_result.auroc_std:.3f}")
print(f"  F1:    {best_result.f1_mean:.3f} ± {best_result.f1_std:.3f}")

print("\nAll months:")
for month in sorted(results.keys()):
    r = results[month]
    above_random = "✓" if r.auroc_mean > 0.5 else "✗"
    print(f"  Month {month:2d}: AUROC={r.auroc_mean:.3f} {above_random}")

print("\nMethodology:")
print(f"  Cross-validation: StratifiedGroupKFold (n={N_SPLITS})")
print(f"  Grouping: Subject-level (no data leakage)")
print(f"  Model: LogisticRegression (class_weight='balanced')")
```

---

### Section 6: Export Results

**Cell 6.0: Export** (Code)
```python
"""Export results to files."""

# Save summary CSV
summary_df.to_csv(RESULTS_DIR / "summary.csv", index=False)
print(f"Saved: {RESULTS_DIR / 'summary.csv'}")

# Save detailed results
detailed = []
for month, r in results.items():
    for fold, (auroc, f1) in enumerate(zip(r.fold_aurocs, r.fold_f1s)):
        detailed.append({
            "month": month,
            "fold": fold,
            "auroc": auroc,
            "f1": f1,
        })

pd.DataFrame(detailed).to_csv(RESULTS_DIR / "detailed_results.csv", index=False)
print(f"Saved: {RESULTS_DIR / 'detailed_results.csv'}")

print("\n✅ All results exported")
```

---

## Verification Checklist

- [ ] Notebook runs top-to-bottom without errors
- [ ] All cells have type hints
- [ ] All random seeds are set
- [ ] Package versions logged
- [ ] Results saved to `results/` directory
- [ ] Figures are publication-quality
- [ ] No data leakage (subject-level splits verified)
