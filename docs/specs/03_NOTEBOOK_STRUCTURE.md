# 03: Notebook Structure

## Overview

Define the exact structure of `notebooks/01_baseline_classifier.ipynb`.

Key principle: the notebook consumes **prepared, validated** artifacts created by `scripts/prepare_data.py`:

- `data/processed/unified_samples.csv` (ground-truth sample table; includes `subject_id` and true `collection_month`)
- `data/processed/microbiome_embeddings_100d.h5` (one 100-d embedding per `srs_id`)

This keeps training/evaluation reproducible and avoids re-deriving IDs or labels inside the notebook.

---

## Notebook Principles

1. **Runs top-to-bottom** without edits
2. **No network required** for training/eval
3. **No leakage**: all evaluation is subject-level
4. **Time-aware**: avoid using samples collected after the prediction horizon
5. **Minimal + robust**: one baseline model, strong validation

---

## Notebook Outline

```
01_baseline_classifier.ipynb
├── 0. Setup & Configuration
├── 1. Load Prepared Data
├── 2. Build Time-Aware Subject Tables
├── 3. Logistic Regression + CV (StratifiedGroupKFold)
├── 4. Country Generalization (Leave-One-Country-Out)
├── 5. Visualizations + Exports
└── 6. Reproducibility Footer
```

---

## Cell-by-Cell Specification

### 0. Setup & Configuration

**Cell 0.1: Imports** (Code)
```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
```

**Cell 0.2: Paths + config** (Code)
```python
PROJECT_ROOT = Path.cwd().parent  # notebook lives in notebooks/

SAMPLES_CSV = PROJECT_ROOT / "data" / "processed" / "unified_samples.csv"
EMBED_H5 = PROJECT_ROOT / "data" / "processed" / "microbiome_embeddings_100d.h5"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
N_SPLITS_MAX = 5
TIME_HORIZONS_MONTHS = [1, 3, 6, 12, 18, 24]

for p in [SAMPLES_CSV, EMBED_H5]:
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run: python scripts/prepare_data.py (repo root).")

print("Python:", __import__("platform").python_version())
print("NumPy:", np.__version__)
print("Pandas:", pd.__version__)
print("scikit-learn:", sklearn.__version__)
```

**Cell 0.3: Outcome definition** (Markdown)
```markdown
**Baseline outcome (`label`)**: `1` if **any** `allergy_*` is true **or** `totalige_high` is true (from the RData ground truth; matches HuggingFace label).  
This is broader than “food allergy only”.
```

---

### 1. Load Prepared Data

**Cell 1.1: Load sample table** (Code)
```python
samples = pd.read_csv(SAMPLES_CSV)

assert len(samples) == 785
assert samples["srs_id"].is_unique
assert samples["subject_id"].nunique() == 212
assert set(samples["label"].unique()) <= {0, 1}

samples.head()
```

**Cell 1.2: Load embeddings into `X`** (Code)
```python
def load_embeddings_for_samples(samples_df: pd.DataFrame, h5_path: Path) -> np.ndarray:
    srs_list = samples_df["srs_id"].tolist()
    X = np.empty((len(srs_list), 100), dtype=np.float32)

    with h5py.File(h5_path, "r") as f:
        for i, srs in enumerate(srs_list):
            X[i] = np.asarray(f[srs], dtype=np.float32)

    return X


X = load_embeddings_for_samples(samples, EMBED_H5)
y = samples["label"].to_numpy(dtype=np.int64)
groups = samples["subject_id"].to_numpy(dtype=str)

print("X:", X.shape, X.dtype)
```

---

### 2. Build Time-Aware Subject Tables (Avoid Time Leakage)

This notebook evaluates “predict by month *m*” by aggregating *all* samples with `collection_month <= m` into **one row per subject**.

**Cell 2.1: Subject table builder** (Code)
```python
def build_subject_table(
    samples_df: pd.DataFrame,
    X_samples: np.ndarray,
    horizon_month: int,
    aggregation: str = "mean",  # "mean" or "last"
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    if aggregation not in {"mean", "last"}:
        raise ValueError("aggregation must be 'mean' or 'last'")

    mask = samples_df["collection_month"].astype(int) <= int(horizon_month)
    df = samples_df.loc[mask].copy()
    X = X_samples[mask.to_numpy()]
    df["_x_row"] = np.arange(len(df))

    df = df.sort_values(["subject_id", "collection_month", "_x_row"], kind="mergesort")

    rows: list[dict[str, object]] = []
    feats: list[np.ndarray] = []

    for subject_id, g in df.groupby("subject_id", sort=False):
        Xg = X[g["_x_row"].to_numpy()]

        feat = Xg.mean(axis=0) if aggregation == "mean" else Xg[-1]
        feats.append(feat.astype(np.float32))

        rows.append(
            {
                "subject_id": subject_id,
                "label": int(g["label"].iloc[0]),
                "country": g["country"].iloc[0],
                "n_samples_upto_horizon": int(len(g)),
                "max_month": int(g["collection_month"].max()),
            }
        )

    subj = pd.DataFrame(rows)
    X_subj = np.stack(feats)
    y_subj = subj["label"].to_numpy(dtype=np.int64)
    groups_subj = subj["subject_id"].to_numpy(dtype=str)

    return X_subj, y_subj, groups_subj, subj
```

---

### 3. Logistic Regression + CV (StratifiedGroupKFold)

**Cell 3.1: Model factory** (Code)
```python
def make_model() -> Pipeline:
    # liblinear is deterministic for this binary classification baseline
    clf = LogisticRegression(
        solver="liblinear",
        penalty="l2",
        C=1.0,
        max_iter=2000,
    )
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )
```

**Cell 3.2: Dynamic `n_splits`** (Code)
```python
def make_cv(y_subj: np.ndarray, n_splits_max: int, random_seed: int) -> StratifiedGroupKFold:
    n_pos = int((y_subj == 1).sum())
    n_neg = int((y_subj == 0).sum())
    n_splits = min(n_splits_max, n_pos, n_neg)
    if n_splits < 2:
        raise ValueError(f"Not enough subjects per class for CV: pos={n_pos} neg={n_neg}")
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
```

**Cell 3.3: Evaluate horizons** (Code)
```python
@dataclass(frozen=True)
class FoldResult:
    horizon_month: int
    fold: int
    auroc: float


def evaluate_horizon(horizon_month: int, aggregation: str = "mean") -> list[FoldResult]:
    X_subj, y_subj, groups_subj, _ = build_subject_table(samples, X, horizon_month, aggregation=aggregation)
    cv = make_cv(y_subj, N_SPLITS_MAX, RANDOM_SEED)

    out: list[FoldResult] = []
    for fold, (tr, te) in enumerate(cv.split(X_subj, y_subj, groups=groups_subj)):
        model = make_model()
        model.fit(X_subj[tr], y_subj[tr])
        prob = model.predict_proba(X_subj[te])[:, 1]
        out.append(FoldResult(horizon_month, fold, float(roc_auc_score(y_subj[te], prob))))

    return out


all_results: list[FoldResult] = []
for m in TIME_HORIZONS_MONTHS:
    try:
        all_results.extend(evaluate_horizon(m, aggregation="mean"))
    except ValueError as e:
        print(f"Skipping horizon={m}: {e}")

results_df = pd.DataFrame([r.__dict__ for r in all_results])
results_df.groupby("horizon_month")["auroc"].agg(["mean", "std", "count"]).reset_index()
```

---

### 4. Country Generalization (Leave-One-Country-Out)

**Cell 4.1: LOCO evaluation** (Code)
```python
def evaluate_leave_one_country_out(horizon_month: int, aggregation: str = "mean") -> pd.DataFrame:
    X_subj, y_subj, _, subj = build_subject_table(samples, X, horizon_month, aggregation=aggregation)

    rows: list[dict[str, object]] = []
    for held_out in sorted(subj["country"].unique()):
        train_mask = subj["country"] != held_out
        test_mask = subj["country"] == held_out

        # Require both classes in train/test
        if len(set(y_subj[train_mask.to_numpy()])) < 2:
            continue
        if len(set(y_subj[test_mask.to_numpy()])) < 2:
            continue

        model = make_model()
        model.fit(X_subj[train_mask.to_numpy()], y_subj[train_mask.to_numpy()])
        prob = model.predict_proba(X_subj[test_mask.to_numpy()])[:, 1]

        rows.append(
            {
                "horizon_month": horizon_month,
                "held_out_country": held_out,
                "n_test_subjects": int(test_mask.sum()),
                "auroc": float(roc_auc_score(y_subj[test_mask.to_numpy()], prob)),
            }
        )

    return pd.DataFrame(rows)


evaluate_leave_one_country_out(horizon_month=12)
```

---

### 5. Visualizations + Exports

**Cell 5.1: AUROC vs horizon** (Code)
```python
summary = results_df.groupby("horizon_month")["auroc"].agg(["mean", "std"]).reset_index()

fig, ax = plt.subplots(figsize=(7, 4))
ax.errorbar(summary["horizon_month"], summary["mean"], yerr=summary["std"], marker="o", capsize=4)
ax.axhline(0.5, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Prediction horizon (month)")
ax.set_ylabel("Subject-level AUROC")
ax.set_title("DIABIMMUNE baseline (LogReg on 100-d embeddings)")
ax.grid(alpha=0.3)
fig.savefig(RESULTS_DIR / "auroc_by_horizon.png", dpi=150, bbox_inches="tight")
```

**Cell 5.2: Save results tables** (Code)
```python
results_df.to_csv(RESULTS_DIR / "cv_results_by_horizon.csv", index=False)
summary.to_csv(RESULTS_DIR / "cv_summary_by_horizon.csv", index=False)
```

---

### 6. Reproducibility Footer

**Cell 6.1: Record runtime metadata** (Code)
```python
import json
import platform

payload = {
    "random_seed": RANDOM_SEED,
    "n_splits_max": N_SPLITS_MAX,
    "time_horizons_months": TIME_HORIZONS_MONTHS,
    "python": platform.python_version(),
    "numpy": np.__version__,
    "pandas": pd.__version__,
    "sklearn": sklearn.__version__,
}

(RESULTS_DIR / "run_metadata.json").write_text(json.dumps(payload, indent=2) + "\n")
payload
```
