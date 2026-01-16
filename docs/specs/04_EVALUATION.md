# 04: Evaluation Strategy

## Goal

Scientifically valid evaluation for a longitudinal cohort:
- no **subject leakage**,
- no **time leakage** (when claiming early prediction),
- robust to **class imbalance**,
- explicit about **country confounding**.

Primary dataset: `data/processed/16s/*` (food allergies only).

---

## Leakage Risks

### Subject leakage (must prevent)

Multiple samples per infant mean random sample-splitting is invalid.

Fix:
- `groups = subject_id`
- split with `StratifiedGroupKFold`

### Time leakage (must prevent for “predict by month m”)

If the claim is “predict by month m”, you must not use samples with `collection_month > m` anywhere in training or evaluation.

Fix:
- filter to `collection_month <= m`
- aggregate to **one row per subject** (mean or last sample up to m)

---

## Dataset Counts (Food Allergy Only)

From `data/processed/16s/dataset_manifest.json`:
- **1,450 labeled 16S samples**
- **203 labeled subjects**
- Sample labels: **491 positive / 959 negative**
- Subject labels: **72 positive / 131 negative**

---

## Cross-Validation

### Primary: StratifiedGroupKFold (subject-level)

```python
from sklearn.model_selection import StratifiedGroupKFold

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
```

Notes:
- Stratification uses `y` (label_food) and grouping uses `subject_id`.
- With small horizons/months, `n_splits=5` may be infeasible. Use dynamic `n_splits`.

### Dynamic n_splits

```python
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

def choose_n_splits(y_subj: np.ndarray, n_splits_max: int = 5) -> int:
    n_pos = int((y_subj == 1).sum())
    n_neg = int((y_subj == 0).sum())
    n_splits = min(n_splits_max, n_pos, n_neg)
    if n_splits < 2:
        raise ValueError(f"Insufficient subjects for CV: pos={n_pos} neg={n_neg}")
    return n_splits
```

---

## Within-Subject Weighting (Samples vs Subjects)

If you train directly on all samples, subjects with many samples dominate.

Preferred (simple + robust):
- Build a **subject table** per horizon: one row per subject.

Alternative:
- Train on samples with `sample_weight = 1 / n_samples_for_subject`
- Report metrics aggregated at the subject level.

---

## Metrics

Primary:
- AUROC (subject-level)

Secondary:
- Precision / Recall / F1 at a fixed threshold (0.5) or a calibrated threshold
- Confusion matrix (normalized)

---

## Country Confounding

Food allergy prevalence differs strongly by country in this cohort.

Required analysis:
- **Leave-one-country-out (LOCO)**: train on 2 countries, test on the held-out country.
- Report AUROC per held-out country (and per horizon if doing time-aware evaluation).

Tip: Use `data/processed/16s/dataset_manifest.json` to inspect subject counts by country and label.

---

## Reporting Template

Minimum output tables:
- `results/metrics_by_horizon.csv`
- `results/metrics_loco.csv`

Minimum plots:
- AUROC vs horizon month (mean ± std across folds)
- LOCO AUROC bars by country
