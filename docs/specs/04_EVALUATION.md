# 04: Evaluation Strategy

## Overview

Detailed specification of the evaluation methodology to ensure scientifically valid, reproducible results without data leakage.

---

## The Data Leakage Problem

### Why Random Splits Fail

In longitudinal studies like DIABIMMUNE, each infant has multiple stool samples across different months. If we randomly split samples:

```
❌ Random Split (WRONG):
   Subject_001: Month 1 → Train, Month 3 → Test, Month 6 → Train
   Subject_002: Month 2 → Train, Month 6 → Test

   Problem: Model sees Subject_001's Month 1 & 6 during training,
            then "predicts" Month 3 — but it's memorizing the infant,
            not learning the microbiome signal.
```

### Why Subject-Level Splits Work

```
✅ Subject-Level Split (CORRECT):
   Train: All samples from Subject_001, Subject_002, Subject_003
   Test:  All samples from Subject_004, Subject_005

   Result: Model must generalize to unseen infants,
           which is what we want in clinical deployment.
```

---

## Cross-Validation Strategy

### StratifiedGroupKFold

Combines two critical requirements:

| Requirement | How It's Met |
|-------------|--------------|
| **No leakage** | `groups=subject_ids` keeps all samples from one infant together |
| **Class balance** | Stratification maintains ~42/58 allergic/healthy ratio (sample-level; subject-level is 90/122) |

### Implementation

```python
from sklearn.model_selection import StratifiedGroupKFold
import numpy as np

def create_cv_splitter(
    n_splits: int = 5,
    random_state: int = 42,
) -> StratifiedGroupKFold:
    """Create cross-validator with subject-level grouping."""
    return StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )


def validate_cv_splits(
    cv: StratifiedGroupKFold,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> None:
    """Verify no subject appears in both train and test."""
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups)):
        train_subjects = set(groups[train_idx])
        test_subjects = set(groups[test_idx])

        overlap = train_subjects & test_subjects
        if overlap:
            raise ValueError(f"Fold {fold}: Subject overlap detected: {overlap}")

        # Check class balance
        train_ratio = y[train_idx].mean()
        test_ratio = y[test_idx].mean()

        print(f"Fold {fold}:")
        print(f"  Train: {len(train_idx)} samples, {len(train_subjects)} subjects, "
              f"{train_ratio:.1%} allergic")
        print(f"  Test:  {len(test_idx)} samples, {len(test_subjects)} subjects, "
              f"{test_ratio:.1%} allergic")
```

### Small-Month Handling (Dynamic `n_splits`)

Some timepoints/months have too few subjects per class to support 5-fold CV. Use a dynamic splitter:

```python
def choose_n_splits(
    y: np.ndarray,
    groups: np.ndarray,
    n_splits_max: int = 5,
) -> int:
    """Choose the largest feasible n_splits given subject counts per class."""
    unique_groups, first_idx = np.unique(groups, return_index=True)
    y_group = y[first_idx]  # safe because labels are constant within subject

    n_pos = int((y_group == 1).sum())
    n_neg = int((y_group == 0).sum())
    n_splits = min(n_splits_max, n_pos, n_neg)
    if n_splits < 2:
        raise ValueError(f"Insufficient subjects for CV: pos={n_pos} neg={n_neg}")
    return n_splits
```

### Within-Subject Weighting (Subjects vs Samples)

If you train on **samples** (multiple rows per subject), subjects with many samples dominate both training and metrics.

Preferred (simplest + robust): build a **subject-level** feature table (one row per subject) for each prediction horizon `m` by aggregating all samples with `collection_month <= m` (mean or last embedding). Then evaluate with stratified CV on subjects.

Secondary option (sample-level): train with per-subject `sample_weight` so each subject contributes total weight 1, and report metrics at the subject level by aggregating test predictions per subject (e.g., mean predicted probability).

---

## Metrics Specification

### Primary Metric: AUROC

**Why AUROC?**
- Threshold-independent (evaluates ranking ability)
- Handles class imbalance well
- Standard in clinical prediction papers
- Interpretable: probability that a random allergic sample scores higher than a random healthy sample

```python
from sklearn.metrics import roc_auc_score

def compute_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Area Under ROC Curve."""
    return roc_auc_score(y_true, y_prob)
```

### Secondary Metrics

| Metric | Formula | Purpose |
|--------|---------|---------|
| **F1-Score** | 2 × (P × R) / (P + R) | Balance precision/recall |
| **Precision** | TP / (TP + FP) | Of predicted allergic, how many are? |
| **Recall** | TP / (TP + FN) | Of actual allergic, how many caught? |
| **Specificity** | TN / (TN + FP) | Of actual healthy, how many correct? |

```python
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from dataclasses import dataclass


@dataclass
class Metrics:
    """Collection of evaluation metrics."""
    auroc: float
    f1: float
    precision: float
    recall: float
    specificity: float
    confusion_matrix: np.ndarray


def compute_all_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Metrics:
    """Compute all evaluation metrics."""
    y_pred = (y_prob >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    return Metrics(
        auroc=roc_auc_score(y_true, y_prob),
        f1=f1_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        specificity=tn / (tn + fp) if (tn + fp) > 0 else 0,
        confusion_matrix=cm,
    )
```

---

## Handling Class Imbalance

### The Problem (CORRECTED)

| Class | Count | Percentage |
|-------|-------|------------|
| Healthy (0) | 454 | 58% |
| Allergic (1) | 331 | 42% |

**Note**: These are sample counts (785 total). At subject level: 122 healthy, 90 allergic.

A naive model predicting all "healthy" achieves 58% accuracy but is useless.

### Label Definition (IMPORTANT)

HuggingFace `label=1` (allergic) includes **ANY** of:
- Food allergies: milk, egg, peanut
- Environmental allergies: dustmite, cat, dog, birch, timothy
- High total IgE (`totalige_high`)

**This is broader than just "food allergy"!**

### Solution 1: Balanced Class Weights

```python
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_model(random_state: int = 42) -> Pipeline:
    """Baseline model: StandardScaler + LogisticRegression."""
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="liblinear",
                    penalty="l2",
                    C=1.0,
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=random_state,
                ),
            ),
        ]
    )
```

### Solution 2: Stratified Sampling

Already handled by `StratifiedGroupKFold` — each fold maintains the original class distribution.

### Solution 3: Threshold Tuning (Optional)

```python
def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> float:
    """Find threshold that maximizes chosen metric."""
    thresholds = np.linspace(0.1, 0.9, 81)
    best_threshold = 0.5
    best_score = 0.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred)
        elif metric == "youden":
            # Youden's J = Sensitivity + Specificity - 1
            cm = confusion_matrix(y_true, y_pred)
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn) if (tp + fn) > 0 else 0
            spec = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sens + spec - 1

        if score > best_score:
            best_score = score
            best_threshold = t

    return best_threshold
```

---

## Per-Age-Bin Analysis

**⚠️ IMPORTANT**: HuggingFace `Month_N` folders do NOT represent collection month!
Use `data/processed/unified_samples.csv` with TRUE `collection_month` from RData.

### Goal

Determine at which developmental stage the microbiome signal becomes predictive.

**Important**: avoid *time leakage*. If you claim “predict by month m”, inputs must not include samples collected after month m.

### Age Bins

| Age Bin | Collection Months | Samples | Description |
|---------|-------------------|---------|-------------|
| 0-3 | 1-3 | 45 | Very early (high clinical value) |
| 4-6 | 4-6 | 65 | Introduction of solids |
| 7-12 | 7-12 | 197 | First year |
| 13-24 | 13-24 | 381 | Most samples |
| 25+ | 25-38 | 97 | Toddler |

### Approach

Recommended (leakage-resistant): build **one row per subject** at horizon `m` by aggregating samples with `collection_month <= m` (e.g., mean embedding or last observed embedding), then run subject-level CV.

```python
def evaluate_by_age_bin(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    samples_df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate model on each age bin independently."""

    def get_age_bin(month):
        if month <= 3: return "0-3"
        elif month <= 6: return "4-6"
        elif month <= 12: return "7-12"
        elif month <= 24: return "13-24"
        else: return "25+"

    samples_df = samples_df.copy()
    samples_df['age_bin'] = samples_df['collection_month'].apply(get_age_bin)

    results = []

    for age_bin in ["0-3", "4-6", "7-12", "13-24", "25+"]:
        mask = samples_df['age_bin'] == age_bin
        if mask.sum() < 10:  # Skip if too few samples
            continue

        X_bin = X[mask]
        y_bin = y[mask]
        groups_bin = groups[mask]

        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        fold_metrics = []

        for train_idx, test_idx in cv.split(X_bin, y_bin, groups=groups_bin):
            X_train, X_test = X_bin[train_idx], X_bin[test_idx]
            y_train, y_test = y_bin[train_idx], y_bin[test_idx]

            model = create_model()
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]

            fold_metrics.append(compute_all_metrics(y_test, y_prob))

        results.append({
            "age_bin": age_bin,
            "n_samples": mask.sum(),
            "n_subjects": len(set(groups_bin)),
            "n_allergic": int(y_bin.sum()),
            "auroc_mean": np.mean([m.auroc for m in fold_metrics]),
            "auroc_std": np.std([m.auroc for m in fold_metrics]),
            "f1_mean": np.mean([m.f1 for m in fold_metrics]),
            "f1_std": np.std([m.f1 for m in fold_metrics]),
        })

    return pd.DataFrame(results)
```

### Expected Output

```
Age Bin | Samples | Subjects | Allergic | AUROC (mean±std) | F1 (mean±std)
--------|---------|----------|----------|------------------|---------------
0-3     | 45      | ~40      | ~19      | 0.52 ± 0.08      | 0.35 ± 0.10
4-6     | 65      | ~55      | ~27      | 0.58 ± 0.07      | 0.42 ± 0.09
7-12    | 197     | ~100     | ~80      | 0.63 ± 0.06      | 0.48 ± 0.08
13-24   | 381     | ~180     | ~160     | 0.65 ± 0.05      | 0.50 ± 0.07
25+     | 97      | ~80      | ~40      | 0.60 ± 0.07      | 0.45 ± 0.09
```

---

## Statistical Significance

### Bootstrap Confidence Intervals

```python
from scipy import stats


def bootstrap_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for a metric.

    Returns:
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        score = metric_fn(y_true[idx], y_prob[idx])
        scores.append(score)

    point = metric_fn(y_true, y_prob)
    lower = np.percentile(scores, (1 - ci) / 2 * 100)
    upper = np.percentile(scores, (1 + ci) / 2 * 100)

    return point, lower, upper
```

### Comparing Months

```python
def is_significantly_better(
    auroc_1: float,
    auroc_2: float,
    n_1: int,
    n_2: int,
    alpha: float = 0.05,
) -> bool:
    """Test if AUROC difference is significant using DeLong test approximation."""
    # Simplified: use standard error approximation
    se_1 = np.sqrt(auroc_1 * (1 - auroc_1) / n_1)
    se_2 = np.sqrt(auroc_2 * (1 - auroc_2) / n_2)

    z = (auroc_1 - auroc_2) / np.sqrt(se_1**2 + se_2**2)
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return p_value < alpha
```

---

## Country Confounding Check (Leave-One-Country-Out)

Country is a major potential confounder (FIN/EST/RUS differ in environment, diet, and baseline allergy prevalence). Add a strict evaluation where you train on two countries and test on the third.

```python
def leave_one_country_out(
    X_subj: np.ndarray,
    y_subj: np.ndarray,
    subj_df: pd.DataFrame,  # must include columns: subject_id, country
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for held_out in sorted(subj_df["country"].unique()):
        train_mask = subj_df["country"] != held_out
        test_mask = subj_df["country"] == held_out

        if len(set(y_subj[train_mask.to_numpy()])) < 2:
            continue
        if len(set(y_subj[test_mask.to_numpy()])) < 2:
            continue

        model = create_model()
        model.fit(X_subj[train_mask.to_numpy()], y_subj[train_mask.to_numpy()])
        prob = model.predict_proba(X_subj[test_mask.to_numpy()])[:, 1]

        rows.append(
            {
                "held_out_country": held_out,
                "n_test_subjects": int(test_mask.sum()),
                "auroc": float(roc_auc_score(y_subj[test_mask.to_numpy()], prob)),
            }
        )

    return pd.DataFrame(rows)
```

---

## Visualization Functions

### ROC Curve

```python
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    month: int,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot ROC curve for a month."""
    if ax is None:
        _, ax = plt.subplots()

    RocCurveDisplay.from_predictions(
        y_true,
        y_prob,
        ax=ax,
        name=f"Month {month}",
    )

    auroc = roc_auc_score(y_true, y_prob)
    ax.set_title(f"Month {month} (AUROC={auroc:.3f})")
    ax.grid(True, alpha=0.3)

    return ax
```

### Confusion Matrix

```python
from sklearn.metrics import ConfusionMatrixDisplay


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    month: int,
    ax: plt.Axes | None = None,
    normalize: str = "true",
) -> plt.Axes:
    """Plot confusion matrix."""
    if ax is None:
        _, ax = plt.subplots()

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        ax=ax,
        cmap="Blues",
        normalize=normalize,
        display_labels=["Healthy", "Allergic"],
    )
    ax.set_title(f"Month {month}")

    return ax
```

### AUROC Timeline

```python
def plot_auroc_timeline(
    results_df: pd.DataFrame,
    output_path: Path | None = None,
) -> plt.Figure:
    """Plot AUROC across months with error bars."""
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.errorbar(
        results_df["month"],
        results_df["auroc_mean"],
        yerr=results_df["auroc_std"],
        marker="o",
        capsize=5,
        linewidth=2,
        markersize=8,
    )

    ax.axhline(0.5, color="red", linestyle="--", alpha=0.7, label="Random (0.5)")
    ax.axhline(0.7, color="green", linestyle=":", alpha=0.7, label="Good (0.7)")

    ax.set_xlabel("Month", fontsize=12)
    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_title("DIABIMMUNE Prediction: Performance Over Time", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.4, 1.0)

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig
```

---

## Verification Checklist

- [ ] No subject overlap between train/test (verified with `validate_cv_splits`)
- [ ] Class balance maintained in each fold (~42/58 at sample level; 90/122 at subject level)
- [ ] AUROC computed correctly (sanity check: random should be ~0.5)
- [ ] Class weights balanced in model
- [ ] Metrics aggregated with mean ± std across folds
- [ ] Confidence intervals computed where needed
- [ ] All visualizations use consistent styling
