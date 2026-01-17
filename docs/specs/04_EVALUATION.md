# 04: Evaluation Strategy

## Goal

Scientifically valid evaluation for a longitudinal cohort:
- no **subject leakage**,
- no **time leakage** (when claiming early prediction),
- robust to **class imbalance**,
- explicit about **country confounding**,
- **reproducible** (consistent seeds across all folds).

Applies to both datasets:
- **Track A**: `data/processed/longitudinal_wgs_subset/` + `data/processed/hf_legacy/` (785 samples / 212 patients)
- **Track B**: `data/processed/16s/*` (1,450 samples / 203 subjects)

---

## Leakage Risks

### Subject leakage (must prevent)

Multiple samples per infant mean random sample-splitting is invalid.

Fix:
- `groups = subject_id`
- split with `StratifiedGroupKFold`

### Time leakage (must prevent for “predict by month m”)

If the claim is “predict by month m”, you must not use samples with `collection_month > m` anywhere in training or evaluation.

Notes on column names:
- Track A notebook uses `month` derived from the `Month_N.csv` filename.
- Track B uses `collection_month` from `samples_food_allergy.csv`.

Fix:
- filter to `month <= m` (Track A) or `collection_month <= m` (Track B)
- aggregate to **one row per subject** (mean or last sample up to m)

---

## Horizon Analysis (Prediction vs Association)

This project does **cumulative horizons**, not disjoint “month bins”:
- ✅ Use `month <= m` (e.g., `m ∈ {3, 6, 12}`) plus an “all samples” baseline
- ❌ Do not run “months 7–12 only”, “13–24 only”, etc. (answers a different question and is typically underpowered)

Why horizons matter:
- The Track A label is an **endpoint outcome** repeated across all samples for an infant.
- Later samples are more likely to be post-onset / post-management → higher risk of reverse causation.
- Without onset timing, horizons represent **gradations of claim strength**, not a perfect prediction/association boundary.

### Track A Horizon Set (Baseline Notebook)

Use these four horizons:
- `month <= 3` (exploratory “earliest window”)
- `month <= 6` (early-life)
- `month <= 12` (first year; mixed pre/post onset)
- `all samples` (association baseline)

Counts and caveats live in `docs/specs/00_HYPOTHESIS.md`.

### Required Aggregation (One Row Per Subject Per Horizon)

For each horizon, build a **subject table** with one embedding per infant:

```python
# df columns (Track A notebook): sid, patient_id, country, label, month, embedding
df_h = df[df["month"] <= m]  # for m in {3, 6, 12}

def mean_embedding(vectors: pd.Series) -> np.ndarray:
    return np.mean(np.stack(vectors.to_list(), axis=0), axis=0)

X_subj = np.stack(df_h.groupby("patient_id")["embedding"].apply(mean_embedding).to_list())
y_subj = df_h.groupby("patient_id")["label"].first().to_numpy()
country_subj = df_h.groupby("patient_id")["country"].first().to_numpy()
patient_ids = df_h.groupby("patient_id").size().index.to_numpy()
```

Then evaluate:
- CV: `StratifiedGroupKFold(...).split(X_subj, y_subj, groups=patient_ids)`
- LOCO: train on 2 countries, test on held-out country **within the same horizon**

### LOCO Viability Note

At `month <= 3`, Russia has too few samples/patients for meaningful LOCO testing. Skip (or label as “not meaningful”) rather than over-interpreting.

---

## Dataset Counts

### Track A: HF Embeddings (Baseline)

From `data/processed/longitudinal_wgs_subset/`:
- **785 samples / 212 patients**
- Sample labels: ~258 positive (~33%) / ~527 negative (~67%)
- Country distribution: FIN 281 (49% allergic), EST 199 (38%), RUS 305 (15%)

### Track B: 16S OTU (Future)

From `data/processed/16s/dataset_manifest.json`:
- **1,450 samples / 203 subjects**
- Sample labels: 491 positive / 959 negative
- Subject labels: 72 positive / 131 negative

---

## Reproducibility: Seed Policy

**Global seed**: `RANDOM_SEED = 42`

All random operations must use this seed:
- `StratifiedGroupKFold(shuffle=True, random_state=RANDOM_SEED)`
- `train_test_split(..., random_state=RANDOM_SEED)`
- Any model with randomness: `LogisticRegression(..., random_state=RANDOM_SEED)`
- NumPy: `np.random.seed(RANDOM_SEED)` at notebook start

This ensures:
- Same fold assignments across runs
- Same model initialization
- Reproducible results

---

## Cross-Validation Structure

### Baseline: Outer CV Only

For the baseline notebook with fixed hyperparameters, **only outer CV is needed**:

```
┌─────────────────────────────────────────────────────────────┐
│ OUTER LOOP: 5-fold StratifiedGroupKFold                     │
│   Purpose: Estimate generalization performance              │
│   Groups: patient_id                                        │
│                                                             │
│   For each fold:                                            │
│     1. Train LogReg with fixed hyperparams on train set     │
│     2. Evaluate on test set (truly unseen)                  │
│     3. Record metrics                                       │
│                                                             │
│   Result: Unbiased performance estimate (mean ± std)        │
└─────────────────────────────────────────────────────────────┘
```

**Why this is sufficient:**
- No hyperparameter tuning = no selection bias = no inner loop needed
- Fixed hyperparameters (`C=1.0, class_weight='balanced'`) are reasonable defaults for LogReg
- Outer CV alone gives unbiased generalization estimates

### Baseline Implementation

```python
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

RANDOM_SEED = 42
outer_cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

cv_results = []
for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=patient_ids)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Fixed hyperparameters - no tuning, no inner loop
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

    # Evaluate on truly unseen test fold
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    cv_results.append({'fold': fold_idx, 'auroc': roc_auc_score(y_test, y_pred_proba)})

# Report: mean ± std across folds
```

---

### Optional: Nested CV (Only If Hyperparameter Tuning Is Needed)

**This section is OPTIONAL. The baseline does NOT need this.**

If you later want to tune hyperparameters AND get unbiased performance estimates, you must use nested CV. This is the only correct way to both tune and evaluate.

**Why nested CV?**
- GridSearchCV alone on full data → hyperparams selected by seeing ALL data → optimistically biased estimates
- Nested CV → outer test folds are genuinely unseen during hyperparam selection → unbiased

```python
# ONLY if hyperparameter tuning is needed
from sklearn.model_selection import GridSearchCV

for fold_idx, (train_idx, test_idx) in enumerate(outer_cv.split(X, y, groups=patient_ids)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = patient_ids[train_idx]

    # Inner CV for hyperparam selection (only on training data)
    inner_cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED)
    param_grid = {'clf__C': [0.01, 0.1, 1.0, 10.0]}

    grid_search = GridSearchCV(model, param_grid, cv=inner_cv, scoring='roc_auc')
    grid_search.fit(X_train, y_train, groups=groups_train)  # groups= is CRITICAL

    # Evaluate on outer test fold (truly unseen)
    y_pred_proba = grid_search.predict_proba(X_test)[:, 1]
```

**Critical**: Inner CV must use `groups=groups_train` to maintain subject-level splits.

### Dynamic n_splits

For small subsets (e.g., earliest horizons):

```python
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

Food allergy prevalence differs strongly by country:

| Country | Track A Samples | Allergy Rate |
|---------|-----------------|--------------|
| FIN | 281 | 49% |
| EST | 199 | 38% |
| RUS | 305 | 15% |

A model could learn "is this sample from Russia?" instead of "does this microbiome pattern predict allergy?".

### Leave-One-Country-Out (LOCO)

Required secondary analysis:

```python
countries = ['FIN', 'EST', 'RUS']
loco_results = []

for held_out in countries:
    train_mask = df['country'] != held_out
    test_mask = df['country'] == held_out

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    model.fit(X_train, y_train)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    loco_results.append({
        'held_out': held_out,
        'n_test': len(y_test),
        'auroc': roc_auc_score(y_test, y_pred_proba)
    })
```

**Interpretation:**
- If LOCO AUROC >> 0.5: Model learns transferable microbiome signal
- If LOCO AUROC ≈ 0.5: Model may be learning country-specific batch effects

---

## Class Imbalance Handling

With ~33% positive class, use:

1. **`class_weight='balanced'`** in classifiers
2. **AUPRC** (Area Under Precision-Recall Curve) as secondary metric alongside AUROC
3. **Stratified folds** to maintain class ratio in each fold

Do NOT use:
- Accuracy (misleading with imbalance)
- Oversampling/SMOTE (adds complexity without clear benefit for this ratio)

---

## Reporting Template

### Output Tables

- `results/cv_metrics.csv` — per-fold AUROC, AUPRC, F1 (with `horizon` column)
- `results/cv_summary.csv` — mean ± std across folds (grouped by horizon)
- `results/loco_metrics.csv` — AUROC per held-out country (with horizon column)

### Output Plots

- ROC curves (per-fold + mean)
- Precision-Recall curves (per-fold + mean)
- LOCO AUROC bar chart by country
- (Optional) AUROC vs horizon cutoff `m` (cumulative horizons)

### Reproducibility Footer

Every notebook must end with:

```python
print(f"Random seed: {RANDOM_SEED}")
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"pandas: {pd.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"Run completed: {datetime.now().isoformat()}")
```
