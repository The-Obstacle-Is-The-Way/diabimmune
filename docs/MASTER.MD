# DIABIMMUNE Allergy Classifier

## Executive Summary

A binary classifier that predicts food allergy development in infants using gut microbiome embeddings from the DIABIMMUNE cohort.

| Component | Description |
|-----------|-------------|
| **Input** | Pre-computed 100-dimensional microbiome embeddings (one per stool sample) |
| **Output** | Binary prediction (0 = healthy, 1 = allergic) |
| **Evaluation** | Per-month AUROC, F1, confusion matrices with proper subject-level splits |

---

## Why This Matters

Gut microbiome composition in early life predicts food allergy development months before clinical symptoms. The signal exists—we just need to detect it properly without methodological leakage.

---

## The Data Pipeline (Already Done)

```
Stool Sample (SRS ID)
    ↓
OTU IDs (which bacteria present)
    ↓
16S rRNA DNA sequences
    ↓
ProkBERT embeddings (384-dim per OTU)
    ↓
MicrobiomeTransformer aggregation (100-dim per sample)
    ↓
Ready for classifier
```

The embeddings are pre-computed and available on HuggingFace:
**https://huggingface.co/datasets/hugging-science/AI4FA-Diabimmune**

### Structure

```
Diabimmune/
├── metadata/          # Month_1.csv, Month_2.csv, etc.
├── processed/
│   └── microbiome_embeddings/  # The 100-dim vectors we need
```

---

## The Critical Problem

The existing `Month_*.csv` files only have:

```csv
sid,label
SRS1719091,0
SRS1719092,1
```

**Missing:** `subject_id` (which infant) and `country` (FIN/EST/RUS)

> ⚠️ Without `subject_id`, you can't do `GroupKFold` → samples from the same infant leak into train+test → **inflated metrics**.

---

## What You Need to Build

### Option A: Fix the Data Yourself

1. Download the raw metadata from DIABIMMUNE or use ENA API
2. Map `SRS` → `host_subject_id` (the `preprocess_ai_allergies.py` script shows how)
3. Create proper CSVs with: `sid`, `label`, `subject_id`, `country`

### Option B: Work with What Exists (Limited)

- Train/test on embeddings ignoring subject structure
- Acknowledge in results that this is a methodological limitation
- Still useful as a baseline, just not publishable

---

## Proper Evaluation Requirements

```python
from sklearn.model_selection import StratifiedGroupKFold

# Groups = subject_id (infant)
# Stratify = label (allergy status)
# This ensures:
#   - All samples from one infant stay together
#   - Class balance maintained in each fold

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in cv.split(X, y, groups=subject_ids):
    # Train and evaluate
```

### Metrics per Month

- **AUROC** (primary)
- **F1-score**
- **Precision/Recall**
- **Confusion matrix** (normalized)

**Analysis goal:** At which month does the signal become predictive? Month 1? Month 3? Month 12?

---

## Minimal Notebook Structure

```
diabimmune_baseline/
├── notebooks/
│   └── 01_baseline_classifier.ipynb
├── data/                    # Downloaded from HuggingFace
├── requirements.txt         # or pyproject.toml
└── README.md
```

**You don't need:**
- Tests for a notebook (it's exploratory)
- Complex package structure
- `pyproject.toml` (`requirements.txt` is fine for notebooks)

**The notebook should:**
1. Load embeddings from HuggingFace
2. Load/create proper metadata with `subject_id`s
3. Run `StratifiedGroupKFold` evaluation
4. Output per-month metrics and plots
5. Be reproducible (set random seeds)

---

## The Classifier (Simple)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        C=1.0,
        class_weight='balanced',
        max_iter=1000,
        random_state=42
    ))
])
```

Start with LogReg. If it works, try RF/SVM/MLP. The embedding quality matters more than classifier complexity.

---

## Summary: Minimum Viable Notebook

| Step | Task | Notes |
|------|------|-------|
| 1 | **Get embeddings** | Download from HuggingFace (~5 min) |
| 2 | **Get proper metadata** | Either fix CSVs yourself or ask team (**blocker**) |
| 3 | **Run evaluation** | `StratifiedGroupKFold` + LogReg + metrics |
| 4 | **Visualize** | AUROC per month, confusion matrices |
| 5 | **Document** | What worked, what didn't, limitations |
