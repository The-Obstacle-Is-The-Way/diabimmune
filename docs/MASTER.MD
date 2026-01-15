# DIABIMMUNE Allergy Classifier

## Executive Summary

A binary classifier that predicts allergy development in infants using gut microbiome embeddings from the DIABIMMUNE three-country cohort.

| Component | Description |
|-----------|-------------|
| **Input** | Pre-computed 100-dimensional microbiome embeddings (one per WGS sample) |
| **Output** | Binary prediction (0 = healthy, 1 = any allergy or high total IgE) |
| **Evaluation** | Per-age-bin AUROC, F1, confusion matrices with proper subject-level splits |

---

## Critical Corrections (2026-01-15)

After first-principles data validation, several critical issues were found:

| Issue | Original Assumption | Reality |
|-------|---------------------|---------|
| **Label definition** | Food allergy only (milk/egg/peanut) | ANY allergy + totalige_high |
| **HuggingFace Month_N** | Collection month | **Duplicated data** - same samples in multiple folders |
| **Sample count** | ~658 (16S-based) | 785 (WGS-based) |
| **Subject count** | ~148 | 212 |
| **Class balance** | 35% allergic / 65% healthy | **42% allergic / 58% healthy** |
| **ID mapping column** | `host_subject_id` | `gid_wgs` → `subjectID` |

**See `data/README.md` for complete provenance and validation.**

---

## Why This Matters

Gut microbiome composition in early life predicts allergy development **months before clinical symptoms**. The signal exists—we just need to detect it properly without methodological leakage.

Key scientific findings:
- Allergic children exhibit **reduced microbial diversity**
- **Depletion of protective taxa**: *Bifidobacterium*, *Faecalibacterium*, butyrate-producing *Clostridia*
- **Enrichment of pro-inflammatory bacteria**: *Escherichia-Shigella*, *Ruminococcus gnavus*
- These shifts are detectable months before clinical manifestation

---

## The DIABIMMUNE Cohort

### Study Design

The **Three-Country Cohort** studies the **hygiene hypothesis**—why autoimmune diseases and allergies are more common in developed countries.

| Country | T1D Incidence | Subjects | Notes |
|---------|---------------|----------|-------|
| **Finland** | Highest globally | 74 infants | Western, developed |
| **Estonia** | Rapidly increasing | 74 infants | Modernizing |
| **Russia (Karelia)** | Relatively rare | 74 infants | Less developed |

### Cohort Details

- **222 total infants** enrolled (74 per country)
- **Selection criteria**: Similar HLA risk (T1D predisposition), matched gender
- **Follow-up**: Birth to age 3 (monthly stool samples)
- **Data collected**: Stool samples, lab assays, questionnaires (breastfeeding, diet, allergies, infections, family history, medications)

### Sequencing Data

| Type | Samples | Description |
|------|---------|-------------|
| **16S rRNA amplicon** | 1,584 | V4 region, paired-end |
| **Whole-genome shotgun (WGS)** | 785 | Full metagenome |

**⚠️ HuggingFace embeddings are from WGS samples (785), not 16S (1,584)!**

### Label Definition (CORRECTED)

HuggingFace `label=1` (allergic) means **ANY of the following is True**:

| Allergy Type | Column in RData |
|--------------|-----------------|
| Food: Milk | `allergy_milk` |
| Food: Egg | `allergy_egg` |
| Food: Peanut | `allergy_peanut` |
| Environmental: Dust mite | `allergy_dustmite` |
| Environmental: Cat | `allergy_cat` |
| Environmental: Dog | `allergy_dog` |
| Environmental: Birch | `allergy_birch` |
| Environmental: Timothy grass | `allergy_timothy` |
| **High total IgE** | `totalige_high` |

**This is broader than "food allergy" - it includes environmental allergies and elevated IgE!**

### Dataset Numbers (CORRECTED)

| Category | Count |
|----------|-------|
| Unique WGS samples in HuggingFace | **785** |
| Unique subjects with WGS | **212** |
| Allergic subjects (label=1) | **90** (42%) |
| Healthy subjects (label=0) | **122** (58%) |

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

### Embedding Model

The embeddings come from [Matteo's Microbiome World Model](https://the-puzzler.github.io/post.html?p=posts/micro-modelling/micro-modelling.html):

1. **ProkBERT**: DNA language model embeds 16S rRNA sequences (384-dim per OTU)
2. **MicrobiomeTransformer**: Aggregates OTU embeddings into sample-level representation (100-dim)

### Pre-computed Embeddings

Available on HuggingFace: **[hugging-science/AI4FA-Diabimmune](https://huggingface.co/datasets/hugging-science/AI4FA-Diabimmune)**

**⚠️ CRITICAL WARNING: HuggingFace Month_N Structure**

```
The Month_N folders do NOT represent collection month!

Evidence:
- Sample SRS1719502 appears in: Month_1, Month_4, Month_7, Month_10, Month_13, Month_16, Month_19, Month_22, Month_28, Month_36
- Embeddings are IDENTICAL across these folders
- Same 785 samples are duplicated across folders

For proper analysis, use:
  data/processed/unified_samples.csv
Which contains the TRUE collection_month from RData.
```

---

## ID Mapping Chain

HuggingFace sample IDs need to be mapped to subject IDs through a chain:

```
SRS1719091 (HuggingFace sid)
    │
    └── via SRA runinfo (Sample → LibraryName)
        │
        └── G78508 (gid_wgs in RData)
            │
            └── T012374 (subjectID in RData)
```

**The mapping file is: `data/processed/srs_to_subject_mapping.csv`**

---

## Proper Evaluation

### Cross-Validation Setup

```python
from sklearn.model_selection import StratifiedGroupKFold

# Groups = subject_id (infant)
# Stratify = label (allergy status)
# Ensures:
#   - All samples from one infant stay together
#   - Class balance maintained in each fold

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=subject_ids)):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    # Train and evaluate
```

### Why Subject-Level Splits Are Critical

```
❌ Random split (WRONG):
   Train: [SRS001_month1, SRS001_month3, SRS002_month1]  ← Same infant in both!
   Test:  [SRS001_month2, SRS002_month2]

✅ Subject-level split (CORRECT):
   Train: [All samples from infant_001, infant_002]
   Test:  [All samples from infant_003, infant_004]
```

### Metrics

| Metric | Purpose |
|--------|---------|
| **AUROC** | Primary metric, handles class imbalance |
| **F1-score** | Balance of precision/recall |
| **Precision** | Of predicted allergic, how many actually are? |
| **Recall** | Of actual allergic, how many did we catch? |
| **Confusion matrix** | Normalized, visual validation |

### Analysis Goal

**At which age does the signal become predictive?**

Collection months are binned for analysis:

| Age Bin | Samples | Notes |
|---------|---------|-------|
| 0-3 months | 45 | Very early (high clinical value) |
| 4-6 months | 65 | Introduction of solids |
| 7-12 months | 197 | First year |
| 13-24 months | 381 | Most samples here |
| 25+ months | 97 | Toddler |

---

## The Classifier

### Start Simple: Logistic Regression

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(
        C=1.0,
        class_weight='balanced',  # Critical for imbalanced data
        max_iter=1000,
        random_state=42
    ))
])
```

### Why LogReg First?

1. **Interpretable**: Coefficients show which embedding dimensions matter
2. **Fast**: Quick iteration
3. **Baseline**: If LogReg fails, more complex models won't help
4. **Embedding quality > classifier complexity**

### Upgrade Path (If Needed)

| Model | When to Try |
|-------|-------------|
| Random Forest | If non-linear relationships suspected |
| SVM (RBF kernel) | If decision boundary is complex |
| MLP | If deep patterns exist in embeddings |

---

## Project Structure

```
diabimmune/
├── docs/
│   ├── MASTER.MD                 # This file
│   └── specs/                    # Detailed implementation specs
├── notebooks/
│   └── 01_baseline_classifier.ipynb
├── data/                         # COMMITTED (small enough)
│   ├── README.md                 # Provenance documentation
│   ├── raw/
│   │   ├── huggingface/          # Downloaded from HF (6.9 MB)
│   │   └── sra_runinfo.csv       # SRA metadata
│   └── processed/
│       ├── srs_to_subject_mapping.csv  # ID mapping
│       └── unified_samples.csv         # PRIMARY DATASET
├── results/                      # Evaluation outputs (gitignored)
│   └── ...
├── DIABIMMUNE_Karelia_metadata.RData  # Subject metadata (687 KB)
├── pyproject.toml                # Dependencies (uv)
├── .gitignore
├── LICENSE
└── README.md
```

---

## Minimum Viable Notebook

| Step | Task | Status |
|------|------|--------|
| 1 | **Download embeddings** | ✅ Done (in `data/raw/huggingface/`) |
| 2 | **Extract subject_ids** | ✅ Done (in `data/processed/`) |
| 3 | **Create proper metadata** | ✅ Done (`unified_samples.csv`) |
| 4 | **Run evaluation** | `StratifiedGroupKFold` + LogReg |
| 5 | **Visualize** | AUROC per age bin, confusion matrices |
| 6 | **Document** | What worked, limitations |

---

## Known Gotchas

| Issue | Impact | Mitigation |
|-------|--------|------------|
| **HuggingFace Month_N ≠ collection month** | Wrong analysis | Use `unified_samples.csv` |
| **Label includes environmental allergies** | Broader than "food allergy" | Document clearly |
| Missing `subject_id` in HF CSVs | Data leakage | Use mapping file |
| Class imbalance (42/58) | Biased predictions | `class_weight='balanced'` |
| Variable samples per subject | Uneven representation | Subject-level CV |
| Month 38 has only 4 samples | Unreliable metrics | Exclude from analysis |
| Some subjects have only 1 sample | Limited data | Still include |

---

## Data Files Summary

| File | Rows | Description |
|------|------|-------------|
| `data/processed/unified_samples.csv` | 785 | Primary dataset |
| `data/processed/srs_to_subject_mapping.csv` | 785 | ID mapping |
| `data/raw/huggingface/metadata/Month_*.csv` | varies | HF labels (ignore Month_N structure) |
| `data/raw/huggingface/processed/microbiome_embeddings/*/microbiome_embeddings.h5` | varies | 100-dim embeddings |
| `data/raw/sra_runinfo.csv` | 785 | SRA metadata for ID mapping |
| `DIABIMMUNE_Karelia_metadata.RData` | 1,946 | Full subject metadata |

---

## Loading Data (Example)

```python
import pandas as pd
import h5py
import numpy as np
from pathlib import Path

# Load unified samples with TRUE collection months
samples = pd.read_csv("data/processed/unified_samples.csv")

# Load embeddings (use any Month_N - they're all the same)
embed_path = Path("data/raw/huggingface/processed/microbiome_embeddings/Month_1/microbiome_embeddings.h5")
with h5py.File(embed_path, 'r') as f:
    embeddings = {srs: np.array(f[srs]) for srs in f.keys()}

# Build feature matrix
X = np.array([embeddings[srs] for srs in samples['srs_id']])
y = samples['label'].values
groups = samples['subject_id'].values
```

---

## References

### Primary Data
- [DIABIMMUNE Project](https://diabimmune.broadinstitute.org/diabimmune/)
- [HuggingFace Dataset](https://huggingface.co/datasets/hugging-science/AI4FA-Diabimmune)
- [BioProject PRJNA290380](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA290380)

### Key Papers
- Vatanen et al. 2016 - [Cell](https://doi.org/10.1016/j.cell.2016.04.007) - Original DIABIMMUNE paper
- Metwally et al. 2019 - [PLOS Comp Bio](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006693) - LSTM for allergy prediction

### Model
- [Microbiome World Model Blog](https://the-puzzler.github.io/post.html?p=posts/micro-modelling/micro-modelling.html)
- [AI-For-Food-Allergies/gut_microbiome_project](https://github.com/AI-For-Food-Allergies/gut_microbiome_project)

### Related Work
- [De Filippis et al. 2021](https://www.nature.com/articles/s41467-021-26266-z) - Gut microbiome signatures in pediatric allergy
- [Zhang et al. 2022](https://pmc.ncbi.nlm.nih.gov/articles/PMC9463091/) - Multi-scale study of oral/gut environments in peanut allergy
