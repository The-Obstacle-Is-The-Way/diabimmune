# DIABIMMUNE Allergy Classifier

## Executive Summary

A binary classifier that predicts food allergy development in infants using gut microbiome embeddings from the DIABIMMUNE three-country cohort.

| Component | Description |
|-----------|-------------|
| **Input** | Pre-computed 100-dimensional microbiome embeddings (one per stool sample) |
| **Output** | Binary prediction (0 = healthy, 1 = allergic) |
| **Evaluation** | Per-month AUROC, F1, confusion matrices with proper subject-level splits |

---

## Why This Matters

Gut microbiome composition in early life predicts food allergy development **months before clinical symptoms**. The signal exists—we just need to detect it properly without methodological leakage.

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

### Allergy Labels

Food allergy status determined by **IgE measurements** for:
- **Milk**
- **Egg**
- **Peanut**

Subject is labeled **allergic (1)** if positive for any of these three allergens.

### Published Dataset Numbers

From [Metwally et al. 2019](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1006693):

| Category | Count |
|----------|-------|
| Total subjects after filtering | **148** |
| Food-allergic subjects | **52** (35%) |
| Non-allergic subjects | **96** (65%) |
| Total samples | **658** |

> **Note**: Original 195 subjects → 148 after excluding those with <3 timepoints

### Citation

```
Vatanen T, Kostic A, d'Hennezel E, et al.
Variation in Microbiome LPS Immunogenicity Contributes to Autoimmunity in Humans.
Cell. 2016 Apr 28. DOI: 10.1016/j.cell.2016.04.007
```

**BioProject ID**: [PRJNA290380](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA290380)

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

```
AI4FA-Diabimmune/
├── metadata/                      # Month_1.csv, Month_2.csv, etc.
│   └── Month_{1,2,3,...38}.csv   # Contains: sid, label
├── processed/
│   ├── dna_embeddings/           # 384-dim ProkBERT embeddings per OTU
│   │   └── month_{1,2,...}/dna_embeddings.h5
│   ├── dna_sequences/            # Raw DNA sequences per sample
│   │   └── month_{1,2,...}/*.csv
│   └── microbiome_embeddings/    # 100-dim aggregated embeddings ← USE THESE
│       └── month_{1,2,...}/microbiome_embeddings.h5
└── README.md
```

> **Warning**: Dataset is marked experimental, structure frozen for ongoing research.

---

## The Critical Problem

### What's Missing

The existing `Month_*.csv` files only have:

```csv
sid,label
SRS1719091,0
SRS1719092,1
```

**Missing columns**:
- `subject_id` (which infant)
- `country` (FIN/EST/RUS)

### Why This Is Fatal

Without `subject_id`:
1. Cannot do `GroupKFold` → samples from same infant leak into train+test
2. **Inflated metrics** → model memorizes infant-specific patterns
3. **Not publishable** → methodologically unsound

```
❌ Random split (WRONG):
   Train: [SRS001_month1, SRS001_month3, SRS002_month1]  ← Same infant in both!
   Test:  [SRS001_month2, SRS002_month2]

✅ Subject-level split (CORRECT):
   Train: [All samples from infant_001, infant_002]
   Test:  [All samples from infant_003, infant_004]
```

### Class Imbalance

| Class | Count | Percentage |
|-------|-------|------------|
| Healthy (0) | 96 | 65% |
| Allergic (1) | 52 | 35% |

Must use `class_weight='balanced'` or stratified sampling.

---

## Solution: Get Subject IDs

### Option A: Use Local RData File (Recommended)

You have: `DIABIMMUNE_Karelia_metadata.RData` (687KB)

This file contains the **full subject metadata** including:
- `host_subject_id` (e.g., "T012374")
- Country
- Age at collection
- Allergy status
- Diet info
- And more...

**Extraction approach**:
```python
import pyreadr

# Load RData file
result = pyreadr.read_r('DIABIMMUNE_Karelia_metadata.RData')

# Get the dataframe (check keys for exact name)
metadata_df = result[list(result.keys())[0]]

# Map SRS → subject_id
# You'll need to join on sample accession
```

### Option B: Query ENA/NCBI API

```python
# Using ffq tool
# pip install ffq
import subprocess
import json

srs_id = "SRS1719091"
result = subprocess.run(["ffq", srs_id], capture_output=True, text=True)
metadata = json.loads(result.stdout)
subject_id = metadata.get("host_subject_id")
```

Or use ENA Portal API:
```
https://www.ebi.ac.uk/ena/portal/api/search?query=sample_accession="SRS1719091"&result=sample&fields=host_subject_id
```

### Option C: Download from DIABIMMUNE Website

Subject metadata available at: [diabimmune.broadinstitute.org](https://diabimmune.broadinstitute.org/diabimmune/three-country-cohort/resources/subject-metadata)

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

### Metrics

| Metric | Purpose |
|--------|---------|
| **AUROC** | Primary metric, handles class imbalance |
| **F1-score** | Balance of precision/recall |
| **Precision** | Of predicted allergic, how many actually are? |
| **Recall** | Of actual allergic, how many did we catch? |
| **Confusion matrix** | Normalized, visual validation |

### Analysis Goal

**At which month does the signal become predictive?**

- Month 1? (Very early, high clinical value)
- Month 3?
- Month 12?
- Month 36? (Too late for intervention)

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
├── src/                          # Optional: reusable code
│   └── ...
├── data/                         # Downloaded from HuggingFace (gitignored)
│   └── ...
├── results/                      # Evaluation outputs (gitignored)
│   └── ...
├── DIABIMMUNE_Karelia_metadata.RData  # Subject metadata
├── pyproject.toml                # Dependencies (uv)
├── .gitignore
├── LICENSE
└── README.md
```

---

## Minimum Viable Notebook

| Step | Task | Status |
|------|------|--------|
| 1 | **Download embeddings** | HuggingFace (~5 min) |
| 2 | **Extract subject_ids** | From RData file (**blocker solved**) |
| 3 | **Create proper metadata** | `sid`, `label`, `subject_id`, `country` |
| 4 | **Run evaluation** | `StratifiedGroupKFold` + LogReg |
| 5 | **Visualize** | AUROC per month, confusion matrices |
| 6 | **Document** | What worked, limitations |

---

## Known Gotchas

| Issue | Impact | Mitigation |
|-------|--------|------------|
| Missing `subject_id` in HF CSVs | Data leakage | Extract from RData |
| Class imbalance (35/65) | Biased predictions | `class_weight='balanced'` |
| Variable samples per subject | Uneven representation | Subject-level CV |
| Inconsistent CSV schemas in HF | Loader errors | Handle separately |
| Month 38 has very few samples | Unreliable metrics | May need to exclude |
| Some subjects have <3 timepoints | Noise | Consider filtering |

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
