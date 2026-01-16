# Data Directory

## Quick Overview

```
data/
├── README.md                         # This file
├── raw/
│   ├── DIABIMMUNE_Karelia_metadata.RData  # SOURCE OF TRUTH (687 KB)
│   └── sra_runinfo.csv                    # From NCBI - bridges HF to RData (785 rows)
└── processed/                         # WE CREATED THESE
    ├── unified_samples.csv            # PRIMARY DATASET
    ├── srs_to_subject_mapping.csv     # ID mapping
    ├── microbiome_embeddings_100d.h5  # Canonical 100-d embeddings (785 keys)
    └── dataset_manifest.json          # Provenance + checksums + counts
```

---

## What Each File Is

### Ground Truth: `raw/DIABIMMUNE_Karelia_metadata.RData`
- **Source**: Broad Institute DIABIMMUNE Portal
- **What it has**: Everything - 1,946 samples, 222 subjects, allergies, countries
- **Format**: R Data file (use `pyreadr` to read)
- **Trust level**: ✅ Official source

### Downloaded: `raw/sra_runinfo.csv`
- **Source**: NCBI SRA (BioProject PRJNA290380)
- **Purpose**: Bridge between HuggingFace SRS IDs and RData gid_wgs
- **How we got it**: Queried `https://www.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?term=PRJNA290380`

### Created: `processed/unified_samples.csv`
- **What it is**: The clean, merged dataset ready for ML
- **785 samples**, each with:
  - `srs_id` - HuggingFace sample ID (for loading embeddings)
  - `subject_id` - Infant ID (for GroupKFold - CRITICAL)
  - `collection_month` - TRUE collection month (1-38)
  - `label` - 0=healthy, 1=any allergy OR high total IgE (matches HF label)
  - `country` - FIN/EST/RUS
 - **Important**: This is the *embeddings-backed subset* (intersection of PRJNA290380 runinfo + HuggingFace embeddings with the RData metadata).

### Created: `processed/microbiome_embeddings_100d.h5`
- **What it is**: A single canonical `.h5` file with one 100-d embedding per SRS sample (785 keys)
- **Why it exists**: HuggingFace stores the same samples across many `Month_*` folders, and duplicates are not guaranteed to be numerically identical
- **Canonicalization rule**: For each sample, select the embedding from `Month_{collection_month}` (collection_month comes from the RData ground truth)
- **HuggingFace source**: `hugging-science/AI4FA-Diabimmune` @ `7761eea93dad5712a03452786b43031dc9b04233`

### Created: `processed/dataset_manifest.json`
- **What it is**: Machine-readable provenance (sources, counts, and SHA256 checksums of the key artifacts)
- **How it’s created**: `python scripts/prepare_data.py`

---

## How to Load Data

```python
import pandas as pd
import h5py
import numpy as np

# Load the clean dataset
samples = pd.read_csv("data/processed/unified_samples.csv")

# Load canonical embeddings (one vector per SRS)
with h5py.File("data/processed/microbiome_embeddings_100d.h5", 'r') as f:
    embeddings = {srs: np.array(f[srs]) for srs in f.keys()}

# Build arrays
X = np.array([embeddings[srs] for srs in samples['srs_id']])
y = samples['label'].values
groups = samples['subject_id'].values  # FOR GROUPKFOLD!
```

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Total samples | 785 |
| Unique subjects | 212 |
| Allergic (label=1) | 331 (42%) |
| Healthy (label=0) | 454 (58%) |
| Embedding dimensions | 100 |

---

## Known Issues

1. **Multiple samples per subject** - Use subject-level CV and report subject-level metrics
2. **Time leakage** - If predicting “by month m”, do not use samples collected after month m
3. **Country confounding** - Add leave-one-country-out evaluation
4. **Label = ANY allergy** - Includes food + environmental allergies + high IgE

See `docs/data/DATA_INVESTIGATION_FINDINGS.md` for full investigation details.
