# Data Directory

## Overview

This directory contains data for the DIABIMMUNE Allergy Classifier project.

**Last updated**: 2026-01-15

---

## Data Sources

| Source | Location | Description |
|--------|----------|-------------|
| HuggingFace | `raw/huggingface/` | AI4FA-Diabimmune dataset (embeddings + metadata) |
| SRA | `raw/sra_runinfo.csv` | NCBI SRA run info for PRJNA290380 |
| DIABIMMUNE | `../DIABIMMUNE_Karelia_metadata.RData` | Subject metadata from Broad Institute |

---

## Critical Discovery: HuggingFace Month_N Structure

**WARNING**: The HuggingFace `Month_N` folders do NOT represent collection month!

```
Evidence:
- Sample SRS1719502 appears in months: 1, 4, 7, 10, 13, 16, 19, 22, 28, 36
- Embeddings are identical across these folders
- Labels are consistent (same label in all folders)
- Same 785 samples duplicated across folders for unknown purpose
```

**For proper analysis, use `data/processed/unified_samples.csv` which contains the TRUE collection month from RData.**

---

## ID Mapping Chain

HuggingFace uses SRA sample IDs which need to be mapped to subject IDs:

```
SRS* (HuggingFace sid)
    └── maps via SRA runinfo to →
        LibraryName (G* format)
            └── matches RData column →
                gid_wgs
                    └── links to →
                        subjectID (T*/E*/P* format)
```

The mapping file is: `processed/srs_to_subject_mapping.csv`

---

## Directory Structure

```
data/
├── README.md                          # This file
├── raw/
│   ├── huggingface/                   # HuggingFace dataset (6.9 MB)
│   │   ├── metadata/                  # Month_N.csv label files
│   │   │   ├── Month_1.csv           # WARNING: Same samples repeated!
│   │   │   └── ...
│   │   └── processed/
│   │       └── microbiome_embeddings/
│   │           ├── Month_1/
│   │           │   └── microbiome_embeddings.h5  # 100-dim float32
│   │           └── ...
│   └── sra_runinfo.csv                # SRA metadata (785 WGS samples)
│
└── processed/
    ├── srs_to_subject_mapping.csv     # SRS → subject_id mapping
    ├── unified_samples.csv            # Clean dataset with TRUE collection months
    └── Month_N_labels_enriched.csv    # Labels + subject_id (per HF month)
```

---

## Key Files

### `processed/unified_samples.csv`

The **primary dataset** for analysis. Contains:

| Column | Description |
|--------|-------------|
| `srs_id` | HuggingFace sample ID (SRS format) |
| `gid_wgs` | WGS library ID (links to RData) |
| `subject_id` | Infant identifier (for GroupKFold) |
| `collection_month` | TRUE collection month (1-38) |
| `age_days` | Age at collection in days |
| `country` | FIN, EST, or RUS |
| `label` | 0=healthy, 1=allergic |

**Statistics**:
- 785 samples
- 212 subjects
- 331 allergic (42%), 454 healthy (58%)

### `processed/srs_to_subject_mapping.csv`

Maps HuggingFace IDs to subject metadata:

| Column | Description |
|--------|-------------|
| `srs_id` | HuggingFace sample ID |
| `gid_wgs` | WGS library ID |
| `subject_id` | Infant identifier |
| `country` | FIN, EST, or RUS |
| `gender` | Male or Female |

---

## Label Definition

HuggingFace `label=1` (allergic) means:

```
ANY of the following is True:
  - allergy_milk
  - allergy_egg
  - allergy_peanut
  - allergy_dustmite
  - allergy_cat
  - allergy_dog
  - allergy_birch
  - allergy_timothy
  - totalige_high
```

**Note**: This includes environmental allergies, not just food allergies!

---

## Provenance

### HuggingFace Dataset

```
Source: hugging-science/AI4FA-Diabimmune
URL: https://huggingface.co/datasets/hugging-science/AI4FA-Diabimmune
Downloaded: 2026-01-15
Method: huggingface_hub.hf_hub_download()
```

### SRA Run Info

```
Source: NCBI SRA
BioProject: PRJNA290380
URL: https://www.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?term=PRJNA290380
Downloaded: 2026-01-15
Samples: 785 WGS runs
```

### DIABIMMUNE Metadata

```
Source: Broad Institute DIABIMMUNE Portal
URL: https://diabimmune.broadinstitute.org/diabimmune/three-country-cohort/resources/subject-metadata
File: DIABIMMUNE_Karelia_metadata.RData
Size: 687 KB
Format: R Data file (read with pyreadr)
```

---

## Embedding Details

| Property | Value |
|----------|-------|
| Dimensions | 100 |
| Data type | float32 |
| Storage | HDF5 (h5py) |
| Source model | MicrobiomeTransformer |
| Upstream | ProkBERT (384-dim per OTU) |

To load an embedding:

```python
import h5py
import numpy as np

with h5py.File("data/raw/huggingface/processed/microbiome_embeddings/Month_1/microbiome_embeddings.h5", "r") as f:
    embedding = np.array(f["SRS1719091"])  # Shape: (100,)
```

---

## Data Integrity Validation

Run this to validate all data files:

```python
import pandas as pd
import h5py

# Check mapping
mapping = pd.read_csv("data/processed/srs_to_subject_mapping.csv")
assert len(mapping) == 785
assert mapping['subject_id'].nunique() == 212

# Check unified samples
samples = pd.read_csv("data/processed/unified_samples.csv")
assert len(samples) == 785
assert samples['label'].isin([0, 1]).all()

# Check embeddings exist for all samples
with h5py.File("data/raw/huggingface/processed/microbiome_embeddings/Month_1/microbiome_embeddings.h5", "r") as f:
    assert len(f.keys()) > 0

print("All validations passed!")
```
