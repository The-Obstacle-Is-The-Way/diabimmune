# 02: Data Pipeline

## Overview

Load pre-downloaded embeddings and use the pre-computed mapping files to create a unified dataset ready for training.

**Note**: Data has already been downloaded and processed. This spec documents what was done and how to use the data.

---

## Critical Discovery: HuggingFace Month_N Structure

**⚠️ WARNING: The HuggingFace `Month_N` folders do NOT represent “samples collected at month N”.**

```
Evidence found during data validation:
- Sample SRS1719502 appears in: Month_1, Month_4, Month_7, Month_10, Month_13, Month_16, Month_19, Month_22, Month_28, Month_36
- Month_* folders are overlapping subject subsets (the same sample can appear in many folders)
- Duplicate embeddings for the same SRS are NOT guaranteed identical (small numeric differences exist)

Conclusion:
- Use `unified_samples.csv` (from the RData) for TRUE collection month.
- Export a single canonical embedding file so each SRS has exactly one 100-d vector.
```

---

## Data Sources

| Source | Location | Contents | Status |
|--------|----------|----------|--------|
| **Embeddings (canonical)** | `data/processed/microbiome_embeddings_100d.h5` | 100-dim vectors per sample (785 keys) | ✅ Created |
| **SRA Mapping** | `data/raw/sra_runinfo.csv` | SRS → gid_wgs mapping | ✅ Downloaded |
| **Subject IDs** | `data/raw/DIABIMMUNE_Karelia_metadata.RData` | gid_wgs → subjectID | ✅ Present |
| **Unified Dataset** | `data/processed/unified_samples.csv` | Complete mapping | ✅ Created |

Pinned HuggingFace dataset revision (for reproducibility): `7761eea93dad5712a03452786b43031dc9b04233`

---

## ID Mapping Chain

The HuggingFace SRS IDs need to be mapped to subject IDs through a multi-step chain:

```
SRS1719091 (HuggingFace sid)
    │
    └── SRA runinfo: Sample → LibraryName
        │
        └── G78508 (matches RData gid_wgs)
            │
            └── T012374 (RData subjectID)
```

**This mapping is already computed in `data/processed/srs_to_subject_mapping.csv`**

---

## RData Column Names (CORRECTED)

Based on actual exploration of `DIABIMMUNE_Karelia_metadata.RData`:

| Expected (in old spec) | Actual Column | Description |
|------------------------|---------------|-------------|
| `host_subject_id` | **`subjectID`** | Infant identifier (E*, T*, P*) |
| *(SRS sample ID)* | **(not present)** | SRS IDs are not stored in RData |
| `age_months` | **`collection_month`** | Collection month (1-38) |
| *(not in old spec)* | **`gid_wgs`** | WGS library ID (G* format) - KEY FOR MAPPING |
| `country` | **`country`** | FIN, EST, RUS |

**Critical**: The RData `SampleID` column is NOT the SRS format used by HuggingFace. The mapping goes through `gid_wgs` via NCBI SRA (or ENA).

---

## Pre-computed Data Files

All data has been downloaded and processed:

### `data/processed/unified_samples.csv`

The **primary dataset** with TRUE collection months:

```csv
srs_id,gid_wgs,subject_id,collection_month,age_days,country,label
SRS1719087,G69146,E016030,28.0,848.0,EST,0
SRS1719088,G69147,E028155,4.0,122.0,EST,1
...
```

| Column | Description |
|--------|-------------|
| `srs_id` | HuggingFace sample ID |
| `gid_wgs` | WGS library ID (links to RData) |
| `subject_id` | Infant identifier (for GroupKFold) |
| `collection_month` | TRUE collection month (1-38) |
| `age_days` | Age at collection in days |
| `country` | FIN, EST, or RUS |
| `label` | 0=healthy, 1=allergic |

### `data/processed/srs_to_subject_mapping.csv`

ID mapping for reference:

```csv
srs_id,gid_wgs,subject_id,country,gender
SRS1719087,G69146,E016030,EST,Female
...
```

---

## Loading Data (Recommended Approach)

Since data is pre-processed, loading is simple:

```python
import pandas as pd
import h5py
import numpy as np
from pathlib import Path

def load_unified_dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load the complete unified dataset.

    Returns:
        X: (785, 100) embeddings
        y: (785,) binary labels
        groups: (785,) subject IDs for GroupKFold
        metadata: Full DataFrame with all columns
    """
    # Load unified samples
    samples = pd.read_csv("data/processed/unified_samples.csv")

    # Load canonical embeddings (one vector per SRS)
    embed_path = Path("data/processed/microbiome_embeddings_100d.h5")

    embeddings = {}
    with h5py.File(embed_path, 'r') as f:
        for srs in f.keys():
            embeddings[srs] = np.array(f[srs])

    # Build arrays in order of unified_samples
    X = np.array([embeddings[srs] for srs in samples['srs_id']])
    y = samples['label'].values
    groups = samples['subject_id'].values

    return X, y, groups, samples


def load_by_age_bin() -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load data grouped by age bin.

    Returns:
        Dict mapping age_bin to (X, y, groups)
    """
    X, y, groups, samples = load_unified_dataset()

    def get_age_bin(month):
        if month <= 3: return "0-3"
        elif month <= 6: return "4-6"
        elif month <= 12: return "7-12"
        elif month <= 24: return "13-24"
        else: return "25+"

    samples['age_bin'] = samples['collection_month'].apply(get_age_bin)

    result = {}
    for age_bin in samples['age_bin'].unique():
        mask = samples['age_bin'] == age_bin
        result[age_bin] = (X[mask], y[mask], groups[mask])

    return result
```

---

## Dataset Statistics

Note: The RData contains more WGS library IDs (`gid_wgs`) than we have embeddings for. `unified_samples.csv` is the 785-sample intersection defined by PRJNA290380 runinfo + HuggingFace embeddings.

| Metric | Value |
|--------|-------|
| Total samples | 785 |
| Unique subjects | 212 |
| Allergic (label=1) | 331 (42%) |
| Healthy (label=0) | 454 (58%) |
| Embedding dimensions | 100 |
| Embedding dtype | float32 |

### Age Bin Distribution

| Age Bin | Samples | Subjects |
|---------|---------|----------|
| 0-3 months | 45 | ~40 |
| 4-6 months | 65 | ~55 |
| 7-12 months | 197 | ~100 |
| 13-24 months | 381 | ~180 |
| 25+ months | 97 | ~80 |

---

## Validation Functions

```python
def validate_data_integrity() -> bool:
    """Validate all data files exist and are consistent."""

    # Check files exist
    required_files = [
        "data/processed/unified_samples.csv",
        "data/processed/srs_to_subject_mapping.csv",
        "data/processed/microbiome_embeddings_100d.h5",
        "data/raw/DIABIMMUNE_Karelia_metadata.RData",
        "data/raw/sra_runinfo.csv",
    ]

    for f in required_files:
        if not Path(f).exists():
            print(f"❌ Missing: {f}")
            return False

    # Load and validate
    samples = pd.read_csv("data/processed/unified_samples.csv")

    assert len(samples) == 785, f"Expected 785 samples, got {len(samples)}"
    assert samples['subject_id'].nunique() == 212, "Expected 212 subjects"
    assert samples['label'].isin([0, 1]).all(), "Labels must be 0 or 1"

    # Check embeddings exist
    with h5py.File("data/processed/microbiome_embeddings_100d.h5", 'r') as f:
        embed_srs = set(f.keys())

    sample_srs = set(samples['srs_id'])
    missing = sample_srs - embed_srs
    assert len(missing) == 0, f"Missing embeddings: {missing}"

    print("✅ All validations passed!")
    return True
```

---

## How Data Was Originally Processed

For reference, here's how the mapping was created:

### Step 1: Download SRA Run Info

```python
import urllib.request
import ssl

ssl._create_default_https_context = ssl._create_unverified_context
url = "https://www.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?term=PRJNA290380"

with urllib.request.urlopen(url) as response:
    content = response.read().decode('utf-8')
    # Saved to data/raw/sra_runinfo.csv
```

### Step 2: Create ID Mapping

```python
import pandas as pd
import pyreadr

# Load SRA data
sra_df = pd.read_csv("data/raw/sra_runinfo.csv")

# Load RData
rdata = pyreadr.read_r("data/raw/DIABIMMUNE_Karelia_metadata.RData")['metadata']

# Create mapping chain
# SRS (Sample column) → LibraryName → gid_wgs → subjectID
srs_to_gid = dict(zip(sra_df['Sample'], sra_df['LibraryName']))
gid_to_subject = dict(zip(rdata['gid_wgs'].dropna(),
                          rdata.loc[rdata['gid_wgs'].notna(), 'subjectID']))
```

### Step 3: Validate Mapping

```python
# All 785 HuggingFace samples successfully mapped to 212 subjects
# 100% mapping success rate
```

---

## Verification Checklist

- [x] Canonical embeddings exported: `data/processed/microbiome_embeddings_100d.h5`
- [x] SRA run info downloaded to `data/raw/sra_runinfo.csv`
- [x] ID mapping created: `data/processed/srs_to_subject_mapping.csv`
- [x] Unified dataset created: `data/processed/unified_samples.csv`
- [x] All 785 samples mapped to 212 subjects
- [x] TRUE collection months extracted from RData
- [x] Labels validated against RData allergy columns
