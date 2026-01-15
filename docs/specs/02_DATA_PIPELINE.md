# 02: Data Pipeline

## Overview

Download embeddings from HuggingFace, extract subject metadata from the RData file, and create a unified dataset ready for training.

---

## Data Sources

| Source | Location | Contents |
|--------|----------|----------|
| **Embeddings** | HuggingFace `hugging-science/AI4FA-Diabimmune` | 100-dim vectors per sample |
| **Labels** | HuggingFace `metadata/Month_*.csv` | `sid`, `label` |
| **Subject IDs** | Local `DIABIMMUNE_Karelia_metadata.RData` | `host_subject_id`, `country`, etc. |

---

## Step 1: Download from HuggingFace

### Option A: Using `huggingface_hub` (Recommended)

```python
from huggingface_hub import snapshot_download
from pathlib import Path

DATA_DIR = Path("data/huggingface")
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Download entire dataset
snapshot_download(
    repo_id="hugging-science/AI4FA-Diabimmune",
    repo_type="dataset",
    local_dir=DATA_DIR / "AI4FA-Diabimmune",
    local_dir_use_symlinks=False,
)
```

### Option B: Using CLI

```bash
# Install huggingface-cli if needed
uv pip install huggingface-hub[cli]

# Download dataset
huggingface-cli download hugging-science/AI4FA-Diabimmune \
    --repo-type dataset \
    --local-dir data/huggingface/AI4FA-Diabimmune
```

### Expected Structure After Download

```
data/huggingface/AI4FA-Diabimmune/
├── metadata/
│   ├── Month_1.csv
│   ├── Month_2.csv
│   ├── Month_3.csv
│   └── ... (up to Month_38.csv)
├── processed/
│   ├── dna_embeddings/
│   │   └── month_{N}/dna_embeddings.h5
│   ├── dna_sequences/
│   │   └── month_{N}/*.csv
│   └── microbiome_embeddings/        # ← USE THESE
│       └── month_{N}/microbiome_embeddings.h5
└── README.md
```

---

## Step 2: Load Embeddings from H5 Files

```python
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import TypedDict


class SampleData(TypedDict):
    sid: str
    embedding: np.ndarray
    label: int


def load_embeddings_for_month(
    data_dir: Path,
    month: int,
) -> dict[str, np.ndarray]:
    """Load microbiome embeddings for a specific month.

    Args:
        data_dir: Path to AI4FA-Diabimmune dataset root
        month: Month number (1-38)

    Returns:
        Dictionary mapping sample_id (SRS*) to 100-dim embedding
    """
    h5_path = data_dir / f"processed/microbiome_embeddings/month_{month}/microbiome_embeddings.h5"

    if not h5_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {h5_path}")

    embeddings: dict[str, np.ndarray] = {}

    with h5py.File(h5_path, "r") as f:
        for sample_id in f.keys():
            embeddings[sample_id] = np.array(f[sample_id])

    return embeddings


def load_labels_for_month(
    data_dir: Path,
    month: int,
) -> pd.DataFrame:
    """Load labels for a specific month.

    Args:
        data_dir: Path to AI4FA-Diabimmune dataset root
        month: Month number (1-38)

    Returns:
        DataFrame with columns: sid, label
    """
    csv_path = data_dir / f"metadata/Month_{month}.csv"

    if not csv_path.exists():
        raise FileNotFoundError(f"Labels not found: {csv_path}")

    return pd.read_csv(csv_path)
```

---

## Step 3: Extract Subject IDs from RData

### Understanding the RData File

The `DIABIMMUNE_Karelia_metadata.RData` file contains subject-level metadata including the critical `host_subject_id` that maps samples to infants.

```python
import pyreadr
import pandas as pd
from pathlib import Path


def load_diabimmune_metadata(rdata_path: Path) -> pd.DataFrame:
    """Load subject metadata from RData file.

    Args:
        rdata_path: Path to DIABIMMUNE_Karelia_metadata.RData

    Returns:
        DataFrame with subject metadata
    """
    result = pyreadr.read_r(rdata_path)

    # RData files can contain multiple objects
    # Print keys to see what's available
    print(f"Objects in RData: {list(result.keys())}")

    # Get the main dataframe (usually the first/only key)
    df_name = list(result.keys())[0]
    metadata_df = result[df_name]

    print(f"Columns: {metadata_df.columns.tolist()}")
    print(f"Shape: {metadata_df.shape}")

    return metadata_df


def explore_metadata(metadata_df: pd.DataFrame) -> None:
    """Print metadata structure for exploration."""
    print("\n=== Metadata Structure ===")
    print(f"Total rows: {len(metadata_df)}")
    print(f"\nColumns ({len(metadata_df.columns)}):")

    for col in metadata_df.columns:
        dtype = metadata_df[col].dtype
        n_unique = metadata_df[col].nunique()
        sample = metadata_df[col].dropna().iloc[0] if len(metadata_df[col].dropna()) > 0 else "N/A"
        print(f"  {col}: {dtype}, {n_unique} unique, sample='{sample}'")
```

### Expected Columns (to verify)

Based on DIABIMMUNE documentation, expect columns like:
- `sampleID` or `sample_accession` (SRS ID)
- `host_subject_id` or `subjectID` (infant ID)
- `country` or `geo_loc_name`
- `age_at_collection` or `age_months`
- Allergy-related columns

---

## Step 4: Create Unified Dataset

```python
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass


@dataclass
class MonthDataset:
    """Dataset for a single month."""
    month: int
    X: np.ndarray              # (n_samples, 100) embeddings
    y: np.ndarray              # (n_samples,) binary labels
    sample_ids: list[str]      # SRS IDs
    subject_ids: list[str]     # Infant IDs (for GroupKFold)
    countries: list[str]       # FIN/EST/RUS


def create_unified_dataset(
    embeddings: dict[str, np.ndarray],
    labels_df: pd.DataFrame,
    metadata_df: pd.DataFrame,
    month: int,
    sample_id_col: str = "sampleID",      # Adjust based on exploration
    subject_id_col: str = "host_subject_id",  # Adjust based on exploration
    country_col: str = "country",          # Adjust based on exploration
) -> MonthDataset:
    """Create unified dataset with embeddings, labels, and subject IDs.

    Args:
        embeddings: Dict mapping SRS ID to embedding vector
        labels_df: DataFrame with sid, label columns
        metadata_df: DataFrame with subject metadata
        month: Month number
        sample_id_col: Column name for sample ID in metadata
        subject_id_col: Column name for subject ID in metadata
        country_col: Column name for country in metadata

    Returns:
        MonthDataset with all required fields
    """
    # Merge labels with metadata
    merged = labels_df.merge(
        metadata_df[[sample_id_col, subject_id_col, country_col]],
        left_on="sid",
        right_on=sample_id_col,
        how="inner",
    )

    # Filter to samples with embeddings
    valid_sids = [sid for sid in merged["sid"] if sid in embeddings]
    merged = merged[merged["sid"].isin(valid_sids)]

    if len(merged) == 0:
        raise ValueError(f"No matching samples found for month {month}")

    # Build arrays
    X = np.array([embeddings[sid] for sid in merged["sid"]])
    y = merged["label"].values
    sample_ids = merged["sid"].tolist()
    subject_ids = merged[subject_id_col].tolist()
    countries = merged[country_col].tolist()

    print(f"Month {month}: {len(X)} samples, {len(set(subject_ids))} subjects")
    print(f"  Class distribution: {np.bincount(y)}")

    return MonthDataset(
        month=month,
        X=X,
        y=y,
        sample_ids=sample_ids,
        subject_ids=subject_ids,
        countries=countries,
    )
```

---

## Step 5: Data Validation

```python
def validate_dataset(dataset: MonthDataset) -> None:
    """Validate dataset integrity."""
    errors: list[str] = []

    # Check shapes
    if dataset.X.shape[0] != len(dataset.y):
        errors.append(f"X/y mismatch: {dataset.X.shape[0]} vs {len(dataset.y)}")

    if dataset.X.shape[1] != 100:
        errors.append(f"Expected 100-dim embeddings, got {dataset.X.shape[1]}")

    # Check for NaN
    if np.isnan(dataset.X).any():
        errors.append(f"NaN values in embeddings")

    # Check labels are binary
    unique_labels = set(dataset.y)
    if unique_labels != {0, 1}:
        errors.append(f"Expected binary labels {{0, 1}}, got {unique_labels}")

    # Check subject IDs present
    if len(dataset.subject_ids) != len(dataset.X):
        errors.append("Subject ID count mismatch")

    if any(sid is None or sid == "" for sid in dataset.subject_ids):
        errors.append("Missing subject IDs detected")

    # Report
    if errors:
        for err in errors:
            print(f"❌ {err}")
        raise ValueError(f"Dataset validation failed with {len(errors)} errors")
    else:
        print(f"✅ Month {dataset.month} dataset valid")
        print(f"   Samples: {len(dataset.X)}")
        print(f"   Subjects: {len(set(dataset.subject_ids))}")
        print(f"   Countries: {set(dataset.countries)}")
```

---

## Complete Pipeline Function

```python
def load_month_data(
    month: int,
    hf_data_dir: Path,
    rdata_path: Path,
) -> MonthDataset:
    """Complete pipeline to load data for a month.

    Args:
        month: Month number (1-38)
        hf_data_dir: Path to HuggingFace dataset
        rdata_path: Path to DIABIMMUNE_Karelia_metadata.RData

    Returns:
        Validated MonthDataset ready for training
    """
    # Load components
    embeddings = load_embeddings_for_month(hf_data_dir, month)
    labels_df = load_labels_for_month(hf_data_dir, month)
    metadata_df = load_diabimmune_metadata(rdata_path)

    # Create unified dataset
    dataset = create_unified_dataset(
        embeddings=embeddings,
        labels_df=labels_df,
        metadata_df=metadata_df,
        month=month,
    )

    # Validate
    validate_dataset(dataset)

    return dataset
```

---

## Output Artifacts

After running the pipeline, save processed data for reproducibility:

```python
def save_processed_metadata(
    dataset: MonthDataset,
    output_path: Path,
) -> None:
    """Save processed metadata CSV with all required columns."""
    df = pd.DataFrame({
        "sid": dataset.sample_ids,
        "label": dataset.y,
        "subject_id": dataset.subject_ids,
        "country": dataset.countries,
    })
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
```

Expected output: `data/processed/Month_{N}_metadata.csv`

```csv
sid,label,subject_id,country
SRS1719091,0,T012374,FIN
SRS1719092,1,T012375,EST
...
```

---

## Verification Checklist

- [ ] HuggingFace download completes (~5-10 min)
- [ ] H5 files readable with `h5py`
- [ ] RData file loads with `pyreadr`
- [ ] Column names identified in metadata
- [ ] SRS → subject_id mapping works
- [ ] No NaN in embeddings
- [ ] Binary labels only (0, 1)
- [ ] Each sample has a subject_id
- [ ] Processed CSVs saved
