# Data Investigation Findings

**Date**: 2026-01-15
**Status**: Legacy HuggingFace subset resolved (canonical embeddings + ground-truth metadata)

**Note**: The primary project direction is now the full 16S OTU dataset + food-allergy-only labels.
See `docs/MASTER.MD` and `scripts/prepare_16s_dataset.py`.

---

## TL;DR - What's Going On

We have **two data sources** that don't play nicely together:

1. **DIABIMMUNE_Karelia_metadata.RData** (from Broad Institute)
   - The "ground truth" - has everything: subjects, samples, allergies, countries
   - BUT: Uses internal sample IDs (like `3101193`), not SRS IDs
   - You already had this file

2. **HuggingFace AI4FA-Diabimmune** (from Matteo's project)
   - Has the embeddings we want (100-dim vectors from ProkBERT/MicrobiomeTransformer)
   - Uses SRS IDs (like `SRS1719091`)
   - **Missing subject IDs** - can't do proper ML without them
   - **Month_N folders are BROKEN** - same samples duplicated everywhere

**The core problem**: To use the embeddings, we need to map SRS IDs → subject IDs. This requires going through NCBI SRA metadata as a bridge.

---

## What We Downloaded from HuggingFace

### What we wanted:
- Just the embeddings (100-dim vectors)

### What we got:

Upstream repo: https://huggingface.co/datasets/hugging-science/AI4FA-Diabimmune
Pinned revision (recommended): `7761eea93dad5712a03452786b43031dc9b04233`

The upstream dataset layout (as published) contains:

```
metadata/Month_*.csv
processed/microbiome_embeddings/Month_*/microbiome_embeddings.h5
```

We treat HuggingFace as a source of *embeddings only* and do **not** rely on the Month_* CSVs for subject IDs, timing, or label definition.

### The Problem with HuggingFace Data

#### Issue 1: No Subject IDs
Their CSV files only have:
```csv
sid,label
SRS1719091,0
SRS1719092,1
```

**Missing**: `subject_id`, `country`, `collection_month`

Without `subject_id`, we can't do `StratifiedGroupKFold` → **data leakage**.

#### Issue 2: Month_N Folders Are BROKEN

**What we expected**: Month_1 = samples collected at month 1, Month_2 = month 2, etc.

**What we found**:
```
Sample SRS1719502 appears in: Month_1, Month_4, Month_7, Month_10, Month_13,
                              Month_16, Month_19, Month_22, Month_28, Month_36
The embedding vectors are NOT guaranteed identical across folders (small numeric differences exist).
```

**784 of 785 samples appear in multiple "Month" folders.**

This means the Month_N structure is **not** collection month. It might be:
- Data augmentation?
- Experimental variants?
- A bug in their pipeline?

**We don't know what it means.**

#### Issue 3: Label Definition is Broader Than Expected

Their `label=1` (allergic) includes:
- Food allergies (milk, egg, peanut) ← what we expected
- Environmental allergies (dustmite, cat, dog, birch, timothy) ← unexpected
- High total IgE ← unexpected

**This is "any allergy" not "food allergy".**

---

## What's in the RData File (Ground Truth)

```
DIABIMMUNE_Karelia_metadata.RData
├── 1,946 rows (sample-level data)
├── 222 unique subjects
├── 59 columns including:
│   ├── subjectID (E*, T*, P* format) ← what we need
│   ├── SampleID (3101193 format) ← NOT SRS format!
│   ├── gid_wgs (G* format) ← KEY for mapping!
│   ├── collection_month (1-38)
│   ├── country (FIN, EST, RUS)
│   ├── allergy_milk, allergy_egg, allergy_peanut (True/False)
│   └── ... many more
```

**The gid_wgs column** is the bridge - it matches the SRA LibraryName.

---

## How We Mapped SRS → Subject ID

Since HuggingFace uses SRS IDs and RData uses internal IDs, we needed a bridge:

```
SRS1719091 (HuggingFace)
    ↓
[Query NCBI SRA for PRJNA290380]
    ↓
LibraryName = G78508 (from SRA metadata)
    ↓
gid_wgs = G78508 (matches RData column)
    ↓
subjectID = T012374 (from RData)
```

**We downloaded**: `data/raw/sra_runinfo.csv` (NCBI SRA metadata for BioProject PRJNA290380)

**Result**: 785 HuggingFace samples → 212 unique subjects (100% mapping success)

---

## What We Created (Processed Data)

To make this usable, we created:

### `data/processed/hf_legacy/unified_samples.csv`
The "clean" dataset with everything linked:
```csv
srs_id,gid_wgs,subject_id,collection_month,age_days,country,label
SRS1719087,G69146,E016030,28.0,848.0,EST,0
```

### `data/processed/hf_legacy/srs_to_subject_mapping.csv`
Just the ID mapping for reference.

### `data/processed/hf_legacy/microbiome_embeddings_100d.h5`
A single canonical embedding file (one 100-d vector per SRS, 785 keys) created by selecting, for each sample, the vector from `Month_{collection_month}` where `collection_month` is taken from the RData ground truth.

---

## Current State Assessment

### What's Good ✅
- Embeddings exist and are usable (100-dim, float32)
- We have the mapping from SRS → subject_id
- RData file has ground truth for everything
- Data sizes are small (~8 MB total)

### What's Concerning ⚠️
- HuggingFace Month_N structure is meaningless/broken
- Label definition includes environmental allergies (broader than "food allergy")
- We're trusting embeddings we didn't generate

### What's Bad ❌
- Provenance is non-trivial (HF snapshot + SRA bridge), but is now reproducible via `scripts/prepare_hf_legacy.py`
- Complex mapping chain with external dependency (SRA)
- Can't verify embedding quality without regenerating

---

## Decision Points

### Option A: Trust the Embeddings, Fix the Labels
- Use HuggingFace embeddings as-is
- Use `data/processed/hf_legacy/unified_samples.csv` for proper subject grouping
- Accept that "allergic" = any allergy, not just food

**Pros**: Fast, data is ready
**Cons**: Black box embeddings, broad label definition

### Option B: Extract Everything from RData, Generate Own Embeddings
- Start from RData as sole source of truth
- Download raw 16S sequences from SRA
- Run ProkBERT + MicrobiomeTransformer ourselves

**Pros**: Full control, narrow label definition possible
**Cons**: Much more work, need to set up embedding pipeline

### Option C: Verify Embeddings First
- Check if embeddings make sense (clustering, PCA)
- Spot-check a few samples against raw data
- Then decide whether to trust them

**Pros**: Middle ground
**Cons**: Still some unknown

---

## My Honest Assessment

**The HuggingFace data is usable but sketchy.**

The embeddings themselves are probably fine - they come from a real model (ProkBERT → MicrobiomeTransformer). But the metadata and folder structure are poorly organized.

**The RData file is the source of truth** for subject info, allergy labels, and collection timing.

**For a quick baseline classifier**, Option A is fine - use their embeddings with our proper subject mapping.

**For publication-quality work**, Option B is safer - regenerate everything from scratch.

---

## Files Summary

| File | Source | Purpose | Trust Level |
|------|--------|---------|-------------|
| `data/raw/DIABIMMUNE_Karelia_metadata.RData` | Broad Institute | Ground truth metadata | ✅ High |
| `data/raw/sra_runinfo.csv` | NCBI SRA | ID mapping bridge | ✅ High |
| `data/processed/hf_legacy/microbiome_embeddings_100d.h5` | HuggingFace (exported locally) | Embeddings (one vector per SRS) | ⚠️ Medium |
| `data/processed/hf_legacy/srs_to_subject_mapping.csv` | Us (from NCBI SRA + RData) | SRS → gid_wgs → subjectID | ✅ High |
| `data/processed/hf_legacy/unified_samples.csv` | Us | Clean merged data | ✅ High |

---

## Missing: Provenance Script

This repo includes a single reproducible entrypoint script (`scripts/prepare_hf_legacy.py`) that:
- downloads the HF embeddings at a pinned revision (or uses local files),
- rebuilds `data/processed/hf_legacy/srs_to_subject_mapping.csv` and `data/processed/hf_legacy/unified_samples.csv` from the RData + SRA runinfo,
- exports `microbiome_embeddings_100d.h5`,
- validates row/subject counts and key alignment.
