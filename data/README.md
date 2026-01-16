# Data Directory

## Quick Overview

```
data/
├── README.md                              # This file
├── raw/                                   # SOURCE DATA
│   ├── DIABIMMUNE_Karelia_metadata.RData  # Subject metadata + allergy labels (222 subjects)
│   ├── diabimmune_karelia_16s_otu_table.txt # OTU counts with Greengenes IDs (1,584 × 2,005)
│   ├── diabimmune_karelia_16s_data.rdata  # Taxonomy-collapsed abundances (1,584 × 282)
│   └── sra_runinfo.csv                    # SRS → gid_wgs mapping (785 rows, WGS subset only)
└── processed/                             # LEGACY (from HuggingFace - may be deprecated)
    ├── unified_samples.csv                # 785 samples only (WGS subset)
    ├── srs_to_subject_mapping.csv         # ID mapping for WGS subset
    ├── microbiome_embeddings_100d.h5      # Pre-computed embeddings (785 keys, provenance unclear)
    └── dataset_manifest.json              # Provenance for processed files
```

---

## Raw Data Sources

### `DIABIMMUNE_Karelia_metadata.RData`
- **Source**: [Broad Institute DIABIMMUNE Portal](https://diabimmune.broadinstitute.org/diabimmune/three-country-cohort)
- **Contents**: Subject-level metadata for 222 infants across 3 countries
- **Key columns**: `subjectID`, `SampleID`, `gid_wgs`, `collection_month`, `country`, `allergy_*`
- **Trust level**: ✅ Official source

### `diabimmune_karelia_16s_otu_table.txt`
- **Source**: [Broad Institute DIABIMMUNE Portal - 16S Sequence Data](https://diabimmune.broadinstitute.org/diabimmune/three-country-cohort/resources/16s-sequence-data)
- **Contents**: Raw OTU counts for 1,584 samples × 2,005 Greengenes OTUs
- **Format**: Tab-separated, rows = taxonomic lineages with OTU IDs, columns = sample IDs
- **OTU ID format**: `k__Bacteria|p__...|g__...|s__unclassified|470690` (number at end is Greengenes ID)
- **Trust level**: ✅ Official source

### `diabimmune_karelia_16s_data.rdata`
- **Source**: Same as above
- **Contents**: Taxonomy-collapsed relative abundances (1,584 × 282)
- **Note**: This is aggregated to taxonomy level, NOT individual OTUs
- **Trust level**: ✅ Official source

### `sra_runinfo.csv`
- **Source**: [NCBI SRA BioProject PRJNA290380](https://www.ncbi.nlm.nih.gov/bioproject/PRJNA290380)
- **Contents**: Maps SRS IDs (HuggingFace) → LibraryName (gid_wgs)
- **Note**: Only covers 785 WGS samples, NOT the full 1,584 16S samples
- **Trust level**: ✅ Official NCBI

---

## Key Numbers

| Metric | Value |
|--------|-------|
| **16S samples** | **1,584** |
| **WGS samples** | 785 |
| **Unique subjects** | 222 |
| **Greengenes OTUs** | 2,005 |
| **Countries** | 3 (FIN, EST, RUS) |

---

## Processing Pipeline

### Option A: Full Embedding Pipeline (Recommended)

```
diabimmune_karelia_16s_otu_table.txt
    ↓
Extract 2,005 Greengenes OTU IDs
    ↓
Look up 16S sequences in Greengenes 13_8
    ↓
Run ProkBERT → 384-dim per OTU
    ↓
Aggregate with MicrobiomeTransformer → 100-dim per sample
    ↓
Train classifier with proper food allergy labels
```

### Option B: Direct Classification (Simpler)

```
diabimmune_karelia_16s_otu_table.txt
    ↓
Normalize to relative abundance
    ↓
Use 2,005-dim feature vector directly
    ↓
Train classifier (no embeddings)
```

---

## Label Definition

**IMPORTANT**: Define labels from RData using FOOD ALLERGIES ONLY:

```python
# CORRECT: Food allergies only
label = allergy_milk | allergy_egg | allergy_peanut

# WRONG: What HuggingFace used (includes environmental)
label = any(allergy_*) | totalige_high
```

---

## External Resources Needed

| Resource | URL |
|----------|-----|
| Greengenes 13_8 | `ftp://greengenes.microbio.me/greengenes_release/gg_13_5/gg_13_8_otus.tar.gz` |
| ProkBERT | `neuralbioinfo/prokbert-mini-long` on HuggingFace |
| MicrobiomeTransformer | `_reference/Microbiome-Modelling/` (local copy) |

---

## Known Issues

1. **processed/ directory is legacy** - Based on HuggingFace 785-sample WGS subset
2. **Sample ID mismatch** - OTU table uses internal IDs (3100170), RData uses same
3. **Need Greengenes** - Must download to get 16S sequences for ProkBERT

See `docs/data/FULL_PIPELINE_PLAN.md` for complete pipeline documentation.
