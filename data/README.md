# Data Directory

## Quick Overview

```
data/
├── README.md                              # This file
├── raw/                                   # SOURCE DATA
│   ├── DIABIMMUNE_Karelia_metadata.RData  # Subject metadata + allergy labels (222 subjects)
│   ├── diabimmune_karelia_16s_otu_table.txt # OTU counts with Greengenes IDs (1,584 × 2,005)
│   ├── diabimmune_karelia_16s_data.rdata  # Taxonomy-collapsed abundances (1,584 × 282)
│   └── sra_runinfo.csv                    # SRS → gid_wgs mapping (785 rows, HF/WGS-keyed subset only)
└── processed/
    └── 16s/                               # Primary (food allergy, full 16S cohort)
        ├── samples_food_allergy.csv       # 1,450 labeled 16S samples (203 subjects)
        ├── otus_greengenes_ids.csv        # 2,005 Greengenes OTU IDs + taxonomy strings
        ├── otu_counts.npz                 # (1450, 2005) OTU count matrix aligned to samples_food_allergy.csv
        └── dataset_manifest.json          # Provenance + checksums + counts

# Reference data (in _reference/, not tracked in git):
_reference/
├── greengenes/gg_13_8_otus/               # Greengenes 13_8 reference (305MB tar.gz)
│   ├── rep_set/97_otus.fasta              # 99,322 OTUs at 97% clustering
│   ├── rep_set/99_otus.fasta              # Higher resolution clustering
│   └── taxonomy/97_otu_taxonomy.txt       # Taxonomy mapping
├── hf_legacy/                             # Legacy HuggingFace data (deprecated)
│   ├── unified_samples.csv                # 785 samples only (WGS-keyed subset)
│   ├── microbiome_embeddings_100d.h5      # Pre-computed embeddings (unknown provenance)
│   └── dataset_manifest.json              # Provenance notes
└── Microbiome-Modelling/                  # Matteo's aggregation model code
    ├── model.py                           # MicrobiomeTransformer architecture
    └── main.py                            # Training configuration
```

---

## Raw Data Sources

### `DIABIMMUNE_Karelia_metadata.RData`
- **Source**: [Broad Institute DIABIMMUNE Portal](https://diabimmune.broadinstitute.org/diabimmune/three-country-cohort)
- **Contents**: Subject-level metadata for 222 infants across 3 countries
- **Key columns**: `subjectID`, `SampleID`, `gid_wgs`, `collection_month`, `country`, `allergy_*`
- **Trust level**: ✅ Official source
- **Note**: Ensure you comply with the Broad/DIABIMMUNE data usage terms before redistributing any raw files.

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
- **Note**: Only covers the 785-sample HF subset (WGS-keyed list), NOT the full 1,584 16S samples
- **Trust level**: ✅ Official NCBI

---

## Key Numbers (Authoritative)

| Metric | Value |
|--------|-------|
| **16S samples (raw)** | **1,584** |
| **16S subjects (raw)** | **221** |
| **16S samples w/ known food-allergy outcome** | **1,450** |
| **16S subjects w/ known food-allergy outcome** | **203** |
| **Food-allergy positive samples** | **491** (34%) |
| **Food-allergy negative samples** | **959** (66%) |
| **WGS-keyed HF subset samples** | 785 |
| **Unique subjects (full cohort)** | 222 |
| **Greengenes OTUs** | 2,005 |
| **Countries** | 3 (FIN, EST, RUS) |

---

## Processing Pipeline

### Option A: Full Embedding Pipeline (Planned)

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

### Option B: OTU-Table Baseline (Implemented)

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

**Primary task**: FOOD ALLERGIES ONLY, defined at the subject level:

```python
# CORRECT: Food allergies only
label = allergy_milk | allergy_egg | allergy_peanut

# HF legacy baseline label (broader; not food-only)
label = any(allergy_*) | totalige_high
```

**Missing outcomes**: 18 subjects have no observed values for the food-allergy columns; their 16S samples are excluded from `processed/16s/`.

---

## External Resources Needed

| Resource | URL |
|----------|-----|
| Greengenes 13_8 | `http://greengenes.microbio.me/greengenes_release/gg_13_5/gg_13_8_otus.tar.gz` |
| ProkBERT | `neuralbioinfo/prokbert-mini-long` on HuggingFace |
| MicrobiomeTransformer | https://github.com/the-puzzler/Microbiome-Modelling (optionally mirrored under `_reference/`, not tracked) |

---

## Known Issues

1. **Greengenes tarball integrity**: do not assume a local copy is valid; always verify with `gzip -t` before extracting.
2. **Outcome missingness**: 18 subjects have food-allergy columns entirely missing; exclude them (1450 labeled 16S samples remain).
3. **Country confounding**: Food-allergy prevalence differs strongly by country; always include leave-one-country-out evaluation.

---

## OTU-to-Greengenes Matching (Verified 2026-01-15)

| Clustering | OTUs Matched | Match Rate |
|------------|--------------|------------|
| 97% | 1,356 / 2,005 | 68% |
| 99% | 1,783 / 2,005 | 89% |
| Combined (either) | 1,783 / 2,005 | 89% |
| Unmatched | 222 / 2,005 | 11% |

**Note**: The 222 unmatched OTUs may be de novo assignments or version mismatches. Use 99% clustering for maximum coverage.

---

## Key Paper References

| Model | Paper | Embedding Dim |
|-------|-------|---------------|
| **ProkBERT** | Ligeti et al., Frontiers Microbiol 2024 ([PMID:38282738](https://pubmed.ncbi.nlm.nih.gov/38282738/)) | 384 |
| **Set Transformer (abundance-aware)** | Yoo & Rosen, arXiv 2508.11075 | Variable |
| **Microbiome LM** | PLOS Comp Bio, May 2025 ([link](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1011353)) | Variable |

See `docs/data/FULL_PIPELINE_PLAN.md` for complete pipeline documentation.
