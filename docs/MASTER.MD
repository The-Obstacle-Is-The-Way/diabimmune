# DIABIMMUNE Food Allergy Classifier

## Executive Summary

We build a binary classifier that predicts **food allergy development** in infants from the DIABIMMUNE Three‑Country Cohort (FIN/EST/RUS) using gut microbiome data.

---

## Two Modeling Tracks

### Track A: HF Embeddings Baseline (Active — Use This First)

Pre-computed 100-dim embeddings from MicrobeAtlas/HuggingFace with Ludo's corrected metadata.

**Dataset:**
- **785 samples / 212 patients** (WGS-keyed subset)
- Labels: **food allergy only** (`allergen_class`: 0=none, 1=milk, 2=egg, 3=peanut, 4=multiple)
- Each sample appears in exactly one month (fixed by Ludo 2026-01-16)
- Class balance: ~33% allergic overall

**Files:**
- `data/processed/longitudinal_wgs_subset/Month_*.csv` — metadata (sid, patient_id, country, label, allergen_class)
- `data/processed/hf_legacy/microbiome_embeddings_100d.h5` — embeddings (keyed by SRS ID)

**Country distribution:**
| Country | Samples | Allergy Rate |
|---------|---------|--------------|
| FIN | 281 | 49% |
| EST | 199 | 38% |
| RUS | 305 | 15% |

### Track B: 16S OTU Full Dataset (Future — Requires Embedding Generation)

Full DIABIMMUNE 16S data with food-allergy-only labels.

**Dataset:**
- **1,450 samples / 203 subjects** (after excluding missing outcomes)
- Raw OTU table: 1,584 samples × 2,005 Greengenes OTUs
- Labels: `label_food = allergy_milk | allergy_egg | allergy_peanut`

**Files:**
- `data/processed/16s/samples_food_allergy.csv`
- `data/processed/16s/otu_counts.npz`

**Status:** Requires embedding generation (ProkBERT → sample aggregation). See `docs/specs/10_EMBEDDINGS_PIPELINE.md`.

---

## Current Focus: Track A Baseline Notebook

The immediate deliverable is a baseline notebook using Track A data with:
1. Proper subject-level splits (StratifiedGroupKFold, groups=patient_id)
2. Fixed hyperparameters (no inner CV for baseline)
3. Infrastructure supporting future hyperparameter tuning
4. Leave-One-Country-Out (LOCO) as secondary analysis
5. Cumulative horizon analyses (≤3, ≤6, ≤12 months) + “all” association baseline (see `docs/specs/00_HYPOTHESIS.md`)

Data provenance and exact counts live in `docs/data/SOURCE_REGISTRY.md`.

---

## Source of Truth

**Raw data (ground truth):**
- `data/raw/DIABIMMUNE_Karelia_metadata.RData` — metadata + outcomes from Broad Institute
- `data/raw/diabimmune_karelia_16s_otu_table.txt` — OTU counts (Greengenes IDs in row labels)

**Track A processed data (for baseline notebook):**
- `data/processed/longitudinal_wgs_subset/Month_*.csv` — Ludo's corrected metadata
- `data/processed/hf_legacy/microbiome_embeddings_100d.h5` — pre-computed embeddings

**Track B processed data (for future full pipeline):**
- `data/processed/16s/*` — 16S OTU artifacts

---

## Datasets Detail

### Track A: HF Embeddings + Ludo's Corrected Metadata

**Metadata files** (`data/processed/longitudinal_wgs_subset/`):
- 35 files: `Month_1.csv` through `Month_38.csv` (not all months exist)
- Columns: `sid, patient_id, country, label, allergen_class`
- Each sample appears in exactly ONE month (no leakage)
- Labels are **eventual outcome** (not status at collection)

**Embeddings file** (`data/processed/hf_legacy/`):
- `microbiome_embeddings_100d.h5` — 785 samples, 100-dim float32 vectors
- Keys are SRS IDs (e.g., `SRS1719259`)
- Provenance: MicrobeAtlas → ProkBERT → MicrobiomeTransformer (exact method undocumented)

**Join key:** `sid` in CSVs = keys in H5 file (100% alignment verified)

**Label encoding:**
| allergen_class | Meaning |
|----------------|---------|
| 0 | No food allergy |
| 1 | Milk allergy only |
| 2 | Egg allergy only |
| 3 | Peanut allergy only |
| 4 | Multiple food allergies |

**Binary label:** `label = 1` if `allergen_class > 0`, else `0`

### Track B: 16S OTU Full Dataset

Prepared artifacts (rebuildable):
- `data/processed/16s/samples_food_allergy.csv`
- `data/processed/16s/otus_greengenes_ids.csv`
- `data/processed/16s/otu_counts.npz`
- `data/processed/16s/dataset_manifest.json`

Rebuild:
```bash
uv run python3 scripts/prepare_16s_dataset.py --force
```

**Important**: 18 subjects have all food-allergy fields missing; their 16S samples are excluded (1,450 labeled samples remain).

**Status**: Requires embedding generation before modeling. See `docs/specs/10_EMBEDDINGS_PIPELINE.md`.

---

## Modeling Pipeline

### Track A Pipeline (Current Focus)

```
Pre-computed embeddings (785 samples × 100 dims)
  + Ludo's corrected metadata (patient_id, country, label, allergen_class)
  → StratifiedGroupKFold (groups=patient_id)
  → LogReg baseline (class_weight='balanced')
  → Metrics: AUROC, AUPRC, F1
  → LOCO analysis (train on 2 countries, test on 1)
```

### Track B Pipeline (Future)

```
OTU IDs (2,005 Greengenes IDs)
  → Greengenes 13_8 reference sequences
  → ProkBERT (384-dim per OTU)
  → Sample aggregation (100-dim per sample)
  → classifier
```

Notes:
- Greengenes tarball under `_reference/` is not tracked; verify or re-download before use.
- MicrobiomeTransformer pretrained weights may not be available; training may be required.

---

## Evaluation Strategy

### Core Requirements (Leakage Prevention)

1. **Subject-level splits**: `groups=patient_id` — no patient in both train and test
2. **Stratified folds**: `StratifiedGroupKFold` — maintain class balance
3. **Reproducible seeds**: `RANDOM_SEED = 42` for all random operations
4. **Class imbalance**: `class_weight='balanced'` in classifiers

### Cross-Validation Structure

**Baseline evaluation:** 5-fold StratifiedGroupKFold (outer loop only)
- Purpose: Estimate generalization performance
- Groups: `patient_id`
- Stratify: `label`
- Fixed hyperparameters (`C=1.0, class_weight='balanced'`) — no tuning needed

**Optional nested CV (only if hyperparameter tuning is needed later):**
- Inner loop selects hyperparameters on training data only
- Outer loop evaluates on truly unseen test folds
- This is the ONLY correct way to both tune and get unbiased estimates

### Secondary Analyses

1. **LOCO (Leave-One-Country-Out):** Train on 2 countries, test on 1
   - Addresses country confounding (FIN 49% vs RUS 15% allergy rate)

2. **Cumulative horizon analysis:** Evaluate early-life signal using cumulative cutoffs (not disjoint bins)
   - Horizons: `month <= 3`, `month <= 6`, `month <= 12`, plus "all samples"
   - Required procedure: filter to horizon, aggregate to 1 row per infant (mean embedding), then run CV/LOCO
   - Interpretation: earlier horizons support stronger "prediction" framing; later horizons are increasingly confounded by unknown onset timing
   - Track A horizon sizes (verified, patient-level class balance in parentheses):
     - `month <= 3`: 45 samples / 44 patients (23 healthy, 21 allergic)
     - `month <= 6`: 110 samples / 92 patients (51 healthy, 41 allergic)
     - `month <= 12`: 307 samples / 160 patients (101 healthy, 59 allergic)
     - all: 785 samples / 212 patients (144 healthy, 68 allergic)

See `docs/specs/04_EVALUATION.md` for implementation details.

---

## Known Limitations (Track A)

- **Onset timing unavailable**: Labels are endpoint outcomes; we cannot guarantee any sample is pre-diagnosis.
- **Early milk allergy**: Cow’s milk exposure can occur from birth (formula/breast milk), weakening “pure prediction” claims even at very early horizons.
- **Sparse Russia at ≤3 months**: Too few RUS samples/subjects for meaningful LOCO evaluation at that horizon.

## Project Structure

```
diabimmune/
├── notebooks/
│   └── 01_food_allergy_baseline.ipynb  # Track A baseline (to be created)
├── scripts/
│   └── prepare_16s_dataset.py          # Track B dataset prep
├── data/
│   ├── raw/                            # Source of truth (RData, OTU table)
│   └── processed/
│       ├── longitudinal_wgs_subset/    # Track A: Ludo's corrected Month_*.csv
│       ├── hf_legacy/                  # Track A: HF embeddings (100d H5)
│       └── 16s/                        # Track B: 16S OTU artifacts
├── _reference/                         # Untracked reference data
│   ├── greengenes/                     # Greengenes 13_8 tarball
│   └── Microbiome-Modelling/           # Reference model code
├── docs/
│   ├── MASTER.MD                       # This file
│   ├── data/                           # Data documentation
│   └── specs/                          # Technical specifications
├── pyproject.toml
└── uv.lock
```

---

## Provenance References

**Primary data portal:**
- DIABIMMUNE Three‑Country Cohort: https://diabimmune.broadinstitute.org/diabimmune/three-country-cohort
- 16S sequence data: https://diabimmune.broadinstitute.org/diabimmune/three-country-cohort/resources/16s-sequence-data

**Track A data (HF embeddings):**
- HuggingFace dataset: https://huggingface.co/datasets/hugging-science/AI4FA-Diabimmune
- Ludo's preprocessing branch: https://github.com/AI-For-Food-Allergies/gut_microbiome_project/tree/diab-preprocessing

**Embedding pipeline references (Track B):**
- MicrobiomeTransformer: https://github.com/the-puzzler/Microbiome-Modelling
- ProkBERT: https://huggingface.co/neuralbioinfo/prokbert-mini-long

**Documentation:**
- `docs/data/SOURCE_REGISTRY.md` — full provenance registry
- `docs/specs/04_EVALUATION.md` — evaluation strategy
- `docs/specs/10_EMBEDDINGS_PIPELINE.md` — Track B embedding generation
