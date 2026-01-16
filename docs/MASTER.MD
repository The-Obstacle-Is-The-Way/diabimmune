# DIABIMMUNE Food Allergy Classifier

## Executive Summary

We build a binary classifier that predicts **food allergy development** in infants from the DIABIMMUNE Three‑Country Cohort (FIN/EST/RUS) using **16S rRNA gut microbiome** data.

**Primary modeling dataset (food-allergy-only, 16S):**
- Raw 16S OTU table: **1,584 samples × 2,005 Greengenes OTUs**
- Usable for food-allergy prediction (non-missing outcomes): **1,450 samples / 203 subjects**
- Outcome: `label_food = allergy_milk | allergy_egg | allergy_peanut` (subject-level, propagated to all samples)

**Legacy baseline dataset (do not use for “food allergy only”):**
- HuggingFace embeddings subset: **785 samples / 212 subjects**
- Outcome is broader: `any(allergy_*) | totalige_high`

Data provenance and exact counts live in `data/README.md`.

---

## Source Of Truth

Trust these as ground truth:
- `data/raw/DIABIMMUNE_Karelia_metadata.RData` (metadata + outcomes)
- `data/raw/diabimmune_karelia_16s_otu_table.txt` (OTU counts; Greengenes IDs embedded in row labels)

Treat these as **legacy / convenience only**:
- `data/processed/hf_legacy/` (HuggingFace-derived embeddings subset)

---

## Datasets

### 1) 16S OTU Dataset (Primary)

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

### 2) HuggingFace Embeddings Subset (Legacy)

Prepared artifacts:
- `data/processed/hf_legacy/unified_samples.csv`
- `data/processed/hf_legacy/microbiome_embeddings_100d.h5`
- `data/processed/hf_legacy/srs_to_subject_mapping.csv`
- `data/processed/hf_legacy/dataset_manifest.json`

Rebuild:
```bash
uv run python3 scripts/prepare_hf_legacy.py
```

This subset is WGS-keyed (785 SRS IDs) and includes broken/duplicated Month_* structure upstream; we keep only a canonical exported embedding file.

---

## Pipeline (What We Implement vs What We Plan)

### Implemented: OTU-table baseline (robust + fast)

```
OTU counts (1,450 labeled samples × 2,005 OTUs)
  → relative abundance / transforms
  → classifier (LogReg baseline)
  → subject-level evaluation
```

### Planned: deep embeddings (ProkBERT → MicrobiomeTransformer)

```
OTU IDs (2,005 Greengenes IDs)
  → Greengenes 13_8 reference sequences
  → ProkBERT (384-dim per OTU)
  → MicrobiomeTransformer aggregation (100-dim per sample)
  → classifier
```

Notes:
- Any local Greengenes tarball under `_reference/` is not tracked; verify or re-download before relying on it.
- MicrobiomeTransformer pretrained weights may not be available; training may be required.

---

## Evaluation (Leakage-Safe)

Minimum requirements for scientific validity:
- **Group by subject**: `groups=subject_id` (no infant appears in both train and test).
- **Stratify by outcome**: `StratifiedGroupKFold` to reduce fold imbalance.
- **Time leakage control** (if claiming “predict by month m”): only use samples with `collection_month <= m`.
- **Subject weighting**: avoid subjects with many samples dominating (aggregate to 1 row per subject per horizon, or use per-subject sample weights).
- **Country confounding**: include leave-one-country-out evaluation (food allergy prevalence differs by country).

See `docs/specs/04_EVALUATION.md` for details.

---

## Project Structure (Current)

```
diabimmune/
├── scripts/
│   ├── prepare_16s_dataset.py
│   └── prepare_hf_legacy.py
├── data/
│   ├── raw/
│   └── processed/
│       ├── 16s/
│       └── hf_legacy/
├── docs/
│   ├── MASTER.MD
│   ├── data/
│   └── specs/
├── pyproject.toml
└── uv.lock
```

---

## Provenance References

Primary data portal:
- DIABIMMUNE Three‑Country Cohort: https://diabimmune.broadinstitute.org/diabimmune/three-country-cohort
- 16S sequence data resources: https://diabimmune.broadinstitute.org/diabimmune/three-country-cohort/resources/16s-sequence-data

Embedding references (planned pipeline):
- MicrobiomeTransformer reference code: https://github.com/the-puzzler/Microbiome-Modelling
- Blog post describing the world-model approach: https://the-puzzler.github.io/post.html?p=posts/micro-modelling/micro-modelling.html

Provenance specs:
- `docs/specs/10_EMBEDDINGS_PIPELINE.md`
- `docs/specs/11_PROVENANCE.md`

Source registry:
- `docs/data/SOURCE_REGISTRY.md`
