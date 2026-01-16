# Specs Index

These specs define a minimal, reproducible workflow for the **DIABIMMUNE food allergy classifier**.

Primary dataset: **16S OTU table** (1,584 samples × 2,005 Greengenes OTUs) with **food-allergy-only** labels (1,450 labeled samples after excluding missing outcomes).

Legacy dataset: **HuggingFace embeddings subset** (785 samples) retained for reference only.

---

## Implementation Order

1. `01_PROJECT_SETUP.md` — uv + strict tooling
2. `02_DATA_PIPELINE.md` — prepare 16S + (optional) HF legacy artifacts
3. `03_NOTEBOOK_STRUCTURE.md` — notebook layout (baseline LogReg)
4. `04_EVALUATION.md` — leakage-safe evaluation (StratifiedGroupKFold, time horizons, LOCO)
5. `05_QUALITY.md` — lint/format/typecheck + notebook hygiene
6. `06_DATA_SCHEMA.md` — file formats + invariants
7. `07_OUTCOME_DEFINITION.md` — label + missingness policy
8. `08_FEATURE_ENGINEERING.md` — OTU transforms (no leakage)
9. `09_TESTING.md` — tests that enforce invariants
10. `10_EMBEDDINGS_PIPELINE.md` — ProkBERT + sample embeddings (planned)
11. `11_PROVENANCE.md` — source registry + provenance manifests

---

## Expected Artifacts

After running data prep scripts:

**Primary (16S / food allergy):**
- `data/processed/16s/samples_food_allergy.csv`
- `data/processed/16s/otus_greengenes_ids.csv`
- `data/processed/16s/otu_counts.npz`
- `data/processed/16s/dataset_manifest.json`

**Legacy (HF embeddings subset — DEPRECATED, moved to `_reference/`):**
- `_reference/hf_legacy/unified_samples.csv`
- `_reference/hf_legacy/microbiome_embeddings_100d.h5`
- `_reference/hf_legacy/srs_to_subject_mapping.csv`
- `_reference/hf_legacy/dataset_manifest.json`

Note: These files are untracked. Do not use for "food allergy only" experiments.
