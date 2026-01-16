# DIABIMMUNE Food Allergy Classifier

Leakage-safe baseline for predicting **food allergy** (milk/egg/peanut) in the DIABIMMUNE three-country cohort from **16S rRNA OTU** data. A legacy HuggingFace-embeddings baseline is kept for reference.

## Quickstart

```bash
uv sync
uv run python3 scripts/prepare_16s_dataset.py
```

Docs: `docs/MASTER.MD`
Data provenance: `data/README.md`
