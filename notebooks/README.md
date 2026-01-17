# Notebooks

Primary notebook: `01_food_allergy_baseline.ipynb` (Track A baseline)

Design goals:
- runs top-to-bottom,
- no network calls,
- self-contained + assertion-driven,
- leakage-safe evaluation (subject-level aggregation + `StratifiedGroupKFold` with `groups=patient_id`),
- LOCO analysis to probe country confounding.

Inputs (Track A):
- `data/processed/longitudinal_wgs_subset/Month_*.csv`
- `data/processed/hf_legacy/microbiome_embeddings_100d.h5`

Outputs (committed for transplantability):
- `notebooks/results/cv_metrics.csv`
- `notebooks/results/cv_summary.csv`
- `notebooks/results/loco_metrics.csv`

See `docs/specs/03_NOTEBOOK_STRUCTURE.md`.
