# Notebooks

Primary notebook (planned): `01_food_allergy_baseline.ipynb`

Design goals:
- runs top-to-bottom,
- no network calls,
- uses `data/processed/16s/` artifacts,
- leakage-safe evaluation (`StratifiedGroupKFold` with `groups=subject_id`).

See `docs/specs/03_NOTEBOOK_STRUCTURE.md`.
