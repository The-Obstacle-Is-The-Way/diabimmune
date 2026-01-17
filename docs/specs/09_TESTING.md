# 09: Testing & Validation

## Goal

Make the project “ironclad” by continuously verifying:
- data artifacts are aligned and have expected shapes,
- labels are defined correctly,
- no accidental changes slip in unnoticed.

---

## What We Test

### Data artifact invariants (required)

Tests should verify:
- `samples_food_allergy.csv` rows align with `otu_counts.npz.sample_id`
- `otu_counts.npz.counts` has shape `(n_samples, 2005)`
- `otus_greengenes_ids.csv` has 2,005 unique OTU IDs
- label is constant within each subject

### Notebook invariants (required once notebooks exist)

The baseline notebook is designed to be **assertion-driven**:
- Inline `assert` statements are the notebook’s primary “tests” (fail fast if invariants break).
- The notebook should run top-to-bottom without edits; assertions must pass for a “green” run.

Optional (later, once `notebooks/01_food_allergy_baseline.ipynb` exists):
- Add `nbmake` to execute the notebook under `pytest` so CI can enforce that the inline assertions keep passing.

### Ground-truth property (required)

From `DIABIMMUNE_Karelia_metadata.RData`:
- `allergy_milk|egg|peanut` are constant per subject (within observed values)

---

## Commands

```bash
uv run pytest -q
```

Optional:
- add a notebook execution test via `nbmake` once `notebooks/` exists.
