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
