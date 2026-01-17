# 07: Outcome Definition (Food Allergy Only)

## Goal

Define a binary "food allergy development" target that is:
- derived from the DIABIMMUNE ground truth,
- leakage-safe,
- explicit about missingness.

Applies to both tracks (with different source files).

---

## Food Allergy Columns

We define “food allergy” using **only**:
- `allergy_milk`
- `allergy_egg`
- `allergy_peanut`

No environmental allergies and no total IgE are included in the primary task.

---

## Key Empirical Fact (DIABIMMUNE RData)

Across the `DIABIMMUNE_Karelia_metadata.RData` file, these food-allergy columns are **constant per subject** (within observed values). This means they behave as **subject-level outcomes** repeated across sample rows, not as time-varying labels.

Implications:
- The label represents “ever/endpoint food allergy status” for the subject.
- There is no reliable “onset month” in these columns; do not interpret the earliest observed month as onset.

---

## Missingness Policy (Required)

Some subjects have **no observed values** (all missing) for all three food-allergy columns.

Policy:
- If a subject has no observed values for `allergy_milk|egg|peanut`, that subject’s samples are **excluded** from the labeled modeling dataset.

This yields:
- 1,584 total 16S samples
- 1,450 labeled 16S samples (after excluding fully-missing subjects)

See `data/processed/16s/dataset_manifest.json`.

---

## Formal Definition

For a subject `s`, define:

```python
label_food(s) = 1 if any(allergy_milk(s), allergy_egg(s), allergy_peanut(s)) is True
               0 if all observed values are False
               NA if no observed values exist for the subject
```

For a sample row `i` belonging to subject `s(i)`:

```python
label_food_sample(i) = label_food(s(i))
```

---

## Track A: allergen_class Encoding (Ludo's Metadata)

In `data/processed/longitudinal_wgs_subset/Month_*.csv`, the label is encoded as:

| Column | Meaning |
|--------|---------|
| `label` | Binary: 0 = no food allergy, 1 = has food allergy |
| `allergen_class` | Multi-class encoding (see below) |

**allergen_class values:**

| Value | Meaning |
|-------|---------|
| 0 | No food allergy |
| 1 | Milk allergy only |
| 2 | Egg allergy only |
| 3 | Peanut allergy only |
| 4 | Multiple food allergies |

**Relationship:**
```python
label = 1 if allergen_class > 0 else 0
```

**Note:** This is **eventual outcome** (endpoint allergy status), not status at sample collection time. Ludo fixed this in the 2026-01-16 preprocessing update.

---

## Modeling Unit (Strong Recommendation)

Because each subject has multiple samples, evaluation must be **subject-level**:
- CV grouping: `groups = patient_id` (Track A) or `groups = subject_id` (Track B)
- If doing time-horizon analysis: aggregate to 1 row per subject per horizon (mean or last sample up to horizon)

See `docs/specs/04_EVALUATION.md`.

---

## Label Consistency Verification

Before modeling, verify labels are consistent per patient:

```python
labels_per_patient = df.groupby("patient_id")["label"].nunique()
assert (labels_per_patient == 1).all(), "Inconsistent labels within patient!"
```

This should always pass because labels are subject-level (eventual outcome), not time-varying.
