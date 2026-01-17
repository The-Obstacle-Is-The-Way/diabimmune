# 00: Scientific Hypotheses (Track A Baseline)

## Goal

Lock the scientific questions and experiments **before** writing the baseline notebook.

This spec is written for **Track A** (HF embeddings + Ludo’s corrected `Month_*.csv` metadata).

---

## What “Month_N” Means (Track A)

Each `data/processed/longitudinal_wgs_subset/Month_N.csv` file represents stool samples collected at approximately month `N` of an infant’s life (collection timing).

Important: this repo’s Track A **label is an endpoint outcome** repeated across all samples for the same infant (see `docs/specs/07_OUTCOME_DEFINITION.md`).

---

## Research Questions

### Primary (Association Baseline)
Is the gut microbiome **associated** with eventual food allergy development (endpoint label), when using all available samples across infancy/early childhood?

### Secondary (Clinical Utility)
How early in life can the gut microbiome (as measured in early stool samples) **predict** the endpoint food allergy label?

---

## Hypotheses

### H0 (Null)
Microbiome embeddings contain no usable signal for the endpoint food allergy label (performance ≈ chance).

### H1: Association Baseline (All Samples)
- **Data**: all Track A samples pooled (months 1–38; not all months exist)
- **Claim strength**: **association only**
- **Why**: late samples can occur after disease onset/management and may reflect the allergic state (reverse causation)

### H2: Early-Life (≤3 months) — Exploratory / “Most Predictive”
- **Data**: samples with `month <= 3`
- **Claim strength**: **strongest predictive framing** (earliest window)
- **Why**: typically pre-solid foods for most infants, but see limitations (milk allergy can manifest earlier)

### H3: Early-Life (≤6 months)
- **Data**: samples with `month <= 6`
- **Claim strength**: **predictive framing**, but weaker than ≤3 months
- **Why**: closer to solid food introduction; some infants may already be symptomatic/diagnosed (unknown in our data)

### H4: First Year (≤12 months)
- **Data**: samples with `month <= 12`
- **Claim strength**: **mixed predictive/associative**
- **Why**: many food allergies can manifest during the first year; without onset timing, this window may include post-diagnosis samples

---

## Experimental Design (Cumulative Horizons)

We do **cumulative horizons**, not disjoint bins:
- ✅ **Correct**: `month <= m` for `m ∈ {3, 6, 12}`, plus “all”
- ❌ **Not used**: “months 7–12 only”, “13–24 only”, etc. (those do not answer “how early can we predict?” and also suffer from small-N per bin)

For each horizon:
1. Filter to `month <= m` (or keep all samples for the association baseline)
2. Aggregate to **one row per infant** (mean embedding over available samples in that horizon)
3. Evaluate with leakage-safe splits (see `docs/specs/04_EVALUATION.md`)

---

## Track A Sample Counts (Verified)

Counts are computed from `data/processed/longitudinal_wgs_subset/Month_*.csv` with `month` taken from the filename.

| Horizon | Samples | Patients | Label=0 | Label=1 | Notes |
|---------|---------|----------|---------|---------|------|
| ≤3 months | 45 | 44 | 23 | 22 | Only 3 RUS samples (LOCO for RUS not meaningful) |
| ≤6 months | 110 | 92 | 58 | 52 | Viable class balance |
| ≤12 months | 307 | 160 | 203 | 104 | Viable |
| All | 785 | 212 | 527 | 258 | Association baseline |

---

## Interpretation Framework

| Horizon | If AUROC > 0.5 | Strength of claim |
|---------|----------------|------------------|
| ≤3 months | “Very early microbiome signal precedes endpoint allergy label” | Strongest (still limited by unknown onset) |
| ≤6 months | “Early-life microbiome signal precedes endpoint allergy label” | Moderate |
| ≤12 months | “First-year microbiome relates to endpoint allergy label” | Weak-to-moderate (mixed pre/post onset) |
| All | “Microbiome differs between endpoint-allergic vs non-allergic” | Association only |

---

## Known Limitations (Must Be Stated)

1. **Onset timing is unknown**: The Track A label is an endpoint outcome; we do not have reliable “diagnosis month / first reaction month” for each infant. Any horizon may include post-onset samples for some infants.
2. **Milk allergy can occur very early**: Infants can be exposed to cow’s milk protein via formula (and sometimes via breast milk). This weakens “pure prediction” claims even at very early horizons.
3. **Country imbalance at ≤3 months**: Only 3 RUS samples; LOCO results for RUS at this horizon are not meaningful.
4. **Month is approximate**: `Month_N.csv` is a pragmatic bin; it is not a precise clinical onset timeline.

---

## Related Literature (Pointers; not exhaustive)

Use these as anchors when writing the final Results/Discussion:
- Early-life microbiome maturation in the first year is repeatedly implicated in later allergy risk (multiple cohorts; designs often use 3‑month / 1‑year sampling).
- DIABIMMUNE allergy modeling papers typically treat the task as **subject-level longitudinal prediction**, with attention to country prevalence gradients and variable sampling schedules.

