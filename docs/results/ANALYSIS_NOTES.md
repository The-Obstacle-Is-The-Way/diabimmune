# Results Analysis & Hypotheses

**Author:** Ray
**Date:** 2026-01-17
**Status:** Internal analysis notes (not for external distribution)

---

## Executive Summary

The baseline notebook achieved modest results (AUROC ~0.55-0.62). This document captures our analysis of potential root causes and hypotheses for why the results are not stronger.

---

## Baseline Results Summary

| Horizon | AUROC (mean ± std) | Interpretation |
|---------|-------------------|----------------|
| ≤3 months | 0.575 ± 0.212 | Exploratory (high variance) |
| ≤6 months | 0.553 ± 0.151 | Weak signal |
| ≤12 months | 0.555 ± 0.055 | Weak signal |
| All | 0.617 ± 0.077 | Modest association |

**LOCO Results (concerning):**
- FIN held-out: 0.43 AUROC (worse than random)
- EST held-out: 0.51 AUROC (chance level)
- RUS held-out: 0.62 AUROC (decent, but RUS has lowest allergy rate)

---

## Hypothesis 1: Embedding Model Mismatch

### The Problem

The MicrobiomeTransformer embeddings were trained on **MicrobeAtlas** with a contrastive/denoising objective to learn **bacterial community coherence** - i.e., "which bacteria naturally co-occur together?"

This is fundamentally different from what's needed for allergy prediction.

### Why C-section vs Vaginal Worked (From Ludo's Blog)

- C-section babies: Colonized by skin bacteria (Staphylococcus, Corynebacterium)
- Vaginal babies: Colonized by vaginal/gut bacteria (Lactobacillus, Bifidobacterium)

These are **grossly different** bacterial communities → different embeddings → easy classification.

### Why Allergy Prediction is Harder

Allergic and healthy babies have **very similar** bacterial communities. The differences are subtle:
- Slightly less Bifidobacterium
- Slightly more Enterobacteriaceae
- Reduced diversity
- Lower butyrate producers

Two infants could have communities that look very similar in "coherence space" (similar embeddings) but one has 20% less Bifidobacterium and develops allergy.

### The Abstraction Problem

The 100-dim embedding is an abstract representation where:
- You can't point to dimension 47 and say "that's Bifidobacterium"
- Specific taxa abundances may be "washed out"
- The embedding captures "does this community make sense?" not "how much of each bacteria?"

### Analogy

| Task | Analogy | Embedding works? |
|------|---------|------------------|
| C-section vs vaginal | Forest vs desert | ✅ Totally different |
| Allergy vs healthy | Forest with 100 trees vs forest with 95 trees | ⚠️ Subtle difference |

### What the Published Papers Did Differently

The SOTA papers (phyLoSTM, Metwally et al.) achieved AUROC 0.71-0.76 using:
- **Raw OTU abundance counts** (not pre-computed embeddings)
- **Allergy-specific feature selection** (mRMR, autoencoders)
- **LSTM for longitudinal modeling** (captures temporal trajectories)

They directly measured: "How much Bifidobacterium at each time point?"
We measured: "How coherent is this community according to patterns learned from MicrobeAtlas?"

### Conclusion

The experiment tested: **"Can this coherence-based embedding predict allergy?"**

Answer: Weakly (0.62 AUROC). This is a valid finding, but it's NOT the same as "microbiome doesn't predict allergy."

---

## Hypothesis 2: Sample Size Limitation

### The Problem

We're using **785 samples** (WGS subset) when **~1,450 samples** with allergy labels exist in the 16S dataset.

This was flagged in Discord on 1/15/26:
> "The HuggingFace embeddings dataset has exactly 785 samples - which matches the WGS sample count on DIABIMMUNE, not the 16S count (1,584). If we were to use the embeddings as is, we would be missing ~665 samples."

### Impact

- We're throwing away **~46%** of available labeled samples
- This directly impacts statistical power
- At ≤3 months, we only have **44 patients** → ~9 per test fold → high variance

### Why This Happened

The embeddings were generated from the WGS subset, not the full 16S dataset. This appears to be a data pipeline decision made upstream.

---

## Hypothesis 3: Loss of Temporal Signal

### The Problem

Our approach **averages** embeddings across time points for each infant:
```python
subject_embedding = mean(sample_embeddings for that infant up to month m)
```

This throws away temporal dynamics.

### What LSTM Captures

LSTM learns **patterns of change over time**:
- "Infants whose Bifidobacterium drops between months 3-6 are more likely to develop allergy"
- "A specific trajectory of diversification predicts protection"

### What Our Approach Captures

We only learn:
- "Infants with higher average [embedding dimension 47] are more likely to develop allergy"

The trajectory information is lost.

### Horizon Analysis vs Longitudinal Modeling

| Approach | Question |
|----------|----------|
| Our horizon analysis | "When is the signal strongest?" |
| LSTM longitudinal | "What's the trajectory pattern?" |

These are different questions.

---

## Hypothesis 4: Country Confounding

### The Evidence

LOCO results show the model doesn't generalize across countries:
- Train on EST+RUS, test on FIN → **0.43 AUROC** (worse than random)
- Train on FIN+RUS, test on EST → **0.51 AUROC** (chance)

### Interpretation

The model may be learning country-specific batch effects:
- Sequencing center differences
- Diet differences
- Environmental exposure differences
- Diagnosis criteria differences

Rather than a transferable microbiome → allergy signature.

### Note

This finding is potentially **novel** - we didn't find LOCO analysis reported in the published DIABIMMUNE papers. The published results may be inflated by country-specific effects that wouldn't replicate in external validation.

---

## Summary of Root Causes

| Factor | Impact | Evidence |
|--------|--------|----------|
| Embedding model mismatch | High | Trained for coherence, not allergy-relevant taxa |
| Sample size (785 vs 1,450) | Medium-High | ~46% of labeled data unused |
| Loss of temporal signal | Medium | Averaging destroys trajectories |
| Country confounding | High | LOCO AUROC < 0.50 for FIN |

---

## What Would Improve Results?

1. **Use full 16S dataset** (~1,450 samples with labels)
2. **Use raw OTU abundances** instead of pre-computed embeddings
3. **Add longitudinal modeling** (LSTM or similar)
4. **Allergy-specific feature selection** (mRMR)
5. **Add environmental variables** (diet, antibiotics, delivery mode)

These are documented for completeness but are outside scope of current baseline.

---

## Methodological Validity

Despite modest results, the experiment was methodologically sound:
- Leakage-safe evaluation (subject-level CV)
- Cumulative horizons with clear framing
- LOCO analysis for generalization assessment
- Proper documentation of limitations

The finding "this embedding approach gives modest signal for allergy prediction" is a valid scientific result.

---

## Commit History as Proof

This analysis is documented in this repository with Git commit history as verifiable proof of intellectual contribution. Key commits:
- Data investigation findings
- Notebook implementation
- Results analysis

All timestamps preserved in Git log.
