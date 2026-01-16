# 08: Feature Engineering (OTU Counts)

## Goal

Define leakage-safe feature transforms for OTU count data.

Key constraints:
- high-dimensional (2,005 features),
- sparse with many zeros,
- compositional (per-sample total depth varies).

---

## Recommended Baselines

### Baseline A (simple + common): Relative abundance

For each sample row `x`:

```python
x_rel = x / x.sum()
```

Then model with:
- `StandardScaler` (fit on train fold only)
- `LogisticRegression(class_weight="balanced", solver="liblinear")`

### Baseline B (sometimes strong): log1p counts

```python
x_log = log1p(x)
```

Then same model pipeline as above.

---

## Compositional Option (More principled): CLR transform

CLR requires a pseudocount because of zeros:

```python
x_clr = log(x + eps) - mean(log(x + eps))
```

Notes:
- `eps` must be fixed and documented (e.g., `1.0`).
- Compute per sample (no fitting), then scale in the train fold.

---

## Leakage Rules (Non-negotiable)

- Any scaler/normalizer that learns parameters from the dataset (mean/std, PCA, feature selection) must be fit **only on the training split**.
- Per-sample transforms (relative abundance, log1p, CLR) are safe to compute before splitting as long as they do not use labels and do not pool across samples.

---

## Optional: OTU Filtering

Filtering rare OTUs can reduce noise, but do it correctly:
- Define filtering rules using **training data only** (within each fold), OR
- Predefine filtering thresholds without looking at labels (still document clearly).
