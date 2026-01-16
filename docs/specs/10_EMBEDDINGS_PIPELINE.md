# 10: Embeddings Pipeline (Planned)

## Goal

Generate deep-learning embeddings from the full DIABIMMUNE 16S OTU table with full provenance:

1) OTU-level sequence embeddings (ProkBERT; 384-dim per OTU)
2) Sample-level embeddings (100-dim per sample) from OTU embeddings + OTU counts

This spec is intentionally separate from the baseline so the baseline remains lightweight and deterministic.

---

## Inputs

Required project artifacts:
- `data/processed/16s/otus_greengenes_ids.csv` (2,005 OTU IDs)
- `data/raw/diabimmune_karelia_16s_otu_table.txt` (raw counts for all 1,584 samples)

External resource (not tracked):
- Greengenes 13_8 tarball (305MB)
  - URL: `http://greengenes.microbio.me/greengenes_release/gg_13_5/gg_13_8_otus.tar.gz`
  - Validation: `gzip -t gg_13_8_otus.tar.gz` must pass
  - File inside tarball used for sequences: `gg_13_8_otus/rep_set/99_otus.fasta` (preferred) or `97_otus.fasta`
  - FASTA headers are integer OTU IDs, e.g. `>1952`

### OTU Matching Rates (Verified 2026-01-15)

| Clustering | OTUs Matched | Match Rate |
|------------|--------------|------------|
| 97% | 1,356 / 2,005 | 68% |
| **99%** | **1,783 / 2,005** | **89%** |
| Unmatched | 222 / 2,005 | 11% |

**Recommendation**: Use 99% clustering (`99_otus.fasta`) for maximum coverage. The 222 unmatched OTUs (11%) may be de novo assignments or version mismatches; handle as missing data or impute with zeros.

Model:
- ProkBERT: `neuralbioinfo/prokbert-mini-long`

---

## Outputs (Proposed)

Write under `data/processed/16s/embeddings/`:

1) `otu_sequences_2005.fasta`
   - FASTA with exactly the 2,005 OTUs used in this dataset
   - header: `>{otu_id}`
   - sequence: DNA letters (A/C/G/T/N)

2) `otu_embeddings_prokbert_384.npy`
   - shape `(2005, 384)` float32
   - row order matches `otus_greengenes_ids.csv` (sorted by `otu_id`)

3) `sample_embeddings_100.npy`
   - shape `(1584, 100)` float32 OR `(1450, 100)` float32
   - if `(1450, 100)`, row order matches `samples_food_allergy.csv`
   - store a sidecar `sample_id.npy` to make alignment unambiguous

4) `dataset_manifest.json`
   - record Greengenes tarball sha256, ProkBERT model revision, and output checksums

---

## Step 1: Extract OTU Sequences From Greengenes

Algorithm:
1. Load OTU IDs from `otus_greengenes_ids.csv`
2. Stream-parse `rep_set/97_otus.fasta` from the tarball
3. For each FASTA record whose header matches an OTU ID, store the sequence
4. Validate:
   - found exactly 2,005 sequences
   - no missing OTUs
   - sequence lengths are reasonable (typically ~1.4kb for Greengenes reps)

If an OTU sequence is missing:
- treat as a hard error (stop) unless you explicitly choose an imputation policy.

---

## Step 2: ProkBERT OTU Embeddings (384-dim)

Algorithm (deterministic):
1. Tokenize each OTU sequence (batching required for speed)
2. Run ProkBERT forward pass
3. Pool token embeddings to a single vector per OTU (must be specified; e.g., mean pooling over tokens)
4. Save `otu_embeddings_prokbert_384.npy` aligned to `otus_greengenes_ids.csv`

Reproducibility requirements:
- record `transformers`, `torch`, and model revision in manifest
- fix random seeds (even though inference should be deterministic)

---

## Step 3: Sample Embeddings (100-dim)

Two acceptable options:

### Option A (preferred if you can train it): MicrobiomeTransformer

- Input per sample:
  - OTU embedding matrix (2,005 × 384)
  - OTU weights (counts or relative abundance)
- Output:
  - 100-dim sample embedding

This requires either:
- pretrained weights (if available), OR
- training the aggregator model on DIABIMMUNE (document the objective and training protocol).

### Option B (fallback, fully deterministic): Weighted mean + PCA

1. Convert counts to relative abundance per sample
2. Compute weighted mean of OTU embeddings → 384-dim per sample
3. Fit PCA to 100 dims **on training data only** (within CV) for modeling, OR
   export only the 384-dim vectors and skip PCA in the offline artifact.

This option is simpler and avoids the “missing pretrained weights” problem.

---

## Evaluation Constraints (must still hold)

- Subject-level CV: `StratifiedGroupKFold(groups=subject_id)`
- If claiming “predict by month m”: only use samples with `collection_month <= m`
- Country generalization: include LOCO evaluation
