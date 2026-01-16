# 11: Provenance (End-to-End)

## Goal

Make the full DIABIMMUNE pipeline audit-grade by ensuring every artifact is reproducible and traceable to:
- a **source URL** (or human download instruction),
- a **pinned revision** (commit hash / dataset revision) when applicable,
- a **cryptographic hash** (sha256) of local inputs and outputs,
- an **exact command line** that produced it.

This spec defines the *scripts we should have* to prove provenance across the entire chain. Do not implement these scripts unless explicitly requested.

---

## Provenance Contract (Non-negotiable)

Every pipeline script must:
1) Accept `--out-manifest <path>` (default: alongside outputs).
2) Record:
   - `created_at` (UTC ISO8601)
   - `command` (argv)
   - `git` (repo root, branch, commit, dirty flag)
   - `python` (version)
   - `uv_lock_sha256` (sha256 of `uv.lock`)
   - `inputs`: list of `{path, sha256, bytes, source_url?, notes?}`
   - `external_sources`: list of `{name, url, revision?, sha256? (if downloaded)}`
   - `outputs`: list of `{path, sha256, bytes}`
   - `dataset_stats` (counts and invariants relevant to that step)
3) Fail fast on mismatches (missing columns, wrong counts, missing IDs).

---

## Source Registry (Single File)

Add and maintain a single human-auditable registry:

`docs/data/SOURCE_REGISTRY.md` (or `provenance/sources.json` if you prefer machine-readability).

It must include, at minimum:
- DIABIMMUNE portal link(s) used to obtain:
  - `DIABIMMUNE_Karelia_metadata.RData`
  - `diabimmune_karelia_16s_otu_table.txt`
  - `diabimmune_karelia_16s_data.rdata`
- Greengenes 13_8 tarball URL (`http://greengenes.microbio.me/.../gg_13_8_otus.tar.gz`)
- ProkBERT model ID (`neuralbioinfo/prokbert-mini-long`) + model revision used for embeddings
- MicrobiomeTransformer reference repo (`https://github.com/the-puzzler/Microbiome-Modelling`) + commit pinned
- Blog post reference (conceptual/protocol): `https://the-puzzler.github.io/post.html?p=posts/micro-modelling/micro-modelling.html` + “last accessed” date

---

## Required Provenance Scripts (Specs Only)

### A) Raw-input verification (already possible; formalize)

**Script**: `scripts/verify_raw_inputs.py`
**Purpose**: Validate that `data/raw/*` matches expected shapes/columns and write a manifest with sha256 hashes.

Inputs:
- `data/raw/DIABIMMUNE_Karelia_metadata.RData`
- `data/raw/diabimmune_karelia_16s_otu_table.txt`
- `data/raw/diabimmune_karelia_16s_data.rdata` (optional; verify presence + hash)

Outputs:
- `data/raw/dataset_manifest.json` (new) with:
  - RData shape (expected 1946×59), subjects (222)
  - OTU table dimensions (1584 samples; 2005 Greengenes OTUs detected by `|<int>` row labels)

### B) 16S labeled dataset prep (already implemented)

**Script**: `scripts/prepare_16s_dataset.py` (exists)
**Requirement**: its manifest must include:
- source URLs (DIABIMMUNE portal)
- counts: 1584 total 16S samples, 1450 labeled samples, 203 labeled subjects, 2005 OTUs
- country+label breakdown (already included in `data/processed/16s/dataset_manifest.json`)

### C) Greengenes acquisition + verification

**Script**: `scripts/fetch_greengenes_13_8.py`
**Purpose**: Download Greengenes tarball, verify `gzip -t`, and write sha256 to a manifest.

Inputs:
- URL: `http://greengenes.microbio.me/greengenes_release/gg_13_5/gg_13_8_otus.tar.gz`

Outputs:
- `_reference/greengenes/gg_13_8_otus.tar.gz` (not tracked)
- `_reference/greengenes/greengenes_manifest.json` (sha256, bytes, URL, retrieved_at)

### D) OTU sequence extraction

**Script**: `scripts/extract_greengenes_otu_sequences.py`
**Purpose**: Create a FASTA containing exactly the 2,005 OTU sequences used in this project.

Inputs:
- `_reference/greengenes/gg_13_8_otus.tar.gz`
- `data/processed/16s/otus_greengenes_ids.csv`

Outputs:
- `data/processed/16s/embeddings/otu_sequences_2005.fasta`
- `data/processed/16s/embeddings/otu_sequences_manifest.json`

Validations:
- exactly 2005 sequences
- headers match OTU IDs
- no missing OTUs

### E) ProkBERT OTU embeddings

**Script**: `scripts/embed_otus_prokbert.py`
**Purpose**: Generate deterministic 384-dim OTU embeddings.

Inputs:
- `data/processed/16s/embeddings/otu_sequences_2005.fasta`
- ProkBERT model: `neuralbioinfo/prokbert-mini-long` (pin exact revision)

Outputs:
- `data/processed/16s/embeddings/otu_embeddings_prokbert_384.npy`
- `data/processed/16s/embeddings/otu_embeddings_manifest.json`

Manifest requirements:
- model ID + revision
- transformers/torch versions
- pooling method used (must be specified)

### F) Sample-level embeddings (100-dim)

**Script**: `scripts/build_sample_embeddings.py`
**Purpose**: Produce sample embeddings from OTU embeddings + OTU counts with a fully documented method.

Inputs:
- `data/processed/16s/otu_counts.npz`
- `data/processed/16s/embeddings/otu_embeddings_prokbert_384.npy`
- method config:
  - `microbiome_transformer` (requires training/provided weights) OR
  - `weighted_mean` (+ optional PCA inside CV only)

Outputs (proposed):
- `data/processed/16s/embeddings/sample_embeddings_*.{npy,h5}`
- `data/processed/16s/embeddings/sample_embeddings_manifest.json`

### G) Training + evaluation runner (non-notebook)

**Script**: `scripts/run_experiment.py`
**Purpose**: Run the full evaluation from CLI and write machine-readable results.

Inputs:
- features source (OTU baseline or embeddings)
- evaluation config (horizons, aggregation, LOCO, seeds)

Outputs:
- `results/metrics_by_horizon.csv`
- `results/metrics_loco.csv`
- `results/run_manifest.json` (must reference the data/embeddings manifests)

---

## Extra Provenance: Blog Post + Reference Repo

Because the blog post is not versioned, record:
- URL
- last accessed date
- sha256 of fetched HTML (optional; store hash only)

For `Microbiome-Modelling`, record:
- repo URL
- commit hash used
- any patches applied (diff hash)

---

## What This Gives You

At any point you can answer:
- “Which exact raw files were used (hash)?”
- “Which exact external model/repo revision was used?”
- “Which exact command produced these outputs?”
- “Can a third party re-run it and get the same artifacts?”
