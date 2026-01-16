# Source Registry (Provenance)

**Last updated**: 2026-01-16

This file is the single “source of truth” registry for external inputs and reference materials.
It complements the machine-readable manifests written by pipeline scripts under `data/processed/**/dataset_manifest.json`.

---

## Primary Data (Broad / DIABIMMUNE)

Project page:
- https://diabimmune.broadinstitute.org/diabimmune/three-country-cohort

16S resources page (download location for the OTU table + 16S RData):
- https://diabimmune.broadinstitute.org/diabimmune/three-country-cohort/resources/16s-sequence-data

Local raw files (sha256, bytes):
- `data/raw/DIABIMMUNE_Karelia_metadata.RData`
  - sha256: `a64f1abd4ac75ada3b6cc52ce13e4b3dbf55f92463ec25ce2e0b7cba8cb0300d`
  - bytes: `702989`
- `data/raw/diabimmune_karelia_16s_otu_table.txt`
  - sha256: `3227ba83d7bac56db561c3aeed83a4660254a510c6e14dd2484a232c8f2230c8`
  - bytes: `7811735`
- `data/raw/diabimmune_karelia_16s_data.rdata`
  - sha256: `ee8ac0c7527ce3fb797c80b51e308fd2f973f8cae00cd5ea24f2ab2a5e0292df`
  - bytes: `3624144`

---

## NCBI SRA (Legacy HF subset bridge only)

BioProject:
- https://www.ncbi.nlm.nih.gov/bioproject/PRJNA290380

RunInfo download endpoint (used by `scripts/prepare_hf_legacy.py`):
- https://www.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?term=PRJNA290380

Local file:
- `data/raw/sra_runinfo.csv`
  - sha256: `369f38fc31ef1d81d13c93abdf215fc0e46c175507d27d585c65fd1730a517a0`
  - bytes: `377029`

---

## Greengenes 13_8 (External reference for OTU sequences)

Download:
- `http://greengenes.microbio.me/greengenes_release/gg_13_5/gg_13_8_otus.tar.gz`

Notes:
- The host’s TLS certificate does not validate for `https://...` (use `http://...` and verify hashes).
- Always verify integrity before use: `gzip -t gg_13_8_otus.tar.gz`.

Local file (not tracked; path is gitignored):
- `_reference/greengenes/gg_13_8_otus.tar.gz`
  - sha256: `d5238392577776fb7e9dd77c5e32df9aae816ef64a462dd725521b5ebf26a70d`
  - bytes: `320067060`

Files used inside tarball:
- `gg_13_8_otus/rep_set/97_otus.fasta` (FASTA; headers are OTU IDs like `>1952`)

---

## ProkBERT (OTU sequence embedding model)

Model ID:
- https://huggingface.co/neuralbioinfo/prokbert-mini-long

Pinned revision (recommended default; record again in the embedding manifest when used):
- `3fd3cf496a13c1485211c7c08d185005233b796b`

---

## MicrobiomeTransformer Reference Implementation

Reference repo:
- https://github.com/the-puzzler/Microbiome-Modelling

Pinned commit (recommended default; record again in the embedding manifest when used):
- `c53f1d71d3543ea75ec193804c78dd40df032d09`

Local clone (not tracked; path is gitignored):
- `_reference/Microbiome-Modelling/`

---

## Blog Post (Conceptual / Protocol Reference)

Not a data source; this is a narrative description of the approach:
- https://the-puzzler.github.io/post.html?p=posts/micro-modelling/micro-modelling.html

Last accessed: 2026-01-16  
Optional page snapshot hash (sha256 of fetched HTML): `0d74a552600b75562aa31fe14bbb7d80d196e57e3c33d4a5bcbe2a928defbe60`

---

## HuggingFace Legacy Embeddings Dataset (Not used for food-only experiments)

Dataset:
- https://huggingface.co/datasets/hugging-science/AI4FA-Diabimmune

Pinned dataset revision (used by `scripts/prepare_hf_legacy.py`):
- `7761eea93dad5712a03452786b43031dc9b04233`

