---
license: apache-2.0
---

# Food Allergy Microbiome Dataset (Experimental)

## Dataset Summary

This is an **experimental microbiome dataset** designed for exploratory research in **food allergy classification**. The dataset contains multiple data modalities (DNA embeddings, microbiome embeddings, raw DNA sequences) collected longitudinally at several timepoints.

> **Warning:** This dataset is experimental. Its structure is frozen for ongoing research, and it is **not ready for benchmarking**.

---

## Dataset Structure

The dataset is organized by **data type** and **timepoint**:

```
.
├── dna_embeddings
│   └── month_{1,2,3,....38}/dna_embeddings.h5
├── dna_sequences
│   └── month_{1,2,3,....38}/*.csv
└── microbiome_embeddings
    └── month_{1,2,3,....38}/microbiome_embeddings.h5
```

* **DNA embeddings**: `.h5` files with embedding vectors derived from DNA sequences.
* **Microbiome embeddings**: `.h5` files containing microbiome feature vectors.
* **DNA sequences**: raw `.csv` files representing sequences or processed features.

**Each timepoint contains multiple samples per subject.**
File names (e.g., `SRS1719092.csv`) serve as sample IDs. Subject IDs and mappings are implicit; users must manage them carefully.

---

## Intended Use

* Task: Exploratory **classification of food allergies**.
* Users are expected to **define their own train/test splits**.
* **Critical:** Do **not** split samples randomly; multiple samples per subject exist. Splits should be done at the **subject level** to avoid data leakage.

---

## Data Notes

* Longitudinal: Samples are collected at multiple months (1, 2, 3, 6, 12, 24, 36).
* Multi-modal: Embeddings and sequences are provided separately; users may combine them as needed.
* No labels are embedded per file. Labels must be handled separately or mapped from your internal records.
* This dataset is **research-focused**, not benchmark-ready.

---

## License

This experimental dataset is currently released under Apache License 2.0

---