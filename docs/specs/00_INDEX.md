# Specs Index

## Overview

Vertical slice specifications for the DIABIMMUNE Allergy Classifier notebook.

Each spec is self-contained and can be implemented independently, though they build on each other in order.

---

## Implementation Order

```
┌─────────────────────────────────────────────────────────────────┐
│  01_PROJECT_SETUP.md                                            │
│  ├── pyproject.toml (uv, dependencies)                          │
│  ├── .gitignore                                                  │
│  ├── .pre-commit-config.yaml                                     │
│  └── Directory structure                                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  02_DATA_PIPELINE.md                                            │
│  ├── HuggingFace download                                       │
│  ├── RData extraction (subject_id mapping)                      │
│  ├── H5 embedding loading                                       │
│  └── Unified dataset creation                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  03_NOTEBOOK_STRUCTURE.md                                       │
│  ├── Cell-by-cell specification                                 │
│  ├── Section breakdown                                          │
│  ├── Import organization                                        │
│  └── Configuration constants                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  04_EVALUATION.md                                               │
│  ├── StratifiedGroupKFold implementation                        │
│  ├── Metrics (AUROC, F1, confusion matrix)                      │
│  ├── Per-month analysis                                         │
│  └── Visualization functions                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│  05_QUALITY.md                                                  │
│  ├── ruff configuration                                         │
│  ├── mypy configuration                                         │
│  ├── nbqa for notebooks                                         │
│  └── pre-commit hooks                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Spec Files

| # | File | Description | Time Est |
|---|------|-------------|----------|
| 01 | [PROJECT_SETUP.md](01_PROJECT_SETUP.md) | uv, pyproject.toml, directory structure | 15 min |
| 02 | [DATA_PIPELINE.md](02_DATA_PIPELINE.md) | Download, load, merge data sources | 30 min |
| 03 | [NOTEBOOK_STRUCTURE.md](03_NOTEBOOK_STRUCTURE.md) | Cell-by-cell notebook specification | 45 min |
| 04 | [EVALUATION.md](04_EVALUATION.md) | CV strategy, metrics, visualizations | 30 min |
| 05 | [QUALITY.md](05_QUALITY.md) | Linting, formatting, pre-commit | 15 min |

**Total estimated implementation time: ~2-3 hours**

---

## Dependencies Between Specs

```
01_PROJECT_SETUP ──→ Required by all others
        ↓
02_DATA_PIPELINE ──→ Required by 03, 04
        ↓
03_NOTEBOOK_STRUCTURE ──→ Uses 02, 04
        ↓
04_EVALUATION ──→ Used by 03
        ↓
05_QUALITY ──→ Applies to final notebook
```

---

## Quick Start

After reading all specs, implementation order:

1. **Setup project** (01)
   ```bash
   uv venv && source .venv/bin/activate
   uv pip install -e ".[dev]"
   pre-commit install
   ```

2. **Explore data** (02)
   - Download from HuggingFace
   - Load RData, identify column names
   - Test data loading functions

3. **Build notebook** (03)
   - Create sections 0-2 first (setup, data loading)
   - Verify data pipeline works
   - Add sections 3-6 (training, viz, export)

4. **Verify evaluation** (04)
   - Confirm no subject overlap
   - Check class balance
   - Validate metrics

5. **Final polish** (05)
   - Run `pre-commit run --all-files`
   - Strip notebook outputs
   - Commit clean notebook

---

## Output Artifacts

After implementation, you'll have:

```
diabimmune/
├── pyproject.toml              # From 01
├── .pre-commit-config.yaml     # From 01, 05
├── .gitignore                  # From 01
├── notebooks/
│   └── 01_baseline_classifier.ipynb  # From 03
├── data/                       # From 02 (gitignored)
│   └── huggingface/
│       └── AI4FA-Diabimmune/
├── results/                    # From 03 (gitignored)
│   ├── summary.csv
│   ├── detailed_results.csv
│   ├── auroc_over_time.png
│   ├── roc_curves.png
│   └── confusion_matrices.png
└── docs/
    ├── MASTER.MD
    └── specs/
        ├── 00_INDEX.md         # This file
        ├── 01_PROJECT_SETUP.md
        ├── 02_DATA_PIPELINE.md
        ├── 03_NOTEBOOK_STRUCTURE.md
        ├── 04_EVALUATION.md
        └── 05_QUALITY.md
```

---

## Success Criteria

The notebook is complete when:

- [ ] Runs top-to-bottom without errors
- [ ] Uses subject-level CV (no data leakage)
- [ ] Reports AUROC per month with std
- [ ] Generates all visualizations
- [ ] Passes all pre-commit hooks
- [ ] Outputs are stripped before commit
- [ ] Results are reproducible (seeds set)
