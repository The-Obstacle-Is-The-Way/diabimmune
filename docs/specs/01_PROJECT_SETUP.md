# 01: Project Setup

## Goal

Modern, reproducible Python setup for a strict notebook-driven project:
- `uv` for dependency management (locked)
- `ruff` for lint + format
- `mypy` for type checking
- `pre-commit` for automation
- notebook hygiene via `nbstripout`

---

## Quickstart

```bash
uv sync
pre-commit install
```

Run data prep:
```bash
uv run python3 scripts/prepare_16s_dataset.py
```

Run tests:
```bash
uv run pytest -q
```

---

## Key Files (As Implemented)

- `pyproject.toml` — deps + ruff + mypy config
- `uv.lock` — pinned, reproducible environment
- `.pre-commit-config.yaml` — automated checks
- `.python-version` — Python version pin

---

## Directory Layout (Relevant)

```
scripts/
  prepare_16s_dataset.py     # primary dataset prep
data/
  raw/                       # source data (checked into this repo)
  processed/
    16s/                     # primary processed artifacts
_reference/                  # untracked reference data
  greengenes/                # Greengenes 13_8 (downloaded)
  hf_legacy/                 # deprecated HuggingFace data
  Microbiome-Modelling/      # reference model code
docs/
  specs/                     # these files
```
