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
  prepare_hf_legacy.py       # optional legacy HF baseline prep
data/
  raw/                       # source data (checked into this repo)
  processed/
    16s/                     # primary processed artifacts
    hf_legacy/               # legacy artifacts
docs/
  specs/                     # these files
```
