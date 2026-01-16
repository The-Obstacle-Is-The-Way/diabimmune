# 05: Quality Tooling

## Goal

Keep the notebook + scripts reproducible and reviewable:
- strict formatting + linting
- strict type checking for scripts
- clean notebooks in git

---

## Tooling (Pinned via `uv.lock`)

- `ruff` — lint + format (`pyproject.toml`)
- `mypy` — static typing for `scripts/`
- `pre-commit` — run checks automatically
- `nbstripout` — strip notebook outputs on commit
- `pytest` — enforce data invariants (`docs/specs/09_TESTING.md`)
- Optional: `nbqa` — run ruff on notebooks (once notebooks exist)
- Optional: `pytest` + `nbmake` — execute notebooks in CI (once notebooks exist)

---

## Commands

```bash
uv run ruff check . --fix
uv run ruff format .
uv run mypy scripts
uv run pytest -q
```

---

## Pre-commit Expectations

At minimum:
- ruff + ruff-format
- nbstripout for `*.ipynb`

Once notebooks are in place:
- add nbqa ruff checks for `notebooks/*.ipynb`
