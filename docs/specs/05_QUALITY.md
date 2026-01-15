# 05: Quality Tooling

## Overview

Configure modern Python quality tools for notebooks: linting, formatting, type checking, and pre-commit hooks.

---

## Tool Stack

| Tool | Purpose | Runs On |
|------|---------|---------|
| **ruff** | Linting + formatting (replaces flake8, isort, black) | `.py` files |
| **mypy** | Static type checking | `.py` files |
| **nbqa** | Run linters on notebooks | `.ipynb` files |
| **nbstripout** | Strip outputs before commit | `.ipynb` files |
| **pre-commit** | Git hook automation | All files |

---

## Ruff Configuration

Ruff is the fastest Python linter, replacing flake8, isort, and black.

### In `pyproject.toml`

```toml
[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # pyflakes
    "I",      # isort (import sorting)
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
    "ARG",    # flake8-unused-arguments
    "SIM",    # flake8-simplify
    "TCH",    # flake8-type-checking
    "PTH",    # flake8-use-pathlib
    "NPY",    # numpy-specific rules
    "PD",     # pandas-vet
]

ignore = [
    "E501",   # line too long (formatter handles)
    "B008",   # function call in default argument
    "PD901",  # df is a bad variable name (we use it intentionally)
]

[tool.ruff.lint.isort]
known-first-party = ["diabimmune"]
force-single-line = false
lines-after-imports = 2

[tool.ruff.format]
quote-style = "double"
indent-style = "space"
skip-magic-trailing-comma = false
line-ending = "auto"
```

### Usage

```bash
# Check for issues
ruff check .

# Fix auto-fixable issues
ruff check --fix .

# Format code
ruff format .
```

---

## Mypy Configuration

Static type checking catches bugs before runtime.

### In `pyproject.toml`

```toml
[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
no_implicit_optional = true
warn_redundant_casts = true
warn_unused_ignores = true
show_error_codes = true
show_column_numbers = true

# Strict mode for our code
[[tool.mypy.overrides]]
module = "notebooks.*"
ignore_errors = true  # Notebooks are exploratory

# Third-party packages without stubs
[[tool.mypy.overrides]]
module = [
    "pyreadr.*",
    "h5py.*",
    "huggingface_hub.*",
    "seaborn.*",
]
ignore_missing_imports = true
```

### Usage

```bash
# Check all Python files
mypy src/

# Check specific file
mypy src/data_loader.py
```

---

## nbqa: Notebook Quality

Run any linter on Jupyter notebooks.

### Installation

Already in dev dependencies:
```bash
uv pip install nbqa
```

### Usage

```bash
# Run ruff on notebooks
nbqa ruff notebooks/

# Auto-fix issues
nbqa ruff --fix notebooks/

# Check formatting
nbqa ruff format --check notebooks/

# Format notebooks
nbqa ruff format notebooks/
```

### What nbqa Does

```
notebook.ipynb
    ↓
[nbqa extracts code cells]
    ↓
cell_1.py, cell_2.py, ...
    ↓
[ruff analyzes each cell]
    ↓
[nbqa writes fixes back]
    ↓
notebook.ipynb (updated)
```

---

## nbstripout: Clean Notebooks

Remove outputs before committing to keep git history clean.

### Installation

```bash
uv pip install nbstripout

# Install as git filter (run once per repo)
nbstripout --install
```

### How It Works

When you `git add notebook.ipynb`:
1. Git filter intercepts the file
2. nbstripout removes all cell outputs
3. Clean notebook (code only) is staged
4. Outputs remain in your working copy

### Manual Usage

```bash
# Strip a specific notebook
nbstripout notebook.ipynb

# Check if notebook has outputs
nbstripout --is-stripped notebook.ipynb
```

### Configuration in `.gitattributes`

After `nbstripout --install`, this is added:

```gitattributes
*.ipynb filter=nbstripout
*.ipynb diff=ipynb
```

---

## Pre-commit Configuration

Automate all checks on every commit.

### File: `.pre-commit-config.yaml`

```yaml
# See https://pre-commit.com for more information
default_language_version:
  python: python3.11

repos:
  # Standard hooks
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-json
      - id: check-toml
      - id: check-added-large-files
        args: ['--maxkb=1000']
      - id: check-merge-conflict
      - id: detect-private-key

  # Ruff (linting + formatting)
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix, --exit-non-zero-on-fix]
      - id: ruff-format

  # nbqa for notebooks
  - repo: https://github.com/nbQA-dev/nbQA
    rev: 1.7.1
    hooks:
      - id: nbqa-ruff
        args: [--fix]

  # Strip notebook outputs
  - repo: https://github.com/kynan/nbstripout
    rev: 0.6.1
    hooks:
      - id: nbstripout
        files: '\.ipynb$'
```

### Setup

```bash
# Install pre-commit
uv pip install pre-commit

# Install hooks (run once per repo clone)
pre-commit install

# Run on all files (first time or manual check)
pre-commit run --all-files
```

### What Happens on Commit

```
git commit
    ↓
[pre-commit hooks run]
    ↓
1. trailing-whitespace: fixes whitespace
2. end-of-file-fixer: ensures newline at EOF
3. check-yaml/json/toml: validates config files
4. check-added-large-files: blocks >1MB files
5. ruff: lints Python files
6. ruff-format: formats Python files
7. nbqa-ruff: lints notebooks
8. nbstripout: removes notebook outputs
    ↓
[If all pass] → commit proceeds
[If any fail] → commit blocked, fixes shown
```

---

## Notebook Best Practices

### Type Hints in Notebooks

```python
# Cell: Function with type hints
def load_data(path: Path) -> pd.DataFrame:
    """Load data from path.

    Args:
        path: Path to CSV file

    Returns:
        DataFrame with loaded data
    """
    return pd.read_csv(path)
```

### Magic Commands

```python
# Cell: Setup with magic commands
%load_ext autoreload
%autoreload 2

# Ensures imported modules reload on change
```

### Suppress Warnings

```python
# Cell: Clean output
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

### Progress Bars

```python
# For long loops
from tqdm.notebook import tqdm

for month in tqdm(MONTHS_TO_EVALUATE, desc="Evaluating"):
    # ...
```

---

## CI Integration (Optional)

If you want GitHub Actions:

### File: `.github/workflows/quality.yml`

```yaml
name: Quality Checks

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  quality:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install uv
        run: curl -LsSf https://astral.sh/uv/install.sh | sh

      - name: Install dependencies
        run: |
          uv venv
          source .venv/bin/activate
          uv pip install -e ".[dev]"

      - name: Run ruff
        run: |
          source .venv/bin/activate
          ruff check .
          ruff format --check .

      - name: Run nbqa
        run: |
          source .venv/bin/activate
          nbqa ruff notebooks/
```

---

## Quick Reference

### Daily Commands

```bash
# Before committing
pre-commit run --all-files

# Check code quality
ruff check .

# Format code
ruff format .

# Check notebooks
nbqa ruff notebooks/

# Strip notebook outputs manually
nbstripout notebooks/*.ipynb
```

### Fix Common Issues

| Issue | Fix |
|-------|-----|
| Import order wrong | `ruff check --fix --select I .` |
| Formatting issues | `ruff format .` |
| Unused imports | `ruff check --fix --select F401 .` |
| Notebook has outputs | `nbstripout notebook.ipynb` |

---

## Verification Checklist

- [ ] `uv pip install -e ".[dev]"` installs all dev tools
- [ ] `pre-commit install` completes
- [ ] `pre-commit run --all-files` passes
- [ ] `ruff check .` returns no errors
- [ ] `nbqa ruff notebooks/` returns no errors
- [ ] Git commits trigger pre-commit hooks
- [ ] Notebook outputs are stripped on commit
