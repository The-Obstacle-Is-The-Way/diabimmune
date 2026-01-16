# 01: Project Setup

## Overview

Initialize a modern Python project with `uv` for dependency management, strict type checking, and reproducible environments.

---

## File: `pyproject.toml`

```toml
[project]
name = "diabimmune-classifier"
version = "0.1.0"
description = "Allergy/atopy classifier using DIABIMMUNE microbiome embeddings"
readme = "README.md"
license = { text = "MIT" }
requires-python = ">=3.12"
authors = [
    { name = "Your Name", email = "you@example.com" }
]

dependencies = [
    # Core ML
    "scikit-learn>=1.4.0",
    "numpy>=1.26.0",
    "pandas>=2.2.0",

    # Data loading
    "h5py>=3.10.0",           # Read .h5 embedding files
    "pyreadr>=0.5.0",         # Read .RData metadata file
    "huggingface-hub>=0.20.0", # Download from HuggingFace

    # Visualization
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",

    # Jupyter
    "jupyterlab>=4.0.0",
    "ipykernel>=6.28.0",
]

[project.optional-dependencies]
dev = [
    # Linting & formatting
    "ruff>=0.1.9",

    # Type checking
    "mypy>=1.8.0",
    "pandas-stubs>=2.1.4",

    # Notebook QA
    "nbqa>=1.7.1",            # Run linters on notebooks
    "nbstripout>=0.6.1",      # Strip outputs before commit

    # Pre-commit hooks
    "pre-commit>=3.6.0",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = [
    "E",      # pycodestyle errors
    "W",      # pycodestyle warnings
    "F",      # pyflakes
    "I",      # isort
    "B",      # flake8-bugbear
    "C4",     # flake8-comprehensions
    "UP",     # pyupgrade
    "ARG",    # flake8-unused-arguments
    "SIM",    # flake8-simplify
]
ignore = [
    "E501",   # line too long (handled by formatter)
    "B008",   # function call in default argument
]

[tool.ruff.lint.isort]
known-first-party = ["diabimmune"]

[tool.mypy]
python_version = "3.12"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
disallow_incomplete_defs = true
check_untyped_defs = true
ignore_missing_imports = true  # For packages without stubs

[[tool.mypy.overrides]]
module = "pyreadr.*"
ignore_missing_imports = true

[[tool.mypy.overrides]]
module = "h5py.*"
ignore_missing_imports = true
```

---

## File: `.python-version`

```
3.12
```

---

## File: `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/

# uv
.uv/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Data (downloaded, not tracked)
data/
huggingface_datasets/

# Results (generated, not tracked)
results/
eval_results/

# IDE
.idea/
.vscode/
*.swp
*.swo
.DS_Store

# mypy
.mypy_cache/

# ruff
.ruff_cache/

# Environment variables
.env
.env.local
```

---

## File: `.pre-commit-config.yaml`

```yaml
repos:
  - repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v4.5.0
    hooks:
      - id: trailing-whitespace
      - id: end-of-file-fixer
      - id: check-yaml
      - id: check-added-large-files
        args: ['--maxkb=1000']

  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.1.9
    hooks:
      - id: ruff
        args: [--fix]
      - id: ruff-format

  - repo: https://github.com/nbQA-dev/nbQA
    rev: 1.7.1
    hooks:
      - id: nbqa-ruff
        args: [--fix]

  - repo: https://github.com/kynan/nbstripout
    rev: 0.6.1
    hooks:
      - id: nbstripout
```

---

## Setup Commands

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Create virtual environment and install dependencies (uses uv.lock)
uv sync
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows

# 3. Install pre-commit hooks
pre-commit install

# 4. Register Jupyter kernel
python -m ipykernel install --user --name diabimmune --display-name "DIABIMMUNE Classifier"

# 5. Verify installation
python -c "import sklearn, pandas, h5py, pyreadr; print('All imports OK')"
```

---

## Directory Structure After Setup

```
diabimmune/
├── .git/
├── .gitignore
├── .pre-commit-config.yaml
├── .python-version
├── .venv/                          # Created by uv
├── pyproject.toml
├── LICENSE
├── README.md
├── scripts/
│   └── prepare_data.py
├── docs/
│   ├── MASTER.MD
│   └── specs/
│       ├── 01_PROJECT_SETUP.md     # This file
│       └── ...
├── notebooks/
│   └── 01_baseline_classifier.ipynb
├── data/
│   ├── raw/
│   │   ├── DIABIMMUNE_Karelia_metadata.RData
│   │   └── sra_runinfo.csv
│   └── processed/
│       ├── unified_samples.csv
│       ├── srs_to_subject_mapping.csv
│       └── microbiome_embeddings_100d.h5
└── results/                        # gitignored, created at runtime
    └── ...
```

---

## Verification Checklist

- [ ] `uv sync` creates `.venv/` and installs dependencies
- [ ] `pre-commit run --all-files` passes
- [ ] `python -c "import sklearn, pandas, h5py, pyreadr"` works
- [ ] JupyterLab launches with `jupyter lab`
- [ ] Kernel "DIABIMMUNE Classifier" appears in kernel list
