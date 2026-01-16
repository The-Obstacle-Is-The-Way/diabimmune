from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyreadr

FOOD_ALLERGY_COLUMNS = ("allergy_milk", "allergy_egg", "allergy_peanut")


def coerce_bool_or_na(value: Any) -> bool | pd.NA:
    if value is None or pd.isna(value):
        return pd.NA
    if isinstance(value, bool):
        return value
    if isinstance(value, int | np.integer):
        return bool(int(value))
    if isinstance(value, float | np.floating):
        return bool(float(value))
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "t", "1", "yes", "y"}:
            return True
        if v in {"false", "f", "0", "no", "n", ""}:
            return False
    return bool(value)


def test_food_allergy_columns_constant_within_subject() -> None:
    project_root = Path(__file__).resolve().parents[1]
    rdata_path = project_root / "data" / "raw" / "DIABIMMUNE_Karelia_metadata.RData"
    md = pyreadr.read_r(str(rdata_path))["metadata"].copy()

    for col in FOOD_ALLERGY_COLUMNS:
        md[col] = md[col].map(coerce_bool_or_na).astype("boolean")

    for col in FOOD_ALLERGY_COLUMNS:
        nunique = md.groupby("subjectID", sort=False)[col].nunique(dropna=True)
        assert (nunique <= 1).all(), f"{col} varies within at least one subject"
