from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyreadr

FOOD_ALLERGY_COLUMNS = ("allergy_milk", "allergy_egg", "allergy_peanut")


@dataclass(frozen=True)
class Paths:
    project_root: Path
    raw_dir: Path
    processed_dir: Path
    rdata_path: Path
    otu_table_path: Path
    samples_csv: Path
    otus_csv: Path
    otu_counts_npz: Path
    manifest_json: Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def relpath(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def coerce_bool_or_none(value: Any) -> bool | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(int(value))
    if isinstance(value, (float, np.floating)):
        return bool(float(value))
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "t", "1", "yes", "y"}:
            return True
        if v in {"false", "f", "0", "no", "n", ""}:
            return False
    return bool(value)


def load_rdata_metadata(path: Path) -> pd.DataFrame:
    r = pyreadr.read_r(str(path))
    if "metadata" not in r:
        raise KeyError(f"Expected key 'metadata' in {path}, found: {list(r.keys())}")
    metadata = r["metadata"]
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError(f"Expected 'metadata' to be a pandas DataFrame, got {type(metadata)}")
    return metadata


def build_food_allergy_labels(metadata: pd.DataFrame) -> pd.Series:
    missing = {"subjectID", *FOOD_ALLERGY_COLUMNS} - set(metadata.columns)
    if missing:
        raise KeyError(f"Missing required columns in RData metadata: {sorted(missing)}")

    df = metadata.loc[:, ["subjectID", *FOOD_ALLERGY_COLUMNS]].copy()
    for col in FOOD_ALLERGY_COLUMNS:
        df[col] = df[col].map(coerce_bool_or_none).astype("boolean")

    has_observed = (
        df.loc[:, FOOD_ALLERGY_COLUMNS]
        .notna()
        .groupby(df["subjectID"], sort=False)
        .any()
        .any(axis=1)
    )
    any_true = (
        df.loc[:, FOOD_ALLERGY_COLUMNS]
        .fillna(False)
        .groupby(df["subjectID"], sort=False)
        .any()
        .any(axis=1)
    )

    labels = pd.Series(pd.NA, index=has_observed.index, dtype="Int64", name="label_food")
    labels.loc[has_observed] = any_true.loc[has_observed].astype(int)
    return labels


def filter_16s_samples(metadata: pd.DataFrame) -> pd.DataFrame:
    required = {
        "SampleID",
        "subjectID",
        "collection_month",
        "age_at_collection",
        "country",
        "gender",
        "read_count_16S",
    }
    missing = required - set(metadata.columns)
    if missing:
        raise KeyError(f"Missing required columns in RData metadata: {sorted(missing)}")

    df = metadata.copy()
    df["read_count_16S"] = pd.to_numeric(df["read_count_16S"], errors="coerce")
    df = df.loc[df["read_count_16S"].fillna(0) > 0].copy()

    df = df.rename(
        columns={
            "SampleID": "sample_id",
            "subjectID": "subject_id",
            "age_at_collection": "age_days",
        }
    )
    df["sample_id"] = df["sample_id"].astype(str)
    df["subject_id"] = df["subject_id"].astype(str)
    df["collection_month"] = pd.to_numeric(df["collection_month"], errors="raise").astype(int)
    df["age_days"] = pd.to_numeric(df["age_days"], errors="raise").astype(int)
    df["country"] = df["country"].astype(str)
    df["gender"] = df["gender"].astype(str)

    if not df["sample_id"].is_unique:
        raise ValueError("Expected unique SampleID values after 16S filtering.")
    return df


def load_otu_table(otu_path: Path) -> pd.DataFrame:
    if not otu_path.exists():
        raise FileNotFoundError(f"Missing OTU table: {otu_path}")
    df = pd.read_csv(otu_path, sep="\t", index_col=0)
    df.columns = df.columns.map(str)
    df.index = df.index.map(str)
    return df


def extract_greengenes_otus(otu_table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    pattern = re.compile(r".*\|(\d+)$")
    records: list[dict[str, object]] = []
    for row_label in otu_table.index:
        m = pattern.match(row_label)
        if m:
            records.append({"otu_id": int(m.group(1)), "row_label": str(row_label)})

    otus = pd.DataFrame.from_records(records)
    if len(otus) == 0:
        raise ValueError("No Greengenes OTU IDs found in OTU table row labels.")
    if not otus["otu_id"].is_unique:
        raise ValueError("Duplicate Greengenes OTU IDs found in OTU table row labels.")

    otus = otus.sort_values("otu_id", kind="mergesort").reset_index(drop=True)
    otus_out = otus.rename(columns={"row_label": "taxonomy"})
    return otus, otus_out


def build_otu_counts_matrix(
    *,
    otu_table: pd.DataFrame,
    otus: pd.DataFrame,
    sample_ids: list[str],
) -> np.ndarray:
    missing = set(sample_ids) - set(map(str, otu_table.columns))
    if missing:
        raise KeyError(
            "OTU table missing expected sample_id columns "
            f"(n={len(missing)}): {sorted(missing)[:10]}"
        )

    row_labels = otus["row_label"].tolist()
    counts = otu_table.loc[row_labels, sample_ids].T.to_numpy(dtype=np.int32)
    return counts


def resolve_paths(project_root: Path) -> Paths:
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed" / "16s"
    return Paths(
        project_root=project_root,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        rdata_path=raw_dir / "DIABIMMUNE_Karelia_metadata.RData",
        otu_table_path=raw_dir / "diabimmune_karelia_16s_otu_table.txt",
        samples_csv=processed_dir / "samples_food_allergy.csv",
        otus_csv=processed_dir / "otus_greengenes_ids.csv",
        otu_counts_npz=processed_dir / "otu_counts.npz",
        manifest_json=processed_dir / "dataset_manifest.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare DIABIMMUNE 16S OTU table dataset for food allergy prediction."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild outputs even if they already exist.",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    paths = resolve_paths(project_root)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)

    if not paths.rdata_path.exists():
        raise FileNotFoundError(f"Missing {paths.rdata_path}.")
    if not paths.otu_table_path.exists():
        raise FileNotFoundError(f"Missing {paths.otu_table_path}.")

    if (
        not args.force
        and paths.samples_csv.exists()
        and paths.otu_counts_npz.exists()
        and paths.otus_csv.exists()
    ):
        print(f"✅ Outputs already exist under {paths.processed_dir}. Use --force to rebuild.")
        return

    print("Loading RData metadata...")
    metadata = load_rdata_metadata(paths.rdata_path)

    print("Filtering to 16S samples...")
    samples_16s = filter_16s_samples(metadata)

    print("Building subject-level food allergy labels...")
    label_by_subject = build_food_allergy_labels(metadata)
    samples_16s["label_food"] = samples_16s["subject_id"].map(label_by_subject)

    labeled = samples_16s.loc[samples_16s["label_food"].notna()].copy()
    labeled["label_food"] = labeled["label_food"].astype(int)

    labeled = labeled.sort_values(["subject_id", "collection_month", "sample_id"], kind="mergesort")
    sample_ids = labeled["sample_id"].tolist()

    print("Loading OTU table...")
    otu_table = load_otu_table(paths.otu_table_path)

    print("Extracting 2,005 Greengenes OTU IDs from row labels...")
    otus_internal, otus_out = extract_greengenes_otus(otu_table)
    if len(otus_internal) != 2005:
        raise ValueError(f"Expected 2005 Greengenes OTUs, found {len(otus_internal)}")

    print("Building sample×OTU count matrix...")
    counts = build_otu_counts_matrix(otu_table=otu_table, otus=otus_internal, sample_ids=sample_ids)

    if counts.shape != (len(sample_ids), 2005):
        raise ValueError(f"Unexpected counts shape: {counts.shape}")

    # Write artifacts
    otus_out.to_csv(paths.otus_csv, index=False)
    labeled[
        [
            "sample_id",
            "subject_id",
            "collection_month",
            "age_days",
            "country",
            "gender",
            "label_food",
        ]
    ].to_csv(paths.samples_csv, index=False)

    np.savez_compressed(
        paths.otu_counts_npz,
        counts=counts,
        sample_id=np.asarray(sample_ids),
        otu_id=otus_internal["otu_id"].to_numpy(dtype=np.int32),
    )

    subject_frame = labeled.drop_duplicates("subject_id")[
        ["subject_id", "country", "label_food"]
    ].copy()
    subject_counts_by_country = subject_frame["country"].value_counts().to_dict()
    subject_label_counts_by_country = (
        subject_frame.groupby(["country", "label_food"]).size().unstack(fill_value=0).to_dict()
    )

    sample_counts_by_country = labeled["country"].value_counts().to_dict()
    sample_label_counts_by_country = (
        labeled.groupby(["country", "label_food"]).size().unstack(fill_value=0).to_dict()
    )

    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "sources": {
            "rdata": {
                "path": relpath(paths.rdata_path, paths.project_root),
                "sha256": sha256_file(paths.rdata_path),
            },
            "otu_table": {
                "path": relpath(paths.otu_table_path, paths.project_root),
                "sha256": sha256_file(paths.otu_table_path),
                "format": "TSV with taxonomy row labels; Greengenes ID is final |<int> segment",
            },
        },
        "label_definition": {
            "name": "food_allergy_only",
            "positive_if_any_true": list(FOOD_ALLERGY_COLUMNS),
            "unit": "subject-level (propagated to all samples)",
            "missing_handling": "subjects with no observed food allergy outcomes are excluded",
        },
        "dataset": {
            "n_samples_16s_total": int(len(samples_16s)),
            "n_subjects_16s_total": int(samples_16s["subject_id"].nunique()),
            "n_samples_labeled": int(len(labeled)),
            "n_subjects_labeled": int(labeled["subject_id"].nunique()),
            "sample_label_counts": {
                str(k): int(v) for k, v in labeled["label_food"].value_counts().to_dict().items()
            },
            "subject_label_counts": {
                str(k): int(v)
                for k, v in (
                    labeled.drop_duplicates("subject_id")
                    .set_index("subject_id")["label_food"]
                    .value_counts()
                    .to_dict()
                    .items()
                )
            },
            "subjects_by_country": {str(k): int(v) for k, v in subject_counts_by_country.items()},
            "subjects_by_country_and_label": {
                str(k): {str(kk): int(vv) for kk, vv in v.items()}
                for k, v in subject_label_counts_by_country.items()
            },
            "samples_by_country": {str(k): int(v) for k, v in sample_counts_by_country.items()},
            "samples_by_country_and_label": {
                str(k): {str(kk): int(vv) for kk, vv in v.items()}
                for k, v in sample_label_counts_by_country.items()
            },
            "n_otus": int(len(otus_internal)),
        },
        "artifacts": {
            "samples_csv": {
                "path": relpath(paths.samples_csv, paths.project_root),
                "sha256": sha256_file(paths.samples_csv),
            },
            "otus_csv": {
                "path": relpath(paths.otus_csv, paths.project_root),
                "sha256": sha256_file(paths.otus_csv),
            },
            "otu_counts_npz": {
                "path": relpath(paths.otu_counts_npz, paths.project_root),
                "sha256": sha256_file(paths.otu_counts_npz),
            },
        },
    }
    paths.manifest_json.write_text(json.dumps(manifest, indent=2) + "\n")

    print(f"Wrote {paths.samples_csv}")
    print(f"Wrote {paths.otus_csv}")
    print(f"Wrote {paths.otu_counts_npz}")
    print(f"Wrote {paths.manifest_json}")
    print("✅ 16S dataset preparation complete.")


if __name__ == "__main__":
    main()
