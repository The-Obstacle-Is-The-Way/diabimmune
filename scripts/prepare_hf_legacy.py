from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import h5py
import numpy as np
import pandas as pd
import pyreadr
from huggingface_hub import snapshot_download

HF_REPO_ID = "hugging-science/AI4FA-Diabimmune"
HF_REVISION = "7761eea93dad5712a03452786b43031dc9b04233"
HF_ALLOW_PATTERNS = ["processed/microbiome_embeddings/Month_*/microbiome_embeddings.h5"]

SRA_RUNINFO_URL = "https://www.ncbi.nlm.nih.gov/Traces/sra-db-be/runinfo?term=PRJNA290380"


@dataclass(frozen=True)
class Paths:
    project_root: Path
    raw_dir: Path
    processed_dir: Path
    rdata_path: Path
    sra_runinfo_path: Path
    mapping_csv: Path
    unified_samples_csv: Path
    embeddings_h5: Path
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


def coerce_bool(value: Any) -> bool:
    if value is None:
        return False
    if pd.isna(value):
        return False
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


def age_bin_from_month(month: int) -> str:
    if month <= 3:
        return "0-3"
    if month <= 6:
        return "4-6"
    if month <= 12:
        return "7-12"
    if month <= 24:
        return "13-24"
    return "25+"


def load_rdata_metadata(rdata_path: Path) -> pd.DataFrame:
    r = pyreadr.read_r(str(rdata_path))
    if "metadata" not in r:
        raise KeyError(f"Expected key 'metadata' in {rdata_path}, found: {list(r.keys())}")
    metadata = r["metadata"]
    if not isinstance(metadata, pd.DataFrame):
        raise TypeError(f"Expected 'metadata' to be a pandas DataFrame, got {type(metadata)}")
    return metadata


def load_sra_runinfo_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    required = {"Sample", "LibraryName"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Missing columns in {path}: {sorted(missing)}")
    return df


def download_sra_runinfo(path: Path) -> None:
    import ssl
    import urllib.request

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    with urllib.request.urlopen(SRA_RUNINFO_URL, context=ctx) as resp:  # noqa: S310
        text = resp.read().decode("utf-8")
    path.write_text(text)


def build_subject_labels(metadata: pd.DataFrame) -> pd.Series:
    allergy_cols = [c for c in metadata.columns if c.startswith("allergy_")]
    required = {"subjectID", "totalige_high"}
    missing = required - set(metadata.columns)
    if missing:
        raise KeyError(f"Missing required columns in RData metadata: {sorted(missing)}")
    if not allergy_cols:
        raise KeyError("No columns starting with 'allergy_' found in RData metadata.")

    cols = ["subjectID", *allergy_cols, "totalige_high"]
    df = metadata[cols].copy()

    for c in allergy_cols + ["totalige_high"]:
        df[c] = df[c].map(coerce_bool)

    by_subject = df.groupby("subjectID", sort=False)[allergy_cols + ["totalige_high"]].any()
    label = (by_subject[allergy_cols].any(axis=1) | by_subject["totalige_high"]).astype(int)
    label.name = "label"
    return label


def build_mapping_and_samples(
    *,
    metadata: pd.DataFrame,
    sra_runinfo: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_meta = {
        "gid_wgs",
        "subjectID",
        "country",
        "gender",
        "collection_month",
        "age_at_collection",
    }
    missing_meta = required_meta - set(metadata.columns)
    if missing_meta:
        raise KeyError(f"Missing required columns in RData metadata: {sorted(missing_meta)}")

    # Mapping: SRS (Sample) -> gid_wgs (LibraryName) -> subjectID (RData)
    mapping = sra_runinfo[["Sample", "LibraryName"]].rename(
        columns={"Sample": "srs_id", "LibraryName": "gid_wgs"},
    )

    meta_wgs = metadata.loc[metadata["gid_wgs"].notna()].copy()
    meta_wgs["gid_wgs"] = meta_wgs["gid_wgs"].astype(str)

    meta_cols = [
        "gid_wgs",
        "subjectID",
        "country",
        "gender",
        "collection_month",
        "age_at_collection",
    ]
    meta_wgs = meta_wgs[meta_cols].copy()

    merged = mapping.merge(meta_wgs, on="gid_wgs", how="left", validate="one_to_one")
    if merged["subjectID"].isna().any():
        missing = merged.loc[merged["subjectID"].isna(), "gid_wgs"].tolist()
        raise ValueError(f"Missing gid_wgs in RData metadata (n={len(missing)}): {missing[:10]}")

    merged = merged.rename(columns={"subjectID": "subject_id", "age_at_collection": "age_days"})
    merged["collection_month"] = merged["collection_month"].astype(int)
    merged["age_days"] = merged["age_days"].astype(int)

    # Add subject-level label derived from RData ground truth
    label_by_subject = build_subject_labels(metadata)
    merged["label"] = merged["subject_id"].map(label_by_subject).astype(int)

    merged["age_bin"] = merged["collection_month"].map(age_bin_from_month)

    mapping_out = merged[["srs_id", "gid_wgs", "subject_id", "country", "gender"]].copy()
    samples_out = merged[
        [
            "srs_id",
            "gid_wgs",
            "subject_id",
            "collection_month",
            "age_days",
            "country",
            "age_bin",
            "label",
        ]
    ].copy()

    # Stable ordering: subject, then month
    samples_out = samples_out.sort_values(
        ["subject_id", "collection_month", "srs_id"], kind="mergesort"
    )
    mapping_out = mapping_out.sort_values(["srs_id"], kind="mergesort")

    return mapping_out, samples_out


def export_canonical_embeddings(
    *,
    samples: pd.DataFrame,
    hf_embeddings_root: Path,
    out_h5_path: Path,
) -> None:
    out_h5_path.parent.mkdir(parents=True, exist_ok=True)

    expected_srs = samples["srs_id"].tolist()
    expected_set = set(expected_srs)

    with h5py.File(out_h5_path, "w") as out_h5:
        for month, g in samples.groupby("collection_month", sort=True):
            month_dir = hf_embeddings_root / f"Month_{int(cast(int, month))}"
            src_h5 = month_dir / "microbiome_embeddings.h5"
            if not src_h5.exists():
                raise FileNotFoundError(f"Missing HuggingFace embeddings file: {src_h5}")

            with h5py.File(src_h5, "r") as in_h5:
                for srs_id in g["srs_id"]:
                    if srs_id not in in_h5:
                        raise KeyError(f"Missing embedding for {srs_id} in {src_h5}")
                    vec = np.asarray(in_h5[srs_id], dtype=np.float32)
                    if vec.shape != (100,):
                        raise ValueError(f"Unexpected embedding shape for {srs_id}: {vec.shape}")
                    out_h5.create_dataset(srs_id, data=vec, dtype=np.float32)

    with h5py.File(out_h5_path, "r") as f:
        keys = set(f.keys())
    missing = expected_set - keys
    extra = keys - expected_set
    if missing or extra:
        raise ValueError(
            f"Canonical embedding export mismatch: missing={len(missing)} extra={len(extra)}"
        )


def build_manifest(
    *,
    paths: Paths,
    rdata_metadata: pd.DataFrame,
    samples: pd.DataFrame,
) -> dict[str, Any]:
    sample_label_counts = samples["label"].value_counts().to_dict()
    subject_label_counts = (
        samples.drop_duplicates("subject_id")
        .set_index("subject_id")["label"]
        .value_counts()
        .to_dict()
    )

    payload: dict[str, Any] = {
        "created_at": datetime.now(UTC).isoformat(),
        "sources": {
            "rdata": {
                "path": relpath(paths.rdata_path, paths.project_root),
                "shape": [int(rdata_metadata.shape[0]), int(rdata_metadata.shape[1])],
                "subjects": int(rdata_metadata["subjectID"].nunique()),
            },
            "ncbi_sra": {
                "bioproject": "PRJNA290380",
                "runinfo_url": SRA_RUNINFO_URL,
                "runinfo_path": relpath(paths.sra_runinfo_path, paths.project_root),
                "rows": int(len(pd.read_csv(paths.sra_runinfo_path))),
            },
            "huggingface": {
                "repo_id": HF_REPO_ID,
                "revision": HF_REVISION,
                "allow_patterns": HF_ALLOW_PATTERNS,
            },
        },
        "label_definition": "label=1 if any(allergy_*) OR totalige_high (computed from RData; matches HF label)",
        "embedding_export": {
            "output_h5": relpath(paths.embeddings_h5, paths.project_root),
            "rule": "for each sample, use the vector from Month_{collection_month} (collection_month from RData)",
            "dims": 100,
        },
        "dataset": {
            "n_samples": int(len(samples)),
            "n_subjects": int(samples["subject_id"].nunique()),
            "sample_label_counts": {str(k): int(v) for k, v in sample_label_counts.items()},
            "subject_label_counts": {str(k): int(v) for k, v in subject_label_counts.items()},
        },
        "artifacts": {
            "unified_samples_csv": {
                "path": relpath(paths.unified_samples_csv, paths.project_root),
                "sha256": sha256_file(paths.unified_samples_csv)
                if paths.unified_samples_csv.exists()
                else None,
            },
            "srs_to_subject_mapping_csv": {
                "path": relpath(paths.mapping_csv, paths.project_root),
                "sha256": sha256_file(paths.mapping_csv) if paths.mapping_csv.exists() else None,
            },
            "microbiome_embeddings_h5": {
                "path": relpath(paths.embeddings_h5, paths.project_root),
                "sha256": sha256_file(paths.embeddings_h5)
                if paths.embeddings_h5.exists()
                else None,
            },
        },
    }
    return payload


def resolve_paths(project_root: Path) -> Paths:
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed" / "hf_legacy"
    return Paths(
        project_root=project_root,
        raw_dir=raw_dir,
        processed_dir=processed_dir,
        rdata_path=raw_dir / "DIABIMMUNE_Karelia_metadata.RData",
        sra_runinfo_path=raw_dir / "sra_runinfo.csv",
        mapping_csv=processed_dir / "srs_to_subject_mapping.csv",
        unified_samples_csv=processed_dir / "unified_samples.csv",
        embeddings_h5=processed_dir / "microbiome_embeddings_100d.h5",
        manifest_json=processed_dir / "dataset_manifest.json",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare DIABIMMUNE processed data artifacts.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild outputs even if they already exist.",
    )
    parser.add_argument(
        "--download-runinfo",
        action="store_true",
        help="Download NCBI SRA runinfo if missing (or if --force).",
    )
    parser.add_argument(
        "--keep-hf-snapshot",
        action="store_true",
        help="Keep a local HF snapshot under data/raw/huggingface_snapshot/ (otherwise uses a temp dir).",
    )

    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    paths = resolve_paths(project_root)
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    paths.processed_dir.mkdir(parents=True, exist_ok=True)

    if not paths.rdata_path.exists():
        raise FileNotFoundError(
            f"Missing {paths.rdata_path}. Place DIABIMMUNE_Karelia_metadata.RData there before running."
        )

    if args.download_runinfo or (not paths.sra_runinfo_path.exists()):
        print(f"Downloading SRA runinfo → {paths.sra_runinfo_path}")
        download_sra_runinfo(paths.sra_runinfo_path)

    if not paths.sra_runinfo_path.exists():
        raise FileNotFoundError(f"Missing {paths.sra_runinfo_path}.")

    # Rebuild mapping + sample table
    if args.force or (not paths.mapping_csv.exists()) or (not paths.unified_samples_csv.exists()):
        print("Loading RData metadata...")
        rdata_metadata = load_rdata_metadata(paths.rdata_path)

        print("Loading SRA runinfo...")
        sra_runinfo = load_sra_runinfo_csv(paths.sra_runinfo_path)

        print("Building mapping + unified sample table...")
        mapping_df, samples_df = build_mapping_and_samples(
            metadata=rdata_metadata, sra_runinfo=sra_runinfo
        )

        mapping_df.to_csv(paths.mapping_csv, index=False)
        samples_df.to_csv(paths.unified_samples_csv, index=False)
        print(f"Wrote {paths.mapping_csv}")
        print(f"Wrote {paths.unified_samples_csv}")
    else:
        rdata_metadata = load_rdata_metadata(paths.rdata_path)
        samples_df = pd.read_csv(paths.unified_samples_csv)

    # Export canonical embeddings
    if args.force or (not paths.embeddings_h5.exists()):
        print("Downloading HuggingFace microbiome embeddings (Month_* subsets)...")

        if args.keep_hf_snapshot:
            snapshot_dir = paths.raw_dir / "huggingface_snapshot"
            snapshot_dir.mkdir(parents=True, exist_ok=True)
            local_dir = snapshot_dir / f"AI4FA-Diabimmune@{HF_REVISION}"
            local_dir.mkdir(parents=True, exist_ok=True)
            snapshot_path = Path(
                snapshot_download(
                    repo_id=HF_REPO_ID,
                    repo_type="dataset",
                    revision=HF_REVISION,
                    local_dir=local_dir,
                    allow_patterns=HF_ALLOW_PATTERNS,
                )
            )
        else:
            tmp = tempfile.TemporaryDirectory()
            snapshot_path = Path(
                snapshot_download(
                    repo_id=HF_REPO_ID,
                    repo_type="dataset",
                    revision=HF_REVISION,
                    local_dir=Path(tmp.name),
                    allow_patterns=HF_ALLOW_PATTERNS,
                )
            )

        hf_embeddings_root = snapshot_path / "processed" / "microbiome_embeddings"
        print("Exporting canonical embeddings...")
        export_canonical_embeddings(
            samples=samples_df,
            hf_embeddings_root=hf_embeddings_root,
            out_h5_path=paths.embeddings_h5,
        )
        print(f"Wrote {paths.embeddings_h5}")

        if not args.keep_hf_snapshot:
            tmp.cleanup()

    # Manifest
    print("Writing manifest...")
    manifest = build_manifest(paths=paths, rdata_metadata=rdata_metadata, samples=samples_df)
    paths.manifest_json.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {paths.manifest_json}")

    # Final validations
    samples_df = pd.read_csv(paths.unified_samples_csv)
    assert len(samples_df) == 785, f"Expected 785 samples, got {len(samples_df)}"
    assert samples_df["subject_id"].nunique() == 212, (
        "Expected 212 subjects in embeddings-backed set"
    )
    assert samples_df["srs_id"].is_unique
    with h5py.File(paths.embeddings_h5, "r") as f:
        assert len(f.keys()) == 785
        assert set(f.keys()) == set(samples_df["srs_id"])

    print("✅ Data preparation complete.")


if __name__ == "__main__":
    main()
