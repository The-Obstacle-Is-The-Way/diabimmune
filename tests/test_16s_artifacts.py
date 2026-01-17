from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def test_16s_processed_artifacts_are_aligned() -> None:
    project_root = Path(__file__).resolve().parents[1]
    processed_dir = project_root / "data" / "processed" / "16s"

    samples_csv = processed_dir / "samples_food_allergy.csv"
    otus_csv = processed_dir / "otus_greengenes_ids.csv"
    counts_npz = processed_dir / "otu_counts.npz"
    manifest_json = processed_dir / "dataset_manifest.json"

    assert samples_csv.exists()
    assert otus_csv.exists()
    assert counts_npz.exists()
    assert manifest_json.exists()

    samples = pd.read_csv(samples_csv)
    otus = pd.read_csv(otus_csv)
    npz = np.load(counts_npz, allow_pickle=False)

    counts = npz["counts"]
    sample_id = npz["sample_id"]
    otu_id = npz["otu_id"]

    assert counts.shape == (len(samples), 2005)
    assert counts.dtype == np.int32
    assert sample_id.shape == (len(samples),)
    assert otu_id.shape == (2005,)

    assert samples["sample_id"].is_unique
    assert set(samples["label_food"].unique()) <= {0, 1}
    assert (samples.groupby("subject_id")["label_food"].nunique() <= 1).all()

    assert len(otus) == 2005
    assert otus["otu_id"].is_unique

    assert np.array_equal(samples["sample_id"].astype(str).to_numpy(), sample_id)
    assert np.array_equal(otus["otu_id"].to_numpy(dtype=np.int32), otu_id)

    manifest = json.loads(manifest_json.read_text())
    assert manifest["dataset"]["n_samples_labeled"] == int(len(samples))
    assert manifest["dataset"]["n_otus"] == 2005
