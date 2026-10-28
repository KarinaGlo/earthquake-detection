"""
tests/test_data_loader.py
=========================
Unit tests untuk data_loader.py mencakup 12 skenario pengujian wajib:
1. Dataset belum tersedia (NOT_FOUND).
2. Metadata tersedia tetapi waveform tidak ada (METADATA_ONLY).
3. Dataset lokal valid (READY).
4. Metadata rusak atau kolom label tidak ditemukan (INVALID / ValueError).
5. Repository utama gagal dan backup berhasil.
6. Kedua repository gagal.
7. Dataset alternatif memiliki label berbeda.
8. Jumlah data lebih sedikit daripada target subset.
9. Nilai SNR hilang atau tidak valid.
10. Split train/validation/test tidak kosong dan tidak mengalami kebocoran grup.
11. Mode synthetic berhasil menghasilkan file dengan format yang diharapkan (is_synthetic=True).
12. Mode inspect tidak memulai unduhan dataset besar.
"""

import sys
import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
import pandas as pd
import h5py
import requests

# Pastikan import src dapat diakses
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data_loader import (
    inspect_local_stead,
    check_remote_repository_health,
    create_balanced_subset,
    split_by_network,
    generate_synthetic_dataset,
    phase_inspect,
    PRIMARY_REMOTE_ROOT,
    BACKUP_REMOTE_ROOT
)


@pytest.fixture
def temp_dir():
    """Fixture untuk direktori temporer terisolasi."""
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


# ─────────────────────────────────────────────────────────────
# Test 1: Dataset belum tersedia
# ─────────────────────────────────────────────────────────────
def test_01_dataset_not_available(temp_dir):
    non_existent = temp_dir / "non_existent_dir"
    res = inspect_local_stead(str(non_existent))
    assert res["status"] == "NOT_FOUND"
    assert res["exists"] is False
    assert res["metadata_file"] is None
    assert res["waveform_file"] is None


# ─────────────────────────────────────────────────────────────
# Test 2: Metadata tersedia tetapi waveform tidak ada
# ─────────────────────────────────────────────────────────────
def test_02_metadata_available_waveforms_missing(temp_dir):
    meta_path = temp_dir / "merged.csv"
    df = pd.DataFrame({
        "trace_name": ["trace_001", "trace_002"],
        "trace_category": ["earthquake_local", "noise"],
        "network_code": ["IU", "II"]
    })
    df.to_csv(meta_path, index=False)

    res = inspect_local_stead(str(temp_dir))
    assert res["status"] == "METADATA_ONLY"
    assert res["metadata_file"] == str(meta_path)
    assert res["waveform_file"] is None
    assert res["n_records"] == 2
    assert any("waveform HDF5 belum ditemukan" in note for note in res["notes"])


# ─────────────────────────────────────────────────────────────
# Test 3: Dataset lokal valid (READY)
# ─────────────────────────────────────────────────────────────
def test_03_valid_local_dataset(temp_dir):
    meta_path = temp_dir / "merged.csv"
    df = pd.DataFrame({
        "trace_name": ["trace_001", "trace_002"],
        "trace_category": ["earthquake_local", "noise"],
        "network_code": ["IU", "II"]
    })
    df.to_csv(meta_path, index=False)

    wf_path = temp_dir / "merged.hdf5"
    with h5py.File(wf_path, "w") as f:
        g = f.create_group("data")
        # STEAD format: 6000 samples, 3 channels
        g.create_dataset("trace_001", data=np.zeros((6000, 3), dtype=np.float32))
        g.create_dataset("trace_002", data=np.zeros((6000, 3), dtype=np.float32))

    res = inspect_local_stead(str(temp_dir))
    assert res["status"] == "READY"
    assert res["metadata_file"] == str(meta_path)
    assert res["waveform_file"] == str(wf_path)
    assert res["waveform_sample_shape"] == [6000, 3]


# ─────────────────────────────────────────────────────────────
# Test 4: Metadata rusak atau kolom label tidak ditemukan
# ─────────────────────────────────────────────────────────────
def test_04_corrupt_metadata_missing_label_col():
    bad_df = pd.DataFrame({
        "trace_name": ["t1", "t2"],
        "random_col": [123, 456]
    })
    with pytest.raises(ValueError, match="Kolom kategori label tidak ditemukan"):
        create_balanced_subset(bad_df, n_earthquake=1, n_noise=1)


# ─────────────────────────────────────────────────────────────
# Test 5: Repository utama gagal dan backup berhasil
# ─────────────────────────────────────────────────────────────
def test_05_primary_repo_fails_backup_succeeds():
    def mock_head(url, **kwargs):
        resp = MagicMock()
        if "desy.de" in url:
            resp.status_code = 503
            resp.headers = {}
        elif "gfz.de" in url:
            resp.status_code = 200
            resp.headers = {"content-length": "402560190"}
        return resp

    with patch("requests.head", side_effect=mock_head):
        res_primary = check_remote_repository_health(PRIMARY_REMOTE_ROOT)
        res_backup = check_remote_repository_health(BACKUP_REMOTE_ROOT)

        assert res_primary["accessible"] is False
        assert res_backup["accessible"] is True
        assert res_backup["files"]["metadata.csv"]["status_code"] == 200


# ─────────────────────────────────────────────────────────────
# Test 6: Kedua repository gagal
# ─────────────────────────────────────────────────────────────
def test_06_both_repositories_fail():
    def mock_head_error(url, **kwargs):
        raise requests.exceptions.ConnectionError("Network unreachable")

    with patch("requests.head", side_effect=mock_head_error):
        res_primary = check_remote_repository_health(PRIMARY_REMOTE_ROOT)
        res_backup = check_remote_repository_health(BACKUP_REMOTE_ROOT)

        assert res_primary["accessible"] is False
        assert res_backup["accessible"] is False
        assert "Network unreachable" in res_primary["error"]
        assert "Network unreachable" in res_backup["error"]


# ─────────────────────────────────────────────────────────────
# Test 7: Dataset alternatif memiliki label berbeda
# ─────────────────────────────────────────────────────────────
def test_07_alternative_dataset_different_labels():
    alt_df = pd.DataFrame({
        "trace_name": [f"alt_{i}" for i in range(10)],
        "trace_type": ["earthquake"] * 5 + ["noise_ambient"] * 5,
        "network": ["IT"] * 10
    })
    subset = create_balanced_subset(
        alt_df,
        n_earthquake=3,
        n_noise=3,
        category_col="trace_type",
        eq_label="earthquake",
        noise_label="noise_ambient",
        source_name="LenDB"
    )
    assert len(subset) == 6
    assert list(subset["label"].value_counts().sort_index()) == [3, 3]
    assert subset["metadata_source"].iloc[0] == "LenDB"


# ─────────────────────────────────────────────────────────────
# Test 8: Jumlah data lebih sedikit daripada target subset
# ─────────────────────────────────────────────────────────────
def test_08_subset_size_exceeds_available_data():
    small_df = pd.DataFrame({
        "trace_name": [f"t_{i}" for i in range(10)],
        "trace_category": ["earthquake_local"] * 6 + ["noise"] * 4,
        "station_network_code": ["IU"] * 10
    })
    # Diminta 100 gempa dan 50 noise, tapi hanya ada 6 dan 4
    subset = create_balanced_subset(
        small_df,
        n_earthquake=100,
        n_noise=50,
        min_snr_db=0.0
    )
    assert len(subset) == 10
    assert (subset["label"] == 1).sum() == 6
    assert (subset["label"] == 0).sum() == 4


# ─────────────────────────────────────────────────────────────
# Test 9: Nilai SNR hilang atau tidak valid
# ─────────────────────────────────────────────────────────────
def test_09_snr_missing_or_invalid():
    df_with_bad_snr = pd.DataFrame({
        "trace_name": [f"t_{i}" for i in range(6)],
        "trace_category": ["earthquake_local"] * 4 + ["noise"] * 2,
        "snr_db": [np.nan, "INVALID_STR", -5.0, 25.0, "15.0", np.nan]
    })
    # SNR >= 10.0: index 3 (25.0) adalah satu-satunya gempa yang memenuhi
    subset = create_balanced_subset(
        df_with_bad_snr,
        n_earthquake=10,
        n_noise=10,
        min_snr_db=10.0
    )
    assert len(subset) == 3  # 1 gempa valid SNR + 2 noise
    eq_sub = subset[subset["label"] == 1]
    assert len(eq_sub) == 1
    assert eq_sub["trace_name"].iloc[0] == "t_3"


# ─────────────────────────────────────────────────────────────
# Test 10: Split train/val/test tidak kosong & tidak bocor grup
# ─────────────────────────────────────────────────────────────
def test_10_split_by_network_no_leakage_and_non_empty():
    df = pd.DataFrame({
        "trace_name": [f"t_{i}" for i in range(60)],
        "station_network_code": (["IU"] * 20) + (["II"] * 20) + (["GE"] * 10) + (["CI"] * 10),
        "label": ([1, 0] * 30)
    })
    split_df = split_by_network(df, train=0.70, val=0.15, test=0.15, seed=42)

    train_nets = set(split_df[split_df["split"] == "train"]["station_network_code"])
    val_nets = set(split_df[split_df["split"] == "val"]["station_network_code"])
    test_nets = set(split_df[split_df["split"] == "test"]["station_network_code"])

    # 1. Pastikan tidak ada split yang kosong
    assert len(train_nets) > 0, "Train split kosong!"
    assert len(val_nets) > 0, "Val split kosong!"
    assert len(test_nets) > 0, "Test split kosong!"

    # 2. Pastikan grup saling lepas (Zero group leakage)
    assert train_nets.isdisjoint(val_nets), "Leakage terdeteksi antara Train dan Val!"
    assert train_nets.isdisjoint(test_nets), "Leakage terdeteksi antara Train dan Test!"
    assert val_nets.isdisjoint(test_nets), "Leakage terdeteksi antara Val dan Test!"


# ─────────────────────────────────────────────────────────────
# Test 11: Mode synthetic menghasilkan file & is_synthetic=True
# ─────────────────────────────────────────────────────────────
def test_11_synthetic_mode_output_format(temp_dir):
    out_dir = generate_synthetic_dataset(
        n_earthquake=15,
        n_noise=5,
        n_samples=500,
        output_dir=str(temp_dir)
    )

    for split in ["train", "val", "test"]:
        npz_file = temp_dir / f"{split}.npz"
        assert npz_file.is_file(), f"File {split}.npz tidak ditemukan!"
        data = np.load(npz_file)
        wf = data["waveforms"]
        lb = data["labels"]
        assert wf.ndim == 3
        assert wf.shape[1:] == (3, 500)
        assert len(wf) == len(lb)
        assert len(lb) > 0

    meta_file = temp_dir / "metadata.csv"
    assert meta_file.is_file()
    meta_df = pd.read_csv(meta_file)
    assert "is_synthetic" in meta_df.columns
    assert bool(meta_df["is_synthetic"].all()) is True


# ─────────────────────────────────────────────────────────────
# Test 12: Mode inspect tidak memulai unduhan dataset besar
# ─────────────────────────────────────────────────────────────
def test_12_inspect_mode_does_not_download_large_dataset(temp_dir):
    mock_config = {
        "dataset": {
            "name": "STEAD",
            "source": "seisbench",
            "local_path": str(temp_dir / "raw_stead"),
            "use_backup_repository": False
        },
        "preprocessing": {
            "processed_dir": str(temp_dir / "processed")
        }
    }

    head_call_count = 0
    def mock_head(url, **kwargs):
        nonlocal head_call_count
        head_call_count += 1
        resp = MagicMock()
        resp.status_code = 200
        resp.headers = {"content-length": "1000"}
        return resp

    get_mock = MagicMock()

    with patch("requests.head", side_effect=mock_head), patch("requests.get", get_mock):
        res = phase_inspect(mock_config)

        # Harus memanggil HTTP HEAD untuk cek status remote
        assert head_call_count >= 2
        # JANGAN PERNAH memanggil requests.get (download payload) saat inspeksi!
        assert get_mock.call_count == 0
        assert res["local_info"]["status"] == "NOT_FOUND"
