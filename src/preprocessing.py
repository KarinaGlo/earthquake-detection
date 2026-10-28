"""
preprocessing.py
================
Pipeline preprocessing waveform seismik untuk deteksi gempa bumi.

Alur pemrosesan:
1. Baca metadata subset dari CSV
2. Untuk setiap rekaman: unduh waveform via SeisBench
3. Validasi: cek missing values, sampling rate, panjang sinyal
4. Normalisasi sinyal
5. Simpan ke format NumPy (.npy) terstruktur

Penggunaan:
    python src/preprocessing.py --config configs/config.yaml
    python src/preprocessing.py --config configs/config.yaml --phase smoke_test --n_samples 100

Referensi:
    Mousavi et al. (2019). STEAD. IEEE Access.
"""

import os
import sys
import argparse
import logging
import yaml
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("preprocessing")


def load_config(config_path: str) -> dict:
    """Memuat konfigurasi YAML."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def validate_waveform(
    waveform: np.ndarray,
    expected_sampling_rate: int,
    expected_n_samples: int,
    expected_channels: int,
    trace_name: str
) -> Tuple[bool, str]:
    """
    Memvalidasi waveform yang diterima dari SeisBench.

    Checks:
    - Shape array sesuai ekspektasi
    - Tidak ada NaN atau Inf
    - Sinyal tidak semua nol (dead channel)

    Args:
        waveform: Array numpy shape (channels, samples)
        expected_sampling_rate: Sampling rate yang diharapkan (Hz)
        expected_n_samples: Jumlah sampel yang diharapkan
        expected_channels: Jumlah channel yang diharapkan
        trace_name: Identitas rekaman untuk logging

    Returns:
        (is_valid, reason): Tuple boolean validitas dan alasan penolakan
    """
    if waveform is None:
        return False, "Waveform adalah None"

    if not isinstance(waveform, np.ndarray):
        return False, f"Tipe data tidak valid: {type(waveform)}"

    # Cek dimensi
    if waveform.ndim != 2:
        return False, f"Dimensi tidak valid: {waveform.ndim}D (diharapkan 2D)"

    actual_channels, actual_samples = waveform.shape

    # Toleransi: channel bisa 1, 2, atau 3
    if actual_channels < 1 or actual_channels > 3:
        return False, f"Jumlah channel tidak valid: {actual_channels}"

    # Cek panjang sampel — toleransi ±10%
    sample_tolerance = int(expected_n_samples * 0.1)
    if not (expected_n_samples - sample_tolerance <= actual_samples <= expected_n_samples + sample_tolerance):
        return False, f"Panjang sampel tidak sesuai: {actual_samples} (diharapkan ~{expected_n_samples})"

    # Cek NaN/Inf
    if np.any(~np.isfinite(waveform)):
        n_bad = np.sum(~np.isfinite(waveform))
        return False, f"Mengandung {n_bad} nilai NaN/Inf"

    # Cek dead channel (semua nol) — hanya peringatan, tidak reject
    for ch_idx in range(actual_channels):
        if np.all(waveform[ch_idx] == 0):
            logger.warning(f"[{trace_name}] Channel {ch_idx} berisi semua nol (dead channel)")

    return True, "OK"


def normalize_waveform(
    waveform: np.ndarray,
    method: str = "peak"
) -> np.ndarray:
    """
    Normalisasi waveform seismik.

    Normalisasi dilakukan per-trace (bukan per-dataset) untuk menghindari
    data leakage antara training dan testing.

    Metode:
    - 'peak': Bagi dengan nilai absolut maksimum per channel
    - 'std': Normalisasi z-score per channel
    - 'global_peak': Bagi dengan nilai absolut maksimum global (semua channel)
    - 'none': Tidak ada normalisasi

    Args:
        waveform: Array numpy shape (channels, samples)
        method: Metode normalisasi

    Returns:
        Waveform ternormalisasi, shape sama dengan input
    """
    waveform = waveform.astype(np.float32)

    if method == "none":
        return waveform

    elif method == "peak":
        # Normalisasi per channel
        for ch_idx in range(waveform.shape[0]):
            ch_max = np.max(np.abs(waveform[ch_idx]))
            if ch_max > 1e-10:  # Hindari pembagian nol
                waveform[ch_idx] /= ch_max

    elif method == "std":
        # Z-score per channel
        for ch_idx in range(waveform.shape[0]):
            ch_mean = np.mean(waveform[ch_idx])
            ch_std = np.std(waveform[ch_idx])
            if ch_std > 1e-10:
                waveform[ch_idx] = (waveform[ch_idx] - ch_mean) / ch_std

    elif method == "global_peak":
        # Normalisasi dengan peak global semua channel
        global_max = np.max(np.abs(waveform))
        if global_max > 1e-10:
            waveform /= global_max

    else:
        raise ValueError(f"Metode normalisasi tidak dikenal: {method}")

    return waveform


def resample_if_needed(
    waveform: np.ndarray,
    source_sr: int,
    target_sr: int
) -> np.ndarray:
    """
    Resample waveform jika sampling rate tidak sesuai target.

    Args:
        waveform: Array shape (channels, samples)
        source_sr: Sampling rate sumber (Hz)
        target_sr: Sampling rate target (Hz)

    Returns:
        Waveform dengan sampling rate target
    """
    if source_sr == target_sr:
        return waveform

    from scipy.signal import resample_poly
    from math import gcd

    logger.debug(f"Resampling dari {source_sr} Hz ke {target_sr} Hz")

    g = gcd(target_sr, source_sr)
    up = target_sr // g
    down = source_sr // g

    resampled = np.zeros((waveform.shape[0], int(waveform.shape[1] * target_sr / source_sr)))
    for ch in range(waveform.shape[0]):
        resampled[ch] = resample_poly(waveform[ch], up, down)

    return resampled


def pad_or_crop(waveform: np.ndarray, target_samples: int) -> np.ndarray:
    """
    Memotong atau menambahkan padding pada waveform agar sesuai panjang target.

    Args:
        waveform: Array shape (channels, samples)
        target_samples: Jumlah sampel yang diinginkan

    Returns:
        Waveform dengan panjang target_samples
    """
    current_samples = waveform.shape[1]

    if current_samples == target_samples:
        return waveform

    elif current_samples > target_samples:
        # Crop dari tengah untuk menjaga fase arrival jika ada
        start = (current_samples - target_samples) // 2
        return waveform[:, start:start + target_samples]

    else:
        # Padding dengan nol di akhir
        pad_width = target_samples - current_samples
        return np.pad(waveform, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)


def process_single_trace(
    data,  # SeisBench dataset
    idx: int,
    config: dict,
    norm_method: str = "peak"
) -> Optional[Tuple[np.ndarray, dict]]:
    """
    Memproses satu trace waveform dari SeisBench.

    Args:
        data: SeisBench STEAD dataset object
        idx: Index rekaman
        config: Konfigurasi proyek
        norm_method: Metode normalisasi

    Returns:
        (waveform, metadata) atau None jika invalid
    """
    waveform_config = config["dataset"]["waveform"]
    target_sr = waveform_config["sampling_rate"]
    target_samples = waveform_config["n_samples"]
    target_channels = waveform_config["n_channels"]

    try:
        # Ambil waveform dan metadata dari SeisBench
        waveform, metadata_row = data[idx]

        if waveform is None:
            logger.debug(f"Index {idx}: Waveform None")
            return None

        # Konversi ke numpy jika belum
        if not isinstance(waveform, np.ndarray):
            waveform = np.array(waveform, dtype=np.float32)

        # Pastikan shape (channels, samples)
        if waveform.ndim == 1:
            waveform = waveform[np.newaxis, :]  # (1, samples)

        trace_name = metadata_row.get("trace_name", f"trace_{idx}")

        # Validasi
        is_valid, reason = validate_waveform(
            waveform, target_sr, target_samples, target_channels, trace_name
        )

        if not is_valid:
            logger.debug(f"[{trace_name}] Ditolak: {reason}")
            return None

        # Pad atau crop ke ukuran standar
        waveform = pad_or_crop(waveform, target_samples)

        # Normalisasi
        waveform = normalize_waveform(waveform, method=norm_method)

        return waveform, dict(metadata_row)

    except Exception as e:
        logger.debug(f"Error pada index {idx}: {e}")
        return None


def run_preprocessing(config: dict, n_samples: Optional[int] = None) -> None:
    """
    Menjalankan pipeline preprocessing untuk seluruh subset.

    Pipeline:
    1. Baca metadata subset dari CSV
    2. Akses SeisBench STEAD
    3. Proses setiap trace (validate, normalize, pad/crop)
    4. Simpan ke file NPZ (compressed NumPy)

    Args:
        config: Konfigurasi proyek
        n_samples: Jika diisi, hanya proses n_samples rekaman pertama (untuk smoke test)
    """
    import seisbench.data as sbd

    # Path konfigurasi
    metadata_path = config["preprocessing"]["metadata_file"]
    processed_dir = Path(config["preprocessing"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    norm_method = config["preprocessing"]["normalization"]["method"]

    # Baca metadata
    if not Path(metadata_path).exists():
        logger.error(f"File metadata tidak ditemukan: {metadata_path}")
        logger.error("Jalankan terlebih dahulu: python src/data_loader.py --phase download")
        sys.exit(1)

    metadata = pd.read_csv(metadata_path)
    logger.info(f"Metadata dimuat: {len(metadata):,} rekaman")

    if n_samples is not None:
        metadata = metadata.head(n_samples)
        logger.info(f"Mode smoke test: hanya memproses {n_samples} rekaman pertama")

    # Akses STEAD
    dataset_config = config["dataset"]
    if dataset_config["source"] == "local":
        data = sbd.STEAD(basepath=dataset_config["local_path"])
    else:
        data = sbd.STEAD()

    logger.info("STEAD dataset dimuat.")

    # Proses per split
    splits = metadata["split"].unique() if "split" in metadata.columns else ["all"]
    waveform_config = config["dataset"]["waveform"]

    preprocessing_log = {
        "config": {
            "normalization": norm_method,
            "target_sr": waveform_config["sampling_rate"],
            "target_samples": waveform_config["n_samples"],
            "n_channels": waveform_config["n_channels"]
        },
        "splits": {}
    }

    # Buat mapping trace_name -> dataset index
    logger.info("Membangun mapping trace_name ke index dataset...")
    name_to_idx = {}
    if "trace_name" in data.metadata.columns:
        for i, row in enumerate(data.metadata.itertuples()):
            name_to_idx[row.trace_name] = i
    else:
        logger.error("Kolom trace_name tidak ditemukan dalam metadata STEAD")
        sys.exit(1)

    for split in splits:
        if split == "all":
            split_metadata = metadata
            split_name = "all"
        else:
            split_metadata = metadata[metadata["split"] == split]
            split_name = split

        logger.info(f"\nMemproses split '{split_name}': {len(split_metadata):,} rekaman")

        waveforms = []
        labels = []
        valid_metadata = []
        n_skipped = 0

        for _, row in tqdm(split_metadata.iterrows(), total=len(split_metadata), desc=f"Split {split_name}"):
            trace_name = row.get("trace_name")
            if trace_name not in name_to_idx:
                n_skipped += 1
                continue

            idx = name_to_idx[trace_name]
            result = process_single_trace(data, idx, config, norm_method)

            if result is None:
                n_skipped += 1
                continue

            waveform, meta = result
            waveforms.append(waveform)
            labels.append(int(row.get("label", 0)))
            valid_metadata.append(meta)

        if not waveforms:
            logger.error(f"Tidak ada rekaman valid untuk split '{split_name}'")
            continue

        # Konversi ke array numpy
        waveforms_array = np.array(waveforms, dtype=np.float32)
        labels_array = np.array(labels, dtype=np.int64)

        # Simpan
        output_path = processed_dir / f"{split_name}.npz"
        np.savez_compressed(
            output_path,
            waveforms=waveforms_array,
            labels=labels_array
        )

        # Simpan metadata valid
        valid_meta_df = pd.DataFrame(valid_metadata)
        valid_meta_path = processed_dir / f"{split_name}_metadata.csv"
        valid_meta_df.to_csv(valid_meta_path, index=False)

        n_valid = len(waveforms)
        n_total = len(split_metadata)
        acceptance_rate = n_valid / n_total * 100 if n_total > 0 else 0

        preprocessing_log["splits"][split_name] = {
            "total": n_total,
            "valid": n_valid,
            "skipped": n_skipped,
            "acceptance_rate_pct": round(acceptance_rate, 2),
            "waveform_shape": list(waveforms_array.shape),
            "label_distribution": {
                "earthquake (1)": int(np.sum(labels_array == 1)),
                "noise (0)": int(np.sum(labels_array == 0))
            },
            "output_file": str(output_path),
        }

        logger.info(f"Split '{split_name}':")
        logger.info(f"  Valid: {n_valid:,} / {n_total:,} ({acceptance_rate:.1f}%)")
        logger.info(f"  Skipped: {n_skipped:,}")
        logger.info(f"  Waveform shape: {waveforms_array.shape}")
        logger.info(f"  Label distribution: {preprocessing_log['splits'][split_name]['label_distribution']}")
        logger.info(f"  Tersimpan: {output_path}")

    # Simpan log preprocessing
    log_path = processed_dir / "preprocessing_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(preprocessing_log, f, indent=2, ensure_ascii=False)

    logger.info(f"\nPreprocessing log tersimpan: {log_path}")
    logger.info("Preprocessing selesai.")
    logger.info("Selanjutnya: python src/train.py --config configs/config.yaml")


def main():
    parser = argparse.ArgumentParser(description="Preprocessing pipeline untuk waveform seismik")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--phase",
        type=str,
        choices=["full", "smoke_test"],
        default="full",
        help="'full' untuk semua data, 'smoke_test' untuk 100 sampel pertama"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Jumlah sampel yang diproses (override phase)"
    )
    args = parser.parse_args()

    # Pindah ke root proyek
    script_dir = Path(__file__).parent
    os.chdir(script_dir.parent)

    config = load_config(args.config)

    n_samples = args.n_samples
    if args.phase == "smoke_test" and n_samples is None:
        n_samples = 100

    run_preprocessing(config, n_samples=n_samples)


if __name__ == "__main__":
    main()
