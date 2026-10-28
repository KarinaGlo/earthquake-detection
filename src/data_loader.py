"""
data_loader.py (v3 — Enterprise Seismic Pipeline)
=================================================
Modul pengelolaan data untuk proyek Deteksi Gempa Bumi Berbasis Deep Learning.

Fitur & Alur Kerja:
1. Inspeksi Aman (Zero-Byte Download):
   - Memeriksa konfigurasi proyek (config.yaml).
   - Memeriksa direktori data lokal (STEAD asli: merged.csv + merged.hdf5).
   - Memeriksa cache SeisBench lokal (~/.seisbench/datasets/stead).
   - Melakukan health-check remote repository SeisBench (Primary & Backup)
     via HTTP HEAD tanpa mengunduh payload puluhan gigabyte.
   - Memeriksa status data olahan di data/processed/.

2. Penanganan Bug Windows SeisBench:
   - Patch otomatis untuk masalah pemisah path (backslash '\\') pada Windows
     yang menyebabkan HTTP 404 saat mengakses remote repository.

3. Validasi Dataset Lokal & Format:
   - Mendukung format asli STEAD (merged.csv + merged.hdf5) dan format SeisBench.
   - Validasi integritas HDF5, bentuk sinyal (channels x samples), dan mapping trace_name.

4. Pembentukan Subset & Split Seismik Bebas Data Leakage:
   - create_balanced_subset(): Aturan label eksplisit, filter SNR toleran missing/NaN,
     serta graceful handling jika data tersedia lebih sedikit dari permintaan.
   - split_by_network(): Pembagian berbasis jaringan stasiun tanpa kebocoran grup
     (train, val, test mutually exclusive) dengan jaminan split tidak kosong.

5. Mode Sintetis Terkontrol:
   - Diberi penanda eksplisit is_synthetic=True untuk testing dan pengujian pipeline.

Penggunaan:
    python src/data_loader.py --config configs/config.yaml --phase inspect
    python src/data_loader.py --config configs/config.yaml --phase download
    python src/data_loader.py --config configs/config.yaml --phase synthetic --n 6000
"""

import os
import sys
import argparse
import logging
import yaml
import json
import time
import posixpath
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any

import numpy as np
import pandas as pd
import requests

# Pastikan output console di Windows aman dari charmap/UnicodeEncodeError
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("data_loader")

# Remote repository URLs resmi SeisBench
PRIMARY_REMOTE_ROOT = "https://hifis-storage.desy.de/Helmholtz/HelmholtzAI/SeisBench/"
BACKUP_REMOTE_ROOT = "https://seisbench.gfz.de/mirror/"


# ─────────────────────────────────────────────────────────────
# 1. Utilities & Patching
# ─────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Memuat file konfigurasi YAML."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"File konfigurasi tidak ditemukan: {config_path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_seisbench() -> Tuple[bool, Optional[str]]:
    """
    Memeriksa ketersediaan library SeisBench dan versinya.
    Returns:
        (is_installed, version_str)
    """
    try:
        import seisbench
        version = getattr(seisbench, "__version__", "unknown")
        logger.debug(f"SeisBench terinstal, versi: {version}")
        return True, version
    except ImportError:
        logger.error("SeisBench tidak terinstal. Jalankan: pip install seisbench")
        return False, None


def patch_seisbench_windows_url_bug():
    """
    Memperbaiki bug SeisBench di lingkungan Windows di mana os.path.join
    digunakan untuk menyusun URL HTTP remote repository. Pada Windows,
    os.path.join menghasilkan tanda '\\' (misal: 'datasets/stead\\metadata.csv')
    yang menyebabkan server web mengembalikan error HTTP 404.
    """
    try:
        import seisbench.data.base as sbdb
        orig_download = getattr(sbdb.WaveformBenchmarkDataset, "_download_preprocessed", None)
        if orig_download is not None and not getattr(orig_download, "_is_patched_for_windows", False):
            def safe_download_preprocessed(self, output_files: List[Path], chunk: str):
                self.path.mkdir(parents=True, exist_ok=True)
                remote_base = self._remote_path().replace("\\", "/")
                for file_name, output_file in zip(self._files, output_files):
                    file_name = file_name.replace("$CHUNK", chunk)
                    remote_file_path = posixpath.join(remote_base, file_name)
                    import seisbench
                    seisbench.util.download_http(
                        remote_file_path, output_file, desc=f"Downloading {file_name}"
                    )
            safe_download_preprocessed._is_patched_for_windows = True
            sbdb.WaveformBenchmarkDataset._download_preprocessed = safe_download_preprocessed
            logger.debug("SeisBench Windows URL patch diterapkan.")
    except Exception as e:
        logger.debug(f"Tidak dapat menerapkan patch SeisBench Windows: {e}")


def setup_seisbench_repository(use_backup: bool = False) -> str:
    """
    Mengatur repository remote SeisBench dalam sesi Python aktif.
    """
    patch_seisbench_windows_url_bug()
    try:
        import seisbench
        if use_backup:
            seisbench.use_backup_repository()
            logger.info(f"Menggunakan SeisBench Backup Repository: {seisbench.remote_root}")
        else:
            seisbench.remote_root = PRIMARY_REMOTE_ROOT
            logger.info(f"Menggunakan SeisBench Primary Repository: {seisbench.remote_root}")
        return seisbench.remote_root
    except Exception as e:
        logger.warning(f"Gagal mengatur repository SeisBench: {e}")
        return PRIMARY_REMOTE_ROOT if not use_backup else BACKUP_REMOTE_ROOT


# ─────────────────────────────────────────────────────────────
# 2. Pemeriksaan Sumber Data (Remote & Local & Cache)
# ─────────────────────────────────────────────────────────────

def check_remote_repository_health(
    base_url: str,
    dataset_name: str = "stead",
    timeout: int = 10
) -> Dict[str, Any]:
    """
    Melakukan health check remote repository SeisBench menggunakan HTTP HEAD.
    TIDAK mengunduh payload/konten data (zero bytes transferred).
    """
    norm_base = base_url.rstrip("/") + "/"
    target_files = ["metadata.csv", "waveforms.hdf5"]
    results = {
        "url": norm_base,
        "accessible": False,
        "files": {},
        "error": None,
        "latency_sec": None
    }

    start_time = time.time()
    try:
        metadata_url = f"{norm_base}datasets/{dataset_name.lower()}/metadata.csv"
        resp = requests.head(metadata_url, timeout=timeout, allow_redirects=True)
        results["latency_sec"] = round(time.time() - start_time, 2)
        results["accessible"] = (resp.status_code == 200)

        results["files"]["metadata.csv"] = {
            "url": metadata_url,
            "status_code": resp.status_code,
            "size_bytes": int(resp.headers.get("content-length", 0)) if resp.status_code == 200 else None
        }

        # Periksa juga waveforms.hdf5
        wf_url = f"{norm_base}datasets/{dataset_name.lower()}/waveforms.hdf5"
        resp_wf = requests.head(wf_url, timeout=timeout, allow_redirects=True)
        results["files"]["waveforms.hdf5"] = {
            "url": wf_url,
            "status_code": resp_wf.status_code,
            "size_bytes": int(resp_wf.headers.get("content-length", 0)) if resp_wf.status_code == 200 else None
        }

    except requests.exceptions.RequestException as e:
        results["error"] = str(e)
        results["latency_sec"] = round(time.time() - start_time, 2)

    return results


def inspect_seisbench_cache(dataset_name: str = "stead") -> Dict[str, Any]:
    """
    Memeriksa ketersediaan dataset pada cache SeisBench di mesin lokal (~/.seisbench).
    """
    try:
        import seisbench
        cache_dir = Path(seisbench.cache_root) / "datasets" / dataset_name.lower()
    except Exception:
        cache_dir = Path.home() / ".seisbench" / "datasets" / dataset_name.lower()

    info = {
        "cache_dir": str(cache_dir),
        "exists": cache_dir.exists(),
        "files": {},
        "is_complete": False
    }

    if cache_dir.exists():
        has_meta = False
        has_waveforms = False
        for f in cache_dir.iterdir():
            if f.is_file():
                size_mb = round(f.stat().st_size / (1024 * 1024), 2)
                info["files"][f.name] = {"size_mb": size_mb}
                if f.name.startswith("metadata") and f.suffix == ".csv":
                    has_meta = True
                if f.name.startswith("waveforms") and f.suffix == ".hdf5":
                    has_waveforms = True
        info["is_complete"] = (has_meta and has_waveforms)

    return info


def inspect_local_stead(local_path: str) -> Dict[str, Any]:
    """
    Memeriksa direktori dataset STEAD lokal (asli atau terkonversi).
    
    Status yang dapat dikembalikan:
    - 'READY': Metadata dan Waveform HDF5 lengkap dan valid.
    - 'METADATA_ONLY': Metadata CSV ditemukan, tetapi waveform HDF5 belum ada.
    - 'WAVEFORMS_ONLY': Waveform HDF5 ditemukan, tetapi metadata CSV belum ada.
    - 'NOT_FOUND': Direktori kosong atau tidak ditemukan.
    - 'INVALID': File rusak atau format tidak dikenali.
    """
    p = Path(local_path)
    report = {
        "local_path": str(p),
        "exists": p.exists(),
        "status": "NOT_FOUND",
        "format": None,
        "metadata_file": None,
        "waveform_file": None,
        "n_records": 0,
        "label_distribution": {},
        "waveform_sample_shape": None,
        "channels": None,
        "notes": []
    }

    if not p.exists():
        report["notes"].append("Direktori lokal belum dibuat.")
        return report

    # 1. Cari file metadata
    meta_candidates = ["merged.csv", "metadata.csv", "stead_metadata.csv"]
    found_meta = None
    for cand in meta_candidates:
        cand_path = p / cand
        if cand_path.is_file() and cand_path.stat().st_size > 0:
            found_meta = cand_path
            break

    # 2. Cari file waveform HDF5
    wf_candidates = ["merged.hdf5", "waveforms.hdf5"]
    found_wf = None
    for cand in wf_candidates:
        cand_path = p / cand
        if cand_path.is_file() and cand_path.stat().st_size > 0:
            found_wf = cand_path
            break
    if not found_wf:
        # Cek apakah ada chunk *.hdf5
        hdf5_files = list(p.glob("*.hdf5")) + list(p.glob("*.h5"))
        if hdf5_files:
            found_wf = hdf5_files[0]

    report["metadata_file"] = str(found_meta) if found_meta else None
    report["waveform_file"] = str(found_wf) if found_wf else None

    # Evaluasi Metadata
    meta_valid = False
    if found_meta:
        try:
            # Baca sample awal untuk validasi cepat
            sample_df = pd.read_csv(found_meta, nrows=100)
            col_lower = {c.lower(): c for c in sample_df.columns}

            # Deteksi trace category / label
            cat_col = next((col_lower[c] for c in ["trace_category", "category", "label", "type"] if c in col_lower), None)
            name_col = next((col_lower[c] for c in ["trace_name", "trace_id", "id", "fname"] if c in col_lower), None)

            if cat_col and name_col:
                meta_valid = True
                full_df = pd.read_csv(found_meta, usecols=[cat_col, name_col], low_memory=False)
                report["n_records"] = len(full_df)
                report["label_distribution"] = full_df[cat_col].value_counts().to_dict()
                report["format"] = "STEAD_original" if "trace_category" in sample_df.columns else "generic_csv"
            else:
                report["notes"].append(f"Metadata ditemukan di {found_meta.name}, namun kolom label/trace_name tidak valid.")
        except Exception as e:
            report["notes"].append(f"Gagal membaca metadata: {e}")

    # Evaluasi Waveform HDF5
    wf_valid = False
    if found_wf:
        try:
            import h5py
            with h5py.File(found_wf, "r") as f:
                # Periksa struktur STEAD ('data' group) atau SeisBench root
                if "data" in f:
                    data_group = f["data"]
                    first_key = next(iter(data_group.keys()), None)
                    if first_key:
                        arr = data_group[first_key][()]
                        report["waveform_sample_shape"] = list(arr.shape)
                        wf_valid = True
                else:
                    # Coba root keys
                    first_key = next(iter(f.keys()), None)
                    if first_key and isinstance(f[first_key], h5py.Dataset):
                        arr = f[first_key][()]
                        report["waveform_sample_shape"] = list(arr.shape)
                        wf_valid = True
        except Exception as e:
            report["notes"].append(f"File HDF5 ditemukan tetapi gagal diakses: {e}")

    # Tentukan Status
    if meta_valid and wf_valid:
        report["status"] = "READY"
    elif meta_valid and not wf_valid:
        report["status"] = "METADATA_ONLY"
        report["notes"].append("Hanya metadata yang tersedia; file waveform HDF5 belum ditemukan.")
    elif not meta_valid and wf_valid:
        report["status"] = "WAVEFORMS_ONLY"
        report["notes"].append("File waveform HDF5 ditemukan, tetapi metadata CSV tidak valid atau tidak ada.")
    elif not found_meta and not found_wf:
        report["status"] = "NOT_FOUND"
    else:
        report["status"] = "INVALID"

    return report


def inspect_processed_data(processed_dir: str = "data/processed") -> Dict[str, Any]:
    """
    Memeriksa dataset yang sudah diproses (.npz) untuk training.
    """
    p = Path(processed_dir)
    res = {
        "dir": str(p),
        "exists": p.exists(),
        "splits": {},
        "is_synthetic": False,
        "is_ready_for_training": False
    }

    if not p.exists():
        return res

    all_present = True
    for split in ["train", "val", "test"]:
        npz_file = p / f"{split}.npz"
        if npz_file.is_file():
            try:
                data = np.load(npz_file, allow_pickle=False)
                wf = data["waveforms"]
                lb = data["labels"]
                res["splits"][split] = {
                    "total": len(lb),
                    "earthquake": int(np.sum(lb == 1)),
                    "noise": int(np.sum(lb == 0)),
                    "shape": list(wf.shape)
                }
            except Exception as e:
                res["splits"][split] = {"error": str(e)}
                all_present = False
        else:
            all_present = False

    meta_file = p / "metadata.csv"
    if meta_file.is_file():
        try:
            mdf = pd.read_csv(meta_file, nrows=50)
            if "is_synthetic" in mdf.columns and bool(mdf["is_synthetic"].iloc[0]):
                res["is_synthetic"] = True
        except Exception:
            pass

    res["is_ready_for_training"] = all_present and len(res["splits"]) == 3
    return res


# ─────────────────────────────────────────────────────────────
# 3. Fase Inspeksi Utama (phase_inspect)
# ─────────────────────────────────────────────────────────────

def phase_inspect(config: dict) -> Dict[str, Any]:
    """
    Inspeksi komprehensif alur data secara aman tanpa mengunduh dataset besar.
    """
    logger.info("=" * 68)
    logger.info("FASE INSPEKSI DATASET — AUDIT KESIAPAN & SUMBER DATA")
    logger.info("=" * 68)

    dataset_cfg = config.get("dataset", {})
    prep_cfg = config.get("preprocessing", {})
    dataset_name = dataset_cfg.get("name", "STEAD")
    local_path = dataset_cfg.get("local_path", "data/raw/stead")
    processed_dir = prep_cfg.get("processed_dir", "data/processed")
    use_backup = dataset_cfg.get("use_backup_repository", False)

    # 1. Cek SeisBench
    sb_ok, sb_ver = check_seisbench()
    patch_seisbench_windows_url_bug()

    # 2. Cek Cache SeisBench Lokal
    cache_info = inspect_seisbench_cache(dataset_name)

    # 3. Cek Data STEAD Lokal
    local_info = inspect_local_stead(local_path)

    # 4. Cek Remote Repository (Zero-byte download)
    remote_primary = check_remote_repository_health(PRIMARY_REMOTE_ROOT, dataset_name=dataset_name)
    remote_backup = check_remote_repository_health(BACKUP_REMOTE_ROOT, dataset_name=dataset_name)

    # 5. Cek Data Processed (NPZ)
    proc_info = inspect_processed_data(processed_dir)

    # Cetak Ringkasan Terstruktur
    print("\n" + "=" * 68)
    print("[AUDIT] RINGKASAN AUDIT DATASET SEISMIK")
    print("=" * 68)
    print(f"* SeisBench Library     : {'v' + sb_ver if sb_ok else '[NOT INSTALLED]'}")
    print(f"* Dataset Target        : {dataset_name}")
    print(f"* Direktori Lokal       : {local_path}")
    print(f"  +- Status             : [{local_info['status']}]")
    if local_info["metadata_file"]:
        print(f"  +- Metadata           : {local_info['metadata_file']} ({local_info['n_records']:,} rekaman)")
    if local_info["waveform_file"]:
        print(f"  +- Waveform HDF5      : {local_info['waveform_file']}")
    for note in local_info["notes"]:
        print(f"  +- Catatan            : [WARN] {note}")

    print(f"\n* Cache SeisBench       : {cache_info['cache_dir']}")
    print(f"  +- Status             : {'[COMPLETE]' if cache_info['is_complete'] else '[EMPTY / INCOMPLETE]'}")
    if cache_info["files"]:
        for fn, fmeta in cache_info["files"].items():
            print(f"     - {fn} ({fmeta['size_mb']} MB)")

    print("\n* Remote Repository SeisBench (Pengecekan HTTP Zero-Byte):")
    p_status = "200 OK" if remote_primary["accessible"] else f"FAIL ({remote_primary['error'] or 'Error'})"
    b_status = "200 OK" if remote_backup["accessible"] else f"FAIL ({remote_backup['error'] or 'Error'})"
    print(f"  1. Primary (DESY)     : [{p_status}] ({remote_primary['latency_sec']}s)")
    if remote_primary["accessible"] and "metadata.csv" in remote_primary["files"]:
        m_sz = remote_primary["files"]["metadata.csv"].get("size_bytes", 0) or 0
        w_sz = remote_primary["files"]["waveforms.hdf5"].get("size_bytes", 0) or 0
        print(f"     +- metadata.csv    : {m_sz / (1024*1024):.1f} MB")
        print(f"     +- waveforms.hdf5  : {w_sz / (1024*1024*1024):.1f} GB")

    print(f"  2. Backup (GFZ)       : [{b_status}] ({remote_backup['latency_sec']}s)")

    print(f"\n* Data Processed (NPZ)  : {processed_dir}")
    print(f"  +- Siap Training      : {'[YA]' if proc_info['is_ready_for_training'] else '[BELUM SIAP]'}")
    if proc_info["is_synthetic"]:
        print("  +- Tipe Data          : [DATA SINTETIS] (Hanya untuk pengujian pipeline)")
    elif proc_info["is_ready_for_training"]:
        print("  +- Tipe Data          : [DATA SEISMIK NYATA]")

    for s_name, s_dat in proc_info["splits"].items():
        if "total" in s_dat:
            print(f"     - {s_name:5s} : {s_dat['total']:,} trace (EQ={s_dat['earthquake']}, Noise={s_dat['noise']})")

    # Rekomendasi & Langkah Selanjutnya
    print("\n" + "=" * 68)
    print("[REKOMENDASI & TINDAKAN LANJUTAN]:")
    print("=" * 68)

    if local_info["status"] == "READY":
        print("1. [DATA NYATA LOKAL SIAP] Jalankan pembentukan subset metadata:")
        print("   python src/data_loader.py --config configs/config.yaml --phase download")
        print("   Lalu jalankan preprocessing: python src/preprocessing.py --config configs/config.yaml")

    elif local_info["status"] == "METADATA_ONLY":
        print("1. [METADATA SAJA] File metadata ditemukan, tetapi waveform HDF5 belum ada.")
        print("   Silakan letakkan file waveform ('merged.hdf5') di direktori:", local_path)

    elif proc_info["is_ready_for_training"] and proc_info["is_synthetic"]:
        print("1. [DATA SINTETIS AKTIF] Pipeline training dan inferensi dapat diuji sekarang:")
        print("   python src/train.py --config configs/config.yaml --model cnn_spectrogram")
        print("   Catatan: Untuk tugas akhir / evaluasi akademik akhir, gunakan data STEAD nyata.")

    else:
        print("1. Untuk memulai pengembangan dan verifikasi pipeline sekarang, buat data sintetis:")
        print("   python src/data_loader.py --config configs/config.yaml --phase synthetic --n 6000")
        print("2. Untuk menggunakan STEAD nyata:")
        print("   Unduh file STEAD ('merged.csv' dan 'merged.hdf5') lalu letakkan di:")
        print(f"   {Path(local_path).resolve()}")

    print("=" * 68 + "\n")

    return {
        "seisbench_version": sb_ver,
        "local_info": local_info,
        "cache_info": cache_info,
        "remote_primary": remote_primary,
        "remote_backup": remote_backup,
        "processed_info": proc_info
    }


# ─────────────────────────────────────────────────────────────
# 4. Pembuatan Balanced Subset & Leakage-Free Split
# ─────────────────────────────────────────────────────────────

def create_balanced_subset(
    metadata: pd.DataFrame,
    n_earthquake: int,
    n_noise: int,
    min_snr_db: float = 10.0,
    seed: int = 42,
    category_col: Optional[str] = None,
    eq_label: str = "earthquake_local",
    noise_label: str = "noise",
    source_name: str = "STEAD"
) -> pd.DataFrame:
    """
    Membuat subset seimbang antara earthquake dan noise dari metadata.
    
    Fitur robustness:
    - Label eksplisit sesuai config.
    - Toleransi terhadap kolom SNR yang missing, bernilai string invalid, atau NaN.
    - Graceful handling jika jumlah data tersedia lebih sedikit daripada target.
    - Melacak asal sumber data (data lineage).
    """
    if metadata.empty:
        raise ValueError("Metadata kosong, tidak dapat membuat subset.")

    # Deteksi kolom kategori/label
    if category_col is None or category_col not in metadata.columns:
        col_lower = {c.lower(): c for c in metadata.columns}
        for candidate in ["trace_category", "category", "label", "type"]:
            if candidate in col_lower:
                category_col = col_lower[candidate]
                break

    if category_col is None or category_col not in metadata.columns:
        raise ValueError("Kolom kategori label tidak ditemukan dalam metadata.")

    # Identifikasi baris gempa dan noise
    eq_mask = metadata[category_col].astype(str).str.lower() == str(eq_label).lower()
    noise_mask = metadata[category_col].astype(str).str.lower() == str(noise_label).lower()

    eq_df = metadata[eq_mask].copy()
    noise_df = metadata[noise_mask].copy()

    logger.info(f"Kandidat tersedia — Earthquake: {len(eq_df):,}, Noise: {len(noise_df):,}")

    # Filter SNR untuk gempa
    snr_col = next((c for c in eq_df.columns if "snr" in c.lower()), None)
    if snr_col and min_snr_db > 0:
        # Konversi ke numerik secara aman (non-numerik menjadi NaN)
        snr_numeric = pd.to_numeric(eq_df[snr_col], errors="coerce")
        valid_snr_mask = snr_numeric.notna() & (snr_numeric >= min_snr_db)
        before_count = len(eq_df)
        filtered_eq = eq_df[valid_snr_mask].copy()
        if len(filtered_eq) > 0:
            eq_df = filtered_eq
            logger.info(f"Filter SNR >= {min_snr_db} dB: {before_count:,} → {len(eq_df):,} earthquake")
        else:
            logger.warning(f"Filter SNR >= {min_snr_db} dB menyisakan 0 gempa! Melewatkan filter SNR.")

    # Validasi kuantitas yang diminta vs yang tersedia
    n_eq_use = min(n_earthquake, len(eq_df))
    n_noise_use = min(n_noise, len(noise_df))

    if n_eq_use < n_earthquake:
        logger.warning(f"Hanya {n_eq_use:,} rekaman earthquake tersedia (diminta {n_earthquake:,}). Menggunakan {n_eq_use:,}.")
    if n_noise_use < n_noise:
        logger.warning(f"Hanya {n_noise_use:,} rekaman noise tersedia (diminta {n_noise:,}). Menggunakan {n_noise_use:,}.")

    if n_eq_use == 0 and n_noise_use == 0:
        raise ValueError("Tidak ada rekaman earthquake atau noise yang cocok untuk dibentuk subset.")

    eq_sub = eq_df.sample(n=n_eq_use, random_state=seed) if n_eq_use > 0 else pd.DataFrame()
    noise_sub = noise_df.sample(n=n_noise_use, random_state=seed) if n_noise_use > 0 else pd.DataFrame()

    subset = pd.concat([eq_sub, noise_sub], ignore_index=True)
    subset = subset.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    # Tambahkan mapping label biner numerik
    subset["label"] = subset[category_col].apply(
        lambda x: 1 if str(x).lower() == str(eq_label).lower() else 0
    )
    subset["metadata_source"] = source_name
    subset["subset_created_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    logger.info(f"Subset berhasil dibentuk: {len(subset):,} rekaman (EQ={n_eq_use:,}, Noise={n_noise_use:,})")
    return subset


def split_by_network(
    subset: pd.DataFrame,
    train: float = 0.70,
    val: float = 0.15,
    test: float = 0.15,
    seed: int = 42
) -> pd.DataFrame:
    """
    Membagi dataset berdasarkan stasiun/jaringan seismik (network-based split)
    untuk mencegah data leakage antarsplit.
    
    Jaminan:
    - Tidak ada stasiun/network yang muncul di lebih dari 1 split.
    - Train, validation, dan test dipastikan TIDAK KOSONG.
    - Jika network terlalu sedikit (< 3), otomatis beralih ke stratified split
      dengan peringatan dokumentasi yang jelas.
    """
    if subset.empty:
        raise ValueError("Subset kosong, tidak dapat melakukan split.")

    subset = subset.copy()
    net_col = next((c for c in ["station_network_code", "network_code", "network"] if c in subset.columns), None)

    # Fallback: Jika tidak ada kolom network atau network unik < 3
    unique_nets = subset[net_col].dropna().unique() if net_col else []
    if net_col is None or len(unique_nets) < 3:
        logger.warning(
            f"Jumlah jaringan unik ({len(unique_nets)}) tidak mencukupi untuk split berbasis jaringan (butuh >= 3). "
            "Menggunakan stratified random split bebas kelompok untuk menjamin ketiga split tidak kosong."
        )
        return _stratified_split(subset, train=train, val=val, test=test, seed=seed)

    # Grouping berdasarkan network
    np.random.seed(seed)
    shuffled_nets = list(unique_nets)
    np.random.shuffle(shuffled_nets)

    # Hitung bobot sampel per network
    net_counts = subset[net_col].value_counts().to_dict()
    total_samples = len(subset)

    target_train = int(total_samples * train)
    target_val = int(total_samples * val)

    train_nets, val_nets, test_nets = set(), set(), set()
    train_count, val_count = 0, 0

    for net in shuffled_nets:
        c = net_counts.get(net, 0)
        if train_count < target_train or not train_nets:
            train_nets.add(net)
            train_count += c
        elif val_count < target_val or not val_nets:
            val_nets.add(net)
            val_count += c
        else:
            test_nets.add(net)

    # Pastikan test_nets tidak kosong
    if not test_nets:
        # Pindahkan satu network dari train atau val ke test
        if len(train_nets) > 1:
            moved = train_nets.pop()
            test_nets.add(moved)
        elif len(val_nets) > 1:
            moved = val_nets.pop()
            test_nets.add(moved)

    def assign_split(row):
        n = row.get(net_col)
        if n in train_nets:
            return "train"
        if n in val_nets:
            return "val"
        return "test"

    subset["split"] = subset.apply(assign_split, axis=1)

    # Verifikasi tidak ada split yang kosong
    counts = subset["split"].value_counts().to_dict()
    for s in ["train", "val", "test"]:
        if counts.get(s, 0) == 0:
            logger.warning(f"Split by network menghasilkan split '{s}' kosong. Menggunakan stratified fallback.")
            return _stratified_split(subset, train=train, val=val, test=test, seed=seed)

    logger.info(f"Distribusi Split by Network (Disjoint Networks):\n{subset['split'].value_counts()}")
    return subset


def _stratified_split(df: pd.DataFrame, train=0.70, val=0.15, test=0.15, seed=42) -> pd.DataFrame:
    """Stratified split internal pembagi label."""
    np.random.seed(seed)
    df = df.copy()
    splits = np.array(["train"] * len(df))

    # Kelompokkan per label jika tersedia
    label_col = "label" if "label" in df.columns else None
    if label_col:
        for lbl in df[label_col].unique():
            idx = df[df[label_col] == lbl].index.to_numpy()
            np.random.shuffle(idx)
            n = len(idx)
            n_tr = max(1, int(n * train))
            n_va = max(1, int(n * val))
            # Pastikan test minimal 1 jika n >= 3
            if n_tr + n_va >= n and n >= 3:
                n_tr = n - 2
                n_va = 1

            splits[idx[:n_tr]] = "train"
            splits[idx[n_tr:n_tr + n_va]] = "val"
            splits[idx[n_tr + n_va:]] = "test"
    else:
        idx = np.arange(len(df))
        np.random.shuffle(idx)
        n = len(df)
        n_tr = int(n * train)
        n_va = int(n * val)
        splits[idx[:n_tr]] = "train"
        splits[idx[n_tr:n_tr + n_va]] = "val"
        splits[idx[n_tr + n_va:]] = "test"

    df["split"] = splits
    return df


# ─────────────────────────────────────────────────────────────
# 5. Data Sintetis untuk Development & Verifikasi Pipeline
# ─────────────────────────────────────────────────────────────

def generate_synthetic_dataset(
    n_earthquake: int = 5000,
    n_noise: int = 1000,
    sampling_rate: int = 100,
    n_samples: int = 6000,
    seed: int = 42,
    output_dir: str = "data/processed"
) -> str:
    """
    Membuat dataset sintetis khusus untuk pengujian alur pipeline end-to-end.
    
    ⚠️ BATASAN PENTING:
    - Dataset ini HANYA untuk development dan verifikasi kode secara teknis.
    - Sinyal sintetis BUKAN bukti performa model ilmiah pada gempa bumi nyata.
    - Setiap rekaman metadata diberi flag `is_synthetic = True`.
    """
    np.random.seed(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.warning("=" * 64)
    logger.warning("PERINGATAN: Membangun DATA SINTETIS!")
    logger.warning("Dataset ini ditandai is_synthetic=True untuk pengujian pipeline.")
    logger.warning("Jangan gunakan hasil sintetis sebagai klaim ilmiah performa nyata.")
    logger.warning("=" * 64)

    t = np.linspace(0, n_samples / sampling_rate, n_samples, dtype=np.float32)
    waveforms = []
    labels = []

    # 1. Sinyal Earthquake Sintetis
    for i in range(n_earthquake):
        arrival = np.random.randint(n_samples // 5, n_samples // 2)
        freq = np.random.uniform(2.0, 12.0)
        decay = np.random.uniform(0.04, 0.15)
        amp = np.random.uniform(0.5, 1.0)
        noise_level = np.random.uniform(0.02, 0.08)

        signal = np.zeros((3, n_samples), dtype=np.float32)
        for ch in range(3):
            ch_freq = freq * np.random.uniform(0.9, 1.1)
            ch_amp = amp * np.random.uniform(0.7, 1.0)
            body = ch_amp * np.sin(2 * np.pi * ch_freq * t) * np.exp(-decay * t)
            full = np.zeros(n_samples, dtype=np.float32)
            end = min(arrival + n_samples, n_samples)
            body_len = end - arrival
            full[arrival:end] = body[:body_len]
            full += np.random.randn(n_samples).astype(np.float32) * noise_level

            # Peak normalize
            mx = np.max(np.abs(full))
            if mx > 1e-8:
                full /= mx
            signal[ch] = full

        waveforms.append(signal)
        labels.append(1)

    # 2. Sinyal Noise Sintetis
    for i in range(n_noise):
        signal = np.zeros((3, n_samples), dtype=np.float32)
        for ch in range(3):
            amp = np.random.uniform(0.02, 0.1)
            low_freq = np.random.uniform(0.1, 1.0)
            noise_ch = amp * np.sin(2 * np.pi * low_freq * t)
            noise_ch += np.random.randn(n_samples).astype(np.float32) * amp * 0.5
            mx = np.max(np.abs(noise_ch))
            if mx > 1e-8:
                noise_ch /= mx
            signal[ch] = noise_ch
        waveforms.append(signal)
        labels.append(0)

    waveforms = np.array(waveforms, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    # Permutasi acak
    perm = np.random.permutation(len(labels))
    waveforms = waveforms[perm]
    labels = labels[perm]

    # Metadata
    networks = ["IU", "II", "IA", "GE", "CI"]
    meta_records = []
    for i in range(len(labels)):
        is_eq = (labels[i] == 1)
        meta_records.append({
            "trace_name": f"synthetic_{i:06d}",
            "trace_category": "earthquake_local" if is_eq else "noise",
            "label": int(labels[i]),
            "station_network_code": np.random.choice(networks),
            "snr_db": round(float(np.random.uniform(12.0, 35.0) if is_eq else np.random.uniform(-5.0, 4.0)), 2),
            "is_synthetic": True,
        })

    meta_df = pd.DataFrame(meta_records)
    meta_df = split_by_network(meta_df, train=0.70, val=0.15, test=0.15, seed=seed)

    split_counts = {}
    for split in ["train", "val", "test"]:
        mask = meta_df["split"] == split
        split_idx = np.where(mask.values)[0]
        split_wf = waveforms[split_idx]
        split_lb = labels[split_idx]

        out_file = out_path / f"{split}.npz"
        np.savez_compressed(out_file, waveforms=split_wf, labels=split_lb)
        meta_df[mask].to_csv(out_path / f"{split}_metadata.csv", index=False)

        split_counts[split] = {
            "total": len(split_idx),
            "earthquake": int(np.sum(split_lb == 1)),
            "noise": int(np.sum(split_lb == 0))
        }
        logger.info(f"Split '{split}': {len(split_idx)} trace disimpan ke {out_file.name}")

    # Simpan metadata lengkap
    meta_df.to_csv(out_path / "metadata.csv", index=False)

    log_content = {
        "mode": "SYNTHETIC_DEVELOPMENT",
        "description": "Dataset sintetis untuk verifikasi arsitektur & pipeline",
        "is_synthetic": True,
        "n_earthquake": n_earthquake,
        "n_noise": n_noise,
        "sampling_rate": sampling_rate,
        "n_samples": n_samples,
        "splits": split_counts
    }
    with open(out_path / "preprocessing_log.json", "w", encoding="utf-8") as f:
        json.dump(log_content, f, indent=2)

    logger.info(f"Dataset sintetis sukses dibuat di: {out_path}")
    return str(out_path)


# ─────────────────────────────────────────────────────────────
# 6. Fase Download / Pembentukan Subset Nyata (phase_download)
# ─────────────────────────────────────────────────────────────

def phase_download(config: dict) -> None:
    """
    Fase download / pembentukan subset dataset.
    Menerapkan validasi ketat agar tidak mengunduh puluhan gigabyte secara tidak sengaja.
    """
    logger.info("=" * 68)
    logger.info("FASE DOWNLOAD / SUBSET — MEMBUAT SUBSET DARI SUMBER TERVALIDASI")
    logger.info("=" * 68)

    dataset_cfg = config.get("dataset", {})
    source_type = dataset_cfg.get("source", "seisbench")
    local_path = dataset_cfg.get("local_path", "data/raw/stead")
    subset_cfg = dataset_cfg.get("subset", {})
    split_cfg = dataset_cfg.get("split", {})
    prep_cfg = config.get("preprocessing", {})
    metadata_out = prep_cfg.get("metadata_file", "data/processed/metadata.csv")
    allow_large = dataset_cfg.get("allow_large_download", False)

    # 1. Jalankan inspeksi lokal
    local_info = inspect_local_stead(local_path)

    if local_info["status"] == "READY":
        logger.info(f"Dataset lokal terverifikasi di: {local_path}")
        logger.info("Membaca metadata lokal untuk membentuk subset...")
        meta_df = pd.read_csv(local_info["metadata_file"], low_memory=False)

        labels_cfg = dataset_cfg.get("labels", {})
        eq_cat = labels_cfg.get("earthquake_category", "earthquake_local")
        noise_cat = labels_cfg.get("noise_category", "noise")

        subset = create_balanced_subset(
            meta_df,
            n_earthquake=subset_cfg.get("n_earthquake", 5000),
            n_noise=subset_cfg.get("n_noise", 1000),
            min_snr_db=subset_cfg.get("min_snr_db", 10.0),
            seed=subset_cfg.get("seed", 42),
            eq_label=eq_cat,
            noise_label=noise_cat,
            source_name=f"local_{dataset_cfg.get('name', 'STEAD')}"
        )

        subset = split_by_network(
            subset,
            train=split_cfg.get("train", 0.70),
            val=split_cfg.get("val", 0.15),
            test=split_cfg.get("test", 0.15),
            seed=subset_cfg.get("seed", 42)
        )

        Path(metadata_out).parent.mkdir(parents=True, exist_ok=True)
        subset.to_csv(metadata_out, index=False)
        logger.info(f"Metadata subset tersimpan: {metadata_out}")
        logger.info("Langkah selanjutnya:")
        logger.info("  python src/preprocessing.py --config configs/config.yaml")
        return

    elif local_info["status"] == "METADATA_ONLY":
        logger.error("Dataset lokal hanya memiliki metadata. Waveform HDF5 belum ada!")
        logger.error(f"Silakan letakkan file waveform HDF5 di: {local_path}")
        return

    # 2. Jika sumber adalah SeisBench remote
    if source_type == "seisbench":
        logger.warning("Dataset lokal tidak ditemukan. Mencoba memeriksa cache / remote SeisBench...")
        cache_info = inspect_seisbench_cache(dataset_cfg.get("name", "STEAD"))

        if not cache_info["is_complete"]:
            if not allow_large:
                logger.error("=" * 68)
                logger.error("PENGHENTIAN PENGUNDUHAN OTOMATIS (SAFEGUARD):")
                logger.error("Dataset STEAD penuh via SeisBench berukuran ~91.1 GB (waveforms.hdf5).")
                logger.error("Sesuai kebijakan keamanan data, pengunduhan puluhan GB tidak dimulai otomatis.")
                logger.error("Solusi yang disarankan:")
                logger.error(f"  1. Unduh dataset STEAD manual dan letakkan di: {Path(local_path).resolve()}")
                logger.error("  2. Atau gunakan mode sintetis untuk menguji seluruh pipeline model:")
                logger.error("     python src/data_loader.py --config configs/config.yaml --phase synthetic")
                logger.error("  3. Jika kuota dan disk mencukupi (>100GB) dan ingin mengunduh SeisBench penuh,")
                logger.error("     setel 'allow_large_download: true' di configs/config.yaml.")
                logger.error("=" * 68)
                return

        # Jika diizinkan atau sudah dalam cache
        setup_seisbench_repository(use_backup=dataset_cfg.get("use_backup_repository", False))
        import seisbench.data as sbd
        try:
            ds = sbd.STEAD()
            meta_df = ds.metadata
            subset = create_balanced_subset(
                meta_df,
                n_earthquake=subset_cfg.get("n_earthquake", 5000),
                n_noise=subset_cfg.get("n_noise", 1000),
                min_snr_db=subset_cfg.get("min_snr_db", 10.0),
                seed=subset_cfg.get("seed", 42)
            )
            subset = split_by_network(subset, seed=subset_cfg.get("seed", 42))
            Path(metadata_out).parent.mkdir(parents=True, exist_ok=True)
            subset.to_csv(metadata_out, index=False)
            logger.info(f"Subset metadata SeisBench disimpan ke {metadata_out}")
        except Exception as e:
            logger.error(f"Gagal memuat SeisBench STEAD: {e}")


# ─────────────────────────────────────────────────────────────
# 7. Entry Point CLI
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Data Loader & Dataset Audit — Earthquake Detection")
    parser.add_argument("--config", default="configs/config.yaml", help="Path konfigurasi YAML")
    parser.add_argument(
        "--phase",
        choices=["inspect", "download", "synthetic"],
        default="inspect",
        help="inspect: audit data tanpa unduh; download: buat subset; synthetic: data sintetis"
    )
    parser.add_argument(
        "--n",
        type=int,
        default=6000,
        help="Jumlah rekaman untuk mode --phase synthetic"
    )
    args = parser.parse_args()

    # Pastikan current working directory konsisten di root proyek
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    os.chdir(project_root)

    cfg = load_config(args.config)

    if args.phase == "inspect":
        phase_inspect(cfg)
    elif args.phase == "download":
        phase_download(cfg)
    elif args.phase == "synthetic":
        wf_cfg = cfg.get("dataset", {}).get("waveform", {})
        prep_cfg = cfg.get("preprocessing", {})
        n_total = args.n
        n_eq = int(n_total * 0.833)
        n_noise = n_total - n_eq
        generate_synthetic_dataset(
            n_earthquake=n_eq,
            n_noise=n_noise,
            sampling_rate=wf_cfg.get("sampling_rate", 100),
            n_samples=wf_cfg.get("n_samples", 6000),
            seed=cfg.get("dataset", {}).get("subset", {}).get("seed", 42),
            output_dir=prep_cfg.get("processed_dir", "data/processed")
        )


if __name__ == "__main__":
    main()
