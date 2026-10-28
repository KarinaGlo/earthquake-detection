"""
data_loader.py  (v2 — Multi-Strategy)
======================================
Strategi download bertingkat:
  1. SeisBench STEAD (primary repo)
  2. SeisBench backup repo  (seisbench.use_backup_repository)
  3. Dataset alternatif kecil via SeisBench (GEOFON / LenDB)
  4. Download metadata STEAD CSV manual dari GitHub release
  5. Generate data sintetis minimal untuk development/testing

Penggunaan:
    python src/data_loader.py --config configs/config.yaml --phase inspect
    python src/data_loader.py --config configs/config.yaml --phase download
    python src/data_loader.py --config configs/config.yaml --phase synthetic --n 5000
"""

import os, sys, argparse, logging, yaml, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("data_loader")


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_seisbench() -> bool:
    try:
        import seisbench
        logger.info(f"SeisBench versi: {seisbench.__version__}")
        return True
    except ImportError:
        logger.error("SeisBench tidak terinstal. Jalankan: pip install seisbench")
        return False


# ─────────────────────────────────────────────────────────────
# Strategi 1 & 2 — SeisBench auto-download (primary + backup)
# ─────────────────────────────────────────────────────────────

def try_seisbench_dataset(dataset_name: str, use_backup: bool = False):
    """
    Coba load dataset via SeisBench.
    Mengembalikan object dataset atau None jika gagal.
    """
    import seisbench
    import seisbench.data as sbd

    if use_backup:
        seisbench.use_backup_repository()
        logger.info("Menggunakan SeisBench backup repository (desy.de)...")

    cls = getattr(sbd, dataset_name, None)
    if cls is None:
        logger.error(f"Dataset '{dataset_name}' tidak dikenal di SeisBench.")
        return None

    try:
        logger.info(f"Mengakses {dataset_name} via SeisBench{'  [backup]' if use_backup else ''}...")
        data = cls()
        logger.info(f"Berhasil! Total rekaman: {len(data):,}")
        return data
    except Exception as e:
        logger.warning(f"{'Backup' if use_backup else 'Primary'} repo gagal: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Strategi 3 — Dataset Alternatif Kecil
# ─────────────────────────────────────────────────────────────

ALTERNATIVE_DATASETS = [
    # (nama_class, keterangan)
    ("LenDB",  "LenDB — Italian local earthquakes, ~1.3 juta trace, 100 Hz"),
    ("GEOFON", "GEOFON — Global teleseismic, 100 Hz"),
    ("NEIC",   "NEIC — USGS global dataset"),
]

def try_alternative_seisbench(use_backup: bool = False):
    """Coba semua dataset alternatif yang mungkin tersedia."""
    import seisbench
    if use_backup:
        seisbench.use_backup_repository()

    import seisbench.data as sbd
    for name, desc in ALTERNATIVE_DATASETS:
        logger.info(f"Mencoba dataset alternatif: {name} ({desc})...")
        try:
            cls = getattr(sbd, name)
            data = cls()
            logger.info(f"Berhasil memuat {name}! Total rekaman: {len(data):,}")
            return data, name
        except Exception as e:
            logger.warning(f"  {name} gagal: {e}")
    return None, None


# ─────────────────────────────────────────────────────────────
# Strategi 4 — Download STEAD CSV manual
# ─────────────────────────────────────────────────────────────

STEAD_METADATA_URLS = [
    # URL resmi dari STEAD GitHub releases
    "https://github.com/smousavi05/STEAD/releases/download/v1.0/metadata.csv",
    # Backup mirror (jika ada)
    "https://raw.githubusercontent.com/smousavi05/STEAD/master/metadata.csv",
]

def download_stead_csv_manual(output_dir: str) -> Optional[str]:
    """
    Download file metadata STEAD CSV secara manual dari GitHub.
    Hanya metadata (~8 MB), tidak termasuk waveform HDF5.

    Returns:
        Path ke file CSV yang diunduh, atau None jika gagal
    """
    import requests

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "stead_metadata.csv"

    if output_path.exists():
        logger.info(f"Metadata STEAD sudah ada di: {output_path}")
        return str(output_path)

    for url in STEAD_METADATA_URLS:
        logger.info(f"Mencoba download metadata dari: {url}")
        try:
            resp = requests.get(url, timeout=30, stream=True)
            if resp.status_code == 200:
                total = int(resp.headers.get("content-length", 0))
                downloaded = 0
                with open(output_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=1024 * 64):
                        f.write(chunk)
                        downloaded += len(chunk)
                        if total > 0:
                            pct = downloaded / total * 100
                            print(f"\r  Download: {downloaded/1e6:.1f}/{total/1e6:.1f} MB ({pct:.0f}%)", end="")
                print()
                logger.info(f"Download selesai: {output_path} ({output_path.stat().st_size/1e6:.1f} MB)")
                return str(output_path)
            else:
                logger.warning(f"  HTTP {resp.status_code} dari {url}")
        except Exception as e:
            logger.warning(f"  Gagal dari {url}: {e}")

    logger.error("Semua URL download metadata STEAD gagal.")
    return None


def load_stead_csv_as_metadata(csv_path: str) -> Optional[pd.DataFrame]:
    """
    Baca CSV metadata STEAD dan normalisasi ke format internal.

    Kolom STEAD asli -> kolom internal proyek
    """
    try:
        df = pd.read_csv(csv_path, low_memory=False)
        logger.info(f"CSV dimuat: {len(df):,} baris, kolom: {list(df.columns)}")
    except Exception as e:
        logger.error(f"Gagal membaca CSV: {e}")
        return None

    # Normalisasi nama kolom
    rename_map = {}
    col_lower = {c.lower(): c for c in df.columns}

    # Deteksi kolom kategori/label
    for candidate in ["trace_category", "category", "type", "label"]:
        if candidate in col_lower:
            rename_map[col_lower[candidate]] = "trace_category"
            break

    # Deteksi kolom jaringan
    for candidate in ["network_code", "station_network_code", "network"]:
        if candidate in col_lower:
            rename_map[col_lower[candidate]] = "station_network_code"
            break

    # Deteksi kolom trace name
    for candidate in ["trace_name", "trace_id", "id", "fname"]:
        if candidate in col_lower:
            rename_map[col_lower[candidate]] = "trace_name"
            break

    # Deteksi SNR
    for candidate in ["snr_db", "trace_snr_db", "snr"]:
        if candidate in col_lower:
            rename_map[col_lower[candidate]] = "snr_db"
            break

    if rename_map:
        df = df.rename(columns=rename_map)
        logger.info(f"Kolom direname: {rename_map}")

    return df


# ─────────────────────────────────────────────────────────────
# Strategi 5 — Data Sintetis untuk Development
# ─────────────────────────────────────────────────────────────

def generate_synthetic_dataset(
    n_earthquake: int = 2500,
    n_noise: int = 500,
    sampling_rate: int = 100,
    n_samples: int = 6000,
    seed: int = 42,
    output_dir: str = "data/processed"
) -> str:
    """
    Generate dataset sintetis untuk development dan testing pipeline.

    ⚠️ HANYA untuk development — jangan gunakan untuk evaluasi akademik!
    Waveform sintetis tidak merepresentasikan sinyal seismik nyata.

    Sinyal sintetis:
    - Earthquake: sinyal impulsif + decay eksponensial + noise
    - Noise: Gaussian noise dengan variasi amplitudo

    Returns:
        Path ke direktori processed dengan file train/val/test NPZ
    """
    np.random.seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.warning("=" * 60)
    logger.warning("PERINGATAN: Menggunakan DATA SINTETIS!")
    logger.warning("Ini hanya untuk development dan testing pipeline.")
    logger.warning("JANGAN gunakan hasil ini untuk evaluasi akademik!")
    logger.warning("=" * 60)

    t = np.linspace(0, n_samples / sampling_rate, n_samples, dtype=np.float32)
    waveforms = []
    labels = []

    # Generate earthquake signals
    for i in range(n_earthquake):
        arrival = np.random.randint(n_samples // 4, n_samples // 2)
        freq = np.random.uniform(2, 15)
        decay = np.random.uniform(0.05, 0.2)
        amp = np.random.uniform(0.5, 1.0)
        noise_level = np.random.uniform(0.02, 0.1)

        signal = np.zeros((3, n_samples), dtype=np.float32)
        for ch in range(3):
            ch_freq = freq * np.random.uniform(0.8, 1.2)
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

    # Generate noise signals
    for i in range(n_noise):
        signal = np.zeros((3, n_samples), dtype=np.float32)
        for ch in range(3):
            amp = np.random.uniform(0.01, 0.1)
            # Low-freq drift + random noise
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

    # Shuffle
    idx = np.random.permutation(len(labels))
    waveforms = waveforms[idx]
    labels = labels[idx]

    # Buat metadata
    n_total = len(labels)
    networks = ["IU", "II", "IA", "GE", "CI"]
    meta_records = []
    for i in range(n_total):
        meta_records.append({
            "trace_name": f"synthetic_{i:06d}",
            "trace_category": "earthquake_local" if labels[i] == 1 else "noise",
            "label": int(labels[i]),
            "station_network_code": np.random.choice(networks),
            "snr_db": np.random.uniform(10, 40) if labels[i] == 1 else np.random.uniform(-5, 5),
            "source_magnitude": np.random.uniform(1.0, 5.5) if labels[i] == 1 else np.nan,
            "is_synthetic": True,
        })

    meta_df = pd.DataFrame(meta_records)

    # Split by network (same strategy as real data)
    nets = sorted(meta_df["station_network_code"].unique())
    np.random.shuffle(nets)
    n_nets = len(nets)
    train_nets = set(nets[:int(n_nets * 0.7)])
    val_nets = set(nets[int(n_nets * 0.7):int(n_nets * 0.85)])
    test_nets = set(nets[int(n_nets * 0.85):])

    def assign_split(net):
        if net in train_nets:
            return "train"
        elif net in val_nets:
            return "val"
        return "test"

    meta_df["split"] = meta_df["station_network_code"].map(assign_split)

    # Simpan per split
    split_counts = {}
    for split in ["train", "val", "test"]:
        mask = meta_df["split"] == split
        split_idx = np.where(mask.values)[0]
        split_wf = waveforms[split_idx]
        split_lb = labels[split_idx]
        out = output_dir / f"{split}.npz"
        np.savez_compressed(out, waveforms=split_wf, labels=split_lb)
        meta_df[mask].to_csv(output_dir / f"{split}_metadata.csv", index=False)
        split_counts[split] = {
            "total": len(split_idx),
            "earthquake": int(np.sum(split_lb == 1)),
            "noise": int(np.sum(split_lb == 0))
        }
        logger.info(f"  Split '{split}': {len(split_idx)} rekaman → {out}")

    # Simpan metadata lengkap
    meta_df.to_csv(output_dir / "metadata.csv", index=False)

    # Simpan log preprocessing
    log = {
        "mode": "SYNTHETIC — hanya untuk development",
        "n_earthquake": n_earthquake,
        "n_noise": n_noise,
        "sampling_rate": sampling_rate,
        "n_samples": n_samples,
        "splits": split_counts,
        "config": {"normalization": {"method": "peak"}, "target_sr": sampling_rate, "target_samples": n_samples, "n_channels": 3}
    }
    with open(output_dir / "preprocessing_log.json", "w") as f:
        json.dump(log, f, indent=2)

    logger.info(f"Dataset sintetis berhasil dibuat di: {output_dir}")
    logger.info(f"Split counts: {split_counts}")
    logger.warning("INGAT: Ini data SINTETIS. Latih ulang dengan data NYATA untuk penelitian akademik.")
    return str(output_dir)


# ─────────────────────────────────────────────────────────────
# Pembuatan subset & split (untuk metadata DataFrame)
# ─────────────────────────────────────────────────────────────

def create_balanced_subset(
    metadata: pd.DataFrame,
    n_earthquake: int,
    n_noise: int,
    min_snr_db: float = 10.0,
    seed: int = 42,
    category_col: str = "trace_category",
    eq_label: str = "earthquake_local",
    noise_label: str = "noise"
) -> pd.DataFrame:
    """Buat subset seimbang dari metadata."""
    np.random.seed(seed)

    eq_df = metadata[metadata[category_col] == eq_label].copy()
    noise_df = metadata[metadata[category_col] == noise_label].copy()

    logger.info(f"Tersedia — Earthquake: {len(eq_df):,}, Noise: {len(noise_df):,}")

    # Filter SNR
    snr_col = next((c for c in eq_df.columns if "snr" in c.lower()), None)
    if snr_col and min_snr_db > 0:
        eq_df[snr_col] = pd.to_numeric(eq_df[snr_col], errors="coerce")
        before = len(eq_df)
        eq_df = eq_df[eq_df[snr_col] >= min_snr_db]
        logger.info(f"Filter SNR >= {min_snr_db} dB: {before:,} → {len(eq_df):,} earthquake")

    n_eq_use = min(n_earthquake, len(eq_df))
    n_noise_use = min(n_noise, len(noise_df))

    if n_eq_use < n_earthquake:
        logger.warning(f"Hanya {n_eq_use:,} earthquake tersedia (diminta {n_earthquake:,})")
    if n_noise_use < n_noise:
        logger.warning(f"Hanya {n_noise_use:,} noise tersedia (diminta {n_noise:,})")

    eq_sub = eq_df.sample(n=n_eq_use, random_state=seed)
    noise_sub = noise_df.sample(n=n_noise_use, random_state=seed)

    subset = pd.concat([eq_sub, noise_sub], ignore_index=True)
    subset = subset.sample(frac=1, random_state=seed).reset_index(drop=True)
    subset["label"] = subset[category_col].map({eq_label: 1, noise_label: 0})

    logger.info(f"Subset dibuat: {len(subset):,} rekaman (EQ={n_eq_use:,}, Noise={n_noise_use:,})")
    return subset


def split_by_network(subset: pd.DataFrame, train=0.70, val=0.15, test=0.15, seed=42) -> pd.DataFrame:
    """Bagi dataset berdasarkan jaringan seismik (mencegah data leakage)."""
    np.random.seed(seed)
    net_col = next((c for c in ["station_network_code", "network_code", "network"]
                    if c in subset.columns), None)
    if net_col is None:
        logger.warning("Kolom network tidak ditemukan — pakai random split")
        n = len(subset)
        perm = np.random.permutation(n)
        n_train = int(n * train)
        n_val = int(n * val)
        splits = np.array(["test"] * n)
        splits[perm[:n_train]] = "train"
        splits[perm[n_train:n_train + n_val]] = "val"
        subset = subset.copy()
        subset["split"] = splits
        return subset

    nets = sorted(subset[net_col].dropna().unique())
    np.random.shuffle(nets)
    n = len(nets)
    train_nets = set(nets[:int(n * train)])
    val_nets = set(nets[int(n * train):int(n * (train + val))])

    def assign(row):
        net = row[net_col]
        if net in train_nets: return "train"
        if net in val_nets:   return "val"
        return "test"

    subset = subset.copy()
    subset["split"] = subset.apply(assign, axis=1)
    logger.info(f"Split distribution:\n{subset['split'].value_counts()}")
    return subset


def save_subset_metadata(subset: pd.DataFrame, output_path: str):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(output_path, index=False)
    logger.info(f"Metadata tersimpan: {output_path} ({len(subset):,} rekaman)")


# ─────────────────────────────────────────────────────────────
# Fase utama
# ─────────────────────────────────────────────────────────────

def phase_inspect(config: dict) -> None:
    """Inspeksi dataset — coba semua strategi berturut-turut."""
    logger.info("=" * 60)
    logger.info("FASE INSPEKSI — Memeriksa ketersediaan dataset")
    logger.info("=" * 60)

    if not check_seisbench():
        return

    data = None
    dataset_name = "STEAD"

    # Strategi 1: primary repo
    data = try_seisbench_dataset("STEAD", use_backup=False)

    # Strategi 2: backup repo
    if data is None:
        logger.info("Mencoba backup repository SeisBench...")
        data = try_seisbench_dataset("STEAD", use_backup=True)

    # Strategi 3: dataset alternatif
    if data is None:
        logger.info("STEAD tidak tersedia. Mencoba dataset alternatif...")
        data, dataset_name = try_alternative_seisbench(use_backup=True)

    if data is not None:
        metadata = data.metadata
        logger.info(f"\nDataset '{dataset_name}' berhasil diakses!")
        logger.info(f"Total rekaman: {len(metadata):,}")
        logger.info(f"Kolom: {list(metadata.columns)}")

        # Cari kolom label
        for col in metadata.columns:
            if any(x in col.lower() for x in ["category", "label", "type"]):
                logger.info(f"Kolom label '{col}':\n{metadata[col].value_counts()}")
                break
    else:
        logger.warning("\nSemua sumber SeisBench tidak tersedia saat ini.")
        logger.warning("Kemungkinan penyebab:")
        logger.warning("  1. Server SeisBench sedang down / maintenance")
        logger.warning("  2. Firewall/proxy jaringan memblokir koneksi")
        logger.warning("  3. Dataset belum tersedia di server yang dikonfigurasi")
        logger.warning("\nAlternatif yang tersedia:")
        logger.warning("  A. Gunakan mode synthetic untuk development:")
        logger.warning("     python src/data_loader.py --phase synthetic --n 5000")
        logger.warning("  B. Download STEAD manual dari:")
        logger.warning("     https://github.com/smousavi05/STEAD")
        logger.warning("     Lalu letakkan di data/raw/stead/ dan set source: local di config.yaml")


def phase_download(config: dict) -> None:
    """Download metadata dan buat subset."""
    logger.info("=" * 60)
    logger.info("FASE DOWNLOAD — Mengakses dan membuat subset dataset")
    logger.info("=" * 60)

    if not check_seisbench():
        return

    subset_cfg = config["dataset"]["subset"]
    split_cfg = config["dataset"]["split"]
    metadata_path = config["preprocessing"]["metadata_file"]

    data = None
    dataset_name = "STEAD"
    eq_label = "earthquake_local"
    noise_label = "noise"

    # Coba semua strategi SeisBench
    for use_backup in [False, True]:
        data = try_seisbench_dataset("STEAD", use_backup=use_backup)
        if data: break

    if data is None:
        logger.info("Mencoba dataset alternatif via SeisBench...")
        data, dataset_name = try_alternative_seisbench(use_backup=True)

    if data is not None:
        metadata = data.metadata.copy()

        # Sesuaikan label berdasarkan dataset yang berhasil diakses
        cat_col = next((c for c in metadata.columns
                        if "category" in c.lower() or "label" in c.lower() or "type" in c.lower()),
                       None)
        if cat_col and cat_col != "trace_category":
            metadata = metadata.rename(columns={cat_col: "trace_category"})

        # Deteksi label values
        if "trace_category" in metadata.columns:
            cats = metadata["trace_category"].dropna().unique()
            logger.info(f"Kategori yang ditemukan: {cats}")
            eq_candidates = [c for c in cats if "eq" in str(c).lower() or "earth" in str(c).lower() or "local" in str(c).lower()]
            noise_candidates = [c for c in cats if "noise" in str(c).lower() or "non" in str(c).lower()]
            if eq_candidates: eq_label = eq_candidates[0]
            if noise_candidates: noise_label = noise_candidates[0]
            logger.info(f"Label earthquake: '{eq_label}', noise: '{noise_label}'")

        if "station_network_code" not in metadata.columns:
            for c in metadata.columns:
                if "network" in c.lower():
                    metadata = metadata.rename(columns={c: "station_network_code"})
                    break

        subset = create_balanced_subset(
            metadata,
            n_earthquake=subset_cfg["n_earthquake"],
            n_noise=subset_cfg["n_noise"],
            min_snr_db=subset_cfg["min_snr_db"],
            seed=subset_cfg["seed"],
            eq_label=eq_label,
            noise_label=noise_label
        )
        subset = split_by_network(subset,
                                  train=split_cfg["train"],
                                  val=split_cfg["val"],
                                  test=split_cfg["test"],
                                  seed=subset_cfg["seed"])
        save_subset_metadata(subset, metadata_path)

        logger.info("Download metadata selesai.")
        logger.info("Selanjutnya: python src/preprocessing.py --config configs/config.yaml")

    else:
        logger.error("Tidak bisa mengakses dataset manapun via SeisBench.")
        logger.info("Gunakan mode synthetic sebagai alternatif:")
        logger.info("  python src/data_loader.py --config configs/config.yaml --phase synthetic")


def phase_synthetic(config: dict, n_samples: int = 5000) -> None:
    """Generate dataset sintetis untuk development pipeline."""
    logger.info("=" * 60)
    logger.info("FASE SYNTHETIC — Membuat data sintetis untuk development")
    logger.info("=" * 60)

    n_eq = int(n_samples * 0.833)
    n_noise = n_samples - n_eq

    wf_cfg = config["dataset"]["waveform"]
    processed_dir = config["preprocessing"]["processed_dir"]

    generate_synthetic_dataset(
        n_earthquake=n_eq,
        n_noise=n_noise,
        sampling_rate=wf_cfg["sampling_rate"],
        n_samples=wf_cfg["n_samples"],
        seed=config["dataset"]["subset"]["seed"],
        output_dir=processed_dir
    )

    logger.info("\nData sintetis selesai dibuat.")
    logger.info("Anda dapat langsung lanjut training:")
    logger.info("  python src/train.py --config configs/config.yaml --model cnn_spectrogram")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Data loader — strategi multi-source")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--phase", choices=["inspect", "download", "synthetic"],
                        default="inspect",
                        help="inspect: cek ketersediaan; download: buat subset; synthetic: buat data sintetis")
    parser.add_argument("--n", type=int, default=5000,
                        help="Jumlah rekaman sintetis (hanya untuk --phase synthetic)")
    args = parser.parse_args()

    os.chdir(Path(__file__).parent.parent)
    config = load_config(args.config)

    if args.phase == "inspect":
        phase_inspect(config)
    elif args.phase == "download":
        phase_download(config)
    elif args.phase == "synthetic":
        phase_synthetic(config, n_samples=args.n)


if __name__ == "__main__":
    main()
