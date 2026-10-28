"""
dataset.py
==========
PyTorch Dataset class untuk dataset gempa bumi yang telah diproses.

Mendukung dua mode:
1. Waveform mode (1D CNN): mengembalikan raw waveform (channels, samples)
2. Spectrogram mode (2D CNN): mengembalikan spektrogram (1_or_3, freq, time)

Penggunaan:
    dataset = EarthquakeDataset("data/processed/train.npz", mode="spectrogram")
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pathlib import Path
from typing import Optional, Tuple
import logging

from spectrogram import waveform_to_spectrogram

logger = logging.getLogger("dataset")


class EarthquakeDataset(Dataset):
    """
    Dataset PyTorch untuk klasifikasi waveform seismik.

    Args:
        npz_path: Path ke file .npz hasil preprocessing
        mode: 'waveform' untuk 1D CNN, 'spectrogram' untuk 2D CNN
        config: Konfigurasi proyek (untuk parameter spektrogram)
        augment: Jika True, terapkan data augmentation (hanya untuk training)
        channel_idx: Untuk mode spectrogram, channel yang digunakan (default: 2=Z)
                     Gunakan -1 untuk multi-channel spectrogram
    """

    def __init__(
        self,
        npz_path: str,
        mode: str = "waveform",
        config: Optional[dict] = None,
        augment: bool = False,
        channel_idx: int = 2
    ):
        super().__init__()

        npz_path = Path(npz_path)
        if not npz_path.exists():
            raise FileNotFoundError(f"File NPZ tidak ditemukan: {npz_path}")

        # Load data
        data = np.load(npz_path, allow_pickle=False)
        self.waveforms = data["waveforms"]  # shape: (N, channels, samples)
        self.labels = data["labels"]        # shape: (N,)

        if mode not in ("waveform", "spectrogram"):
            raise ValueError(f"Mode harus 'waveform' atau 'spectrogram', bukan '{mode}'")

        self.mode = mode
        self.config = config
        self.augment = augment
        self.channel_idx = channel_idx

        # Infer sampling rate dari config atau default 100
        self.sampling_rate = 100
        if config:
            self.sampling_rate = config.get("dataset", {}).get("waveform", {}).get("sampling_rate", 100)

        logger.info(f"Dataset dimuat: {len(self)} rekaman, mode={mode}")
        logger.info(f"Distribusi label: earthquake={np.sum(self.labels==1)}, noise={np.sum(self.labels==0)}")
        logger.info(f"Waveform shape: {self.waveforms.shape}")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mengembalikan satu sampel.

        Returns:
            (input_tensor, label_tensor)
            - Jika mode 'waveform': input shape (channels, samples)
            - Jika mode 'spectrogram': input shape (1_or_3, freq_bins, time_frames)
        """
        waveform = self.waveforms[idx].copy()  # (channels, samples)
        label = int(self.labels[idx])

        # Data augmentation (hanya pada training)
        if self.augment:
            waveform = self._augment(waveform)

        if self.mode == "waveform":
            x = torch.tensor(waveform, dtype=torch.float32)

        else:  # spectrogram
            spec = waveform_to_spectrogram(
                waveform,
                sampling_rate=self.sampling_rate,
                config=self.config,
                channel_idx=self.channel_idx
            )
            x = torch.tensor(spec, dtype=torch.float32)

        y = torch.tensor(label, dtype=torch.long)
        return x, y

    def _augment(self, waveform: np.ndarray) -> np.ndarray:
        """
        Augmentasi data sederhana untuk waveform seismik.

        Teknik yang diterapkan:
        - Amplitudo scaling acak (±20%)
        - Tambahan Gaussian noise kecil
        - Time shift kecil (≤ 1 detik)

        CATATAN: Augmentasi ini harus diaplikasikan HANYA pada data training
        dan menggunakan parameter yang tidak bergantung pada distribusi dataset.
        """
        # Amplitude scaling
        scale = np.random.uniform(0.8, 1.2)
        waveform = waveform * scale

        # Tambah Gaussian noise kecil (SNR ~40 dB)
        noise_std = 1e-4 * np.max(np.abs(waveform))
        waveform = waveform + np.random.randn(*waveform.shape).astype(np.float32) * noise_std

        # Time shift (maksimal 1 detik = 100 sampel pada 100 Hz)
        max_shift = 100
        shift = np.random.randint(-max_shift, max_shift + 1)
        waveform = np.roll(waveform, shift, axis=1)

        return waveform

    def get_class_weights(self) -> torch.Tensor:
        """
        Menghitung class weights untuk menangani ketidakseimbangan kelas.

        Returns:
            Tensor weights (n_classes,) untuk digunakan di CrossEntropyLoss
        """
        n_total = len(self.labels)
        n_noise = np.sum(self.labels == 0)
        n_eq = np.sum(self.labels == 1)

        # Weight: inversely proportional to class frequency
        w_noise = n_total / (2 * n_noise) if n_noise > 0 else 1.0
        w_eq = n_total / (2 * n_eq) if n_eq > 0 else 1.0

        logger.info(f"Class weights — Noise: {w_noise:.3f}, Earthquake: {w_eq:.3f}")
        return torch.tensor([w_noise, w_eq], dtype=torch.float32)

    def get_weighted_sampler(self) -> WeightedRandomSampler:
        """
        Membuat WeightedRandomSampler untuk training batch yang seimbang.

        Returns:
            WeightedRandomSampler untuk digunakan di DataLoader
        """
        n_total = len(self.labels)
        n_noise = np.sum(self.labels == 0)
        n_eq = np.sum(self.labels == 1)

        w_noise = 1.0 / n_noise if n_noise > 0 else 1.0
        w_eq = 1.0 / n_eq if n_eq > 0 else 1.0

        sample_weights = np.where(self.labels == 0, w_noise, w_eq)
        sample_weights = torch.tensor(sample_weights, dtype=torch.float64)

        return WeightedRandomSampler(
            weights=sample_weights,
            num_samples=n_total,
            replacement=True
        )


def create_dataloaders(
    processed_dir: str,
    config: dict,
    mode: str = "waveform",
    use_weighted_sampler: bool = False
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Membuat DataLoader untuk training, validation, dan testing.

    Args:
        processed_dir: Direktori yang berisi file NPZ (train.npz, val.npz, test.npz)
        config: Konfigurasi proyek
        mode: 'waveform' atau 'spectrogram'
        use_weighted_sampler: Jika True, gunakan WeightedRandomSampler untuk training

    Returns:
        (train_loader, val_loader, test_loader)
    """
    processed_dir = Path(processed_dir)
    batch_size = config["training"]["batch_size"]

    # Training dataset dengan augmentasi
    train_dataset = EarthquakeDataset(
        processed_dir / "train.npz",
        mode=mode,
        config=config,
        augment=True
    )

    # Validation dan Testing tanpa augmentasi
    val_dataset = EarthquakeDataset(
        processed_dir / "val.npz",
        mode=mode,
        config=config,
        augment=False
    )

    test_dataset = EarthquakeDataset(
        processed_dir / "test.npz",
        mode=mode,
        config=config,
        augment=False
    )

    # Training loader
    if use_weighted_sampler:
        sampler = train_dataset.get_weighted_sampler()
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,  # Windows compatibility
            pin_memory=True
        )
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True
        )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    logger.info(f"DataLoader dibuat (mode={mode}):")
    logger.info(f"  Train: {len(train_dataset):,} sampel, {len(train_loader):,} batch")
    logger.info(f"  Val:   {len(val_dataset):,} sampel, {len(val_loader):,} batch")
    logger.info(f"  Test:  {len(test_dataset):,} sampel, {len(test_loader):,} batch")

    return train_loader, val_loader, test_loader
