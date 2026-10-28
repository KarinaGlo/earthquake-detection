"""
models.py
=========
Definisi arsitektur model deep learning untuk deteksi gempa bumi.

Dua pendekatan utama:
A. CNN1D — Model berbasis waveform langsung (baseline)
B. CNNSpectrogram — Model berbasis spektrogram (2D CNN)

Desain prinsip:
- Model ringan yang dapat dilatih pada CPU/GPU consumer
- Regularisasi via Dropout dan BatchNorm
- Output berupa probabilitas kelas (Softmax)

Referensi arsitektur:
- Zhu & Beroza (2019). PhaseNet. GJI. doi:10.1093/gji/ggy423
- Mousavi et al. (2020). EQTransformer. Nature Comm. doi:10.1038/s41467-020-17591-w
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

logger = logging.getLogger("models")


# ===========================================================
# MODEL A: 1D CNN Baseline (Waveform Input)
# ===========================================================

class ConvBlock1D(nn.Module):
    """Blok konvolusi 1D: Conv -> BatchNorm -> ReLU -> Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 11,
        stride: int = 1,
        padding: Optional[int] = None,
        dropout: float = 0.1
    ):
        super().__init__()
        if padding is None:
            padding = kernel_size // 2  # "same" padding

        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(F.relu(self.bn(self.conv(x))))


class CNN1D(nn.Module):
    """
    Baseline: 1D CNN untuk klasifikasi waveform seismik.

    Arsitektur:
    Input: (batch, channels, samples) — contoh: (64, 3, 6000)
    -> Conv blocks dengan MaxPool progressif
    -> Global Average Pooling
    -> Fully Connected layers
    -> Output: (batch, 2) probabilitas [noise, earthquake]

    Parameter: ~1.2M (ringan, dapat dilatih pada CPU)
    """

    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Blok konvolusi dengan channel progression: 3 -> 32 -> 64 -> 128 -> 256
        self.conv_blocks = nn.Sequential(
            ConvBlock1D(in_channels, 32, kernel_size=31, dropout=0.1),
            nn.MaxPool1d(kernel_size=4, stride=4),  # 6000 -> 1500

            ConvBlock1D(32, 64, kernel_size=11, dropout=0.1),
            nn.MaxPool1d(kernel_size=4, stride=4),   # 1500 -> 375

            ConvBlock1D(64, 128, kernel_size=7, dropout=0.2),
            nn.MaxPool1d(kernel_size=4, stride=4),   # 375 -> 93

            ConvBlock1D(128, 256, kernel_size=5, dropout=0.2),
            nn.MaxPool1d(kernel_size=4, stride=4),   # 93 -> 23

            ConvBlock1D(256, 256, kernel_size=3, dropout=0.2),
        )

        # Global Average Pooling: (batch, 256, T) -> (batch, 256)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor shape (batch, channels, samples)

        Returns:
            logits: Tensor shape (batch, num_classes)
        """
        x = self.conv_blocks(x)                   # (batch, 256, T')
        x = self.global_avg_pool(x).squeeze(-1)   # (batch, 256)
        x = self.classifier(x)                    # (batch, num_classes)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Mengembalikan probabilitas kelas (setelah Softmax)."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def count_parameters(self) -> int:
        """Menghitung jumlah parameter yang dapat dilatih."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================
# MODEL B: 2D CNN Berbasis Spektrogram
# ===========================================================

class ConvBlock2D(nn.Module):
    """Blok konvolusi 2D: Conv -> BatchNorm -> ReLU -> Dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
        dropout: float = 0.1
    ):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(F.relu(self.bn(self.conv(x))))


class ResidualBlock2D(nn.Module):
    """
    Residual block 2D dengan skip connection.
    Membantu training yang lebih dalam tanpa vanishing gradient.
    """

    def __init__(self, channels: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock2D(channels, channels, kernel_size=3, padding=1, dropout=0.0),
            ConvBlock2D(channels, channels, kernel_size=3, padding=1, dropout=dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(x + self.block(x))


class CNNSpectrogram(nn.Module):
    """
    Model CNN 2D berbasis spektrogram untuk deteksi gempa.

    Arsitektur:
    Input: (batch, 1, freq_bins, time_frames) — contoh: (64, 1, 112, 597)
    -> Conv stages dengan residual blocks
    -> Global Average Pooling 2D
    -> Fully Connected
    -> Output: (batch, 2)

    Parameter: ~2.1M
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 2,
        dropout: float = 0.3
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Stage 1: Ekstraksi fitur lokal
        self.stage1 = nn.Sequential(
            ConvBlock2D(in_channels, 32, kernel_size=7, padding=3, dropout=0.0),
            ConvBlock2D(32, 32, kernel_size=3, padding=1, dropout=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # /2
        )

        # Stage 2
        self.stage2 = nn.Sequential(
            ConvBlock2D(32, 64, kernel_size=3, padding=1, dropout=0.0),
            ResidualBlock2D(64, dropout=0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),  # /4
        )

        # Stage 3
        self.stage3 = nn.Sequential(
            ConvBlock2D(64, 128, kernel_size=3, padding=1, dropout=0.0),
            ResidualBlock2D(128, dropout=0.15),
            nn.MaxPool2d(kernel_size=2, stride=2),  # /8
        )

        # Stage 4
        self.stage4 = nn.Sequential(
            ConvBlock2D(128, 256, kernel_size=3, padding=1, dropout=0.0),
            ResidualBlock2D(256, dropout=0.2),
        )

        # Global Average Pooling: output (batch, 256)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor shape (batch, channels, freq_bins, time_frames)

        Returns:
            logits: Tensor shape (batch, num_classes)
        """
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.global_avg_pool(x).flatten(1)  # (batch, 256)
        x = self.classifier(x)
        return x

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Mengembalikan probabilitas kelas (setelah Softmax)."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ===========================================================
# Factory Function
# ===========================================================

def build_model(
    model_name: str,
    config: dict,
    spectrogram_shape: Optional[Tuple[int, int]] = None
) -> nn.Module:
    """
    Membangun model berdasarkan nama konfigurasi.

    Args:
        model_name: 'cnn_1d' atau 'cnn_spectrogram'
        config: Konfigurasi proyek
        spectrogram_shape: (freq_bins, time_frames) untuk cnn_spectrogram

    Returns:
        Model PyTorch yang siap dilatih
    """
    n_channels = config["dataset"]["waveform"]["n_channels"]
    num_classes = config["model"][model_name]["num_classes"]
    dropout = config["model"][model_name]["dropout"]

    if model_name == "cnn_1d":
        model = CNN1D(
            in_channels=n_channels,
            num_classes=num_classes,
            dropout=dropout
        )

    elif model_name == "cnn_spectrogram":
        in_channels = config["model"][model_name]["input_channels"]
        model = CNNSpectrogram(
            in_channels=in_channels,
            num_classes=num_classes,
            dropout=dropout
        )

    else:
        raise ValueError(f"Model tidak dikenal: '{model_name}'. Pilih 'cnn_1d' atau 'cnn_spectrogram'")

    n_params = model.count_parameters()
    logger.info(f"Model '{model_name}' dibuat dengan {n_params:,} parameter yang dapat dilatih")

    return model


def load_model(
    model_path: str,
    model_name: str,
    config: dict,
    device: str = "cpu"
) -> nn.Module:
    """
    Memuat model dari checkpoint file.

    Args:
        model_path: Path ke file .pth
        model_name: Nama arsitektur model
        config: Konfigurasi proyek
        device: Device target ('cpu' atau 'cuda')

    Returns:
        Model yang telah dimuat dalam mode eval
    """
    import os

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint tidak ditemukan: {model_path}")

    model = build_model(model_name, config)
    state = torch.load(model_path, map_location=device)

    # Handle checkpoint dengan atau tanpa metadata
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
        logger.info(f"Model dimuat dari checkpoint: epoch={state.get('epoch', 'N/A')}, "
                    f"val_f1={state.get('val_f1', 'N/A'):.4f}" if "val_f1" in state else
                    f"Model dimuat dari checkpoint: epoch={state.get('epoch', 'N/A')}")
    else:
        model.load_state_dict(state)
        logger.info(f"Model dimuat dari: {model_path}")

    model = model.to(device)
    model.eval()
    return model
