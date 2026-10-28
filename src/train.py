"""
train.py
========
Script training model deep learning untuk deteksi gempa bumi.

Fitur:
- Training loop dengan early stopping
- Logging metrik per epoch
- Checkpoint model terbaik berdasarkan val_f1
- Class weight untuk menangani ketidakseimbangan kelas
- Dukungan CPU dan CUDA

Penggunaan:
    python src/train.py --config configs/config.yaml --model cnn_spectrogram
    python src/train.py --config configs/config.yaml --model cnn_1d
"""

import os
import sys
import argparse
import logging
import yaml
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from pathlib import Path
from typing import Tuple, Dict, List

# Tambahkan src ke path
sys.path.insert(0, str(Path(__file__).parent))

from dataset import create_dataloaders
from models import build_model
from evaluate import compute_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("train")


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_device(device_str: str = "auto") -> torch.device:
    """Menentukan device (CPU/CUDA) untuk training."""
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    logger.info(f"Device training: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    return device


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int
) -> Dict:
    """
    Training satu epoch.

    Returns:
        Dictionary metrik training epoch ini
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    n_batches = len(loader)
    log_interval = max(1, n_batches // 10)  # Log setiap 10% progress

    for batch_idx, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        # Gradient clipping untuk stabilitas
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        total_loss += loss.item()

        # Kumpulkan prediksi untuk metrik
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

        if (batch_idx + 1) % log_interval == 0:
            avg_loss = total_loss / (batch_idx + 1)
            logger.debug(f"Epoch {epoch} [{batch_idx+1}/{n_batches}] Loss: {avg_loss:.4f}")

    avg_loss = total_loss / n_batches
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        prefix="train"
    )
    metrics["train_loss"] = avg_loss

    return metrics


def validate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    prefix: str = "val"
) -> Dict:
    """
    Evaluasi model pada validation/test set.

    Returns:
        Dictionary metrik validasi
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())  # P(earthquake)

    avg_loss = total_loss / len(loader)
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        prefix=prefix
    )
    metrics[f"{prefix}_loss"] = avg_loss

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: dict,
    config: dict,
    path: str,
    is_best: bool = False
) -> None:
    """Menyimpan checkpoint model."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config
    }

    torch.save(checkpoint, path)

    if is_best:
        best_path = str(Path(path).parent.parent / "best_model.pth")
        torch.save(checkpoint, best_path)
        logger.info(f"Model terbaik disimpan: {best_path} (val_f1={metrics.get('val_f1', 0):.4f})")


def run_training(config: dict, model_name: str) -> None:
    """
    Fungsi training utama.

    Args:
        config: Konfigurasi proyek
        model_name: Nama model ('cnn_1d' atau 'cnn_spectrogram')
    """
    logger.info("=" * 60)
    logger.info(f"TRAINING — Model: {model_name}")
    logger.info("=" * 60)

    train_config = config["training"]
    mode = "spectrogram" if model_name == "cnn_spectrogram" else "waveform"

    # Device
    device = get_device(config.get("inference", {}).get("device", "auto"))

    # DataLoaders
    processed_dir = config["preprocessing"]["processed_dir"]
    train_loader, val_loader, test_loader = create_dataloaders(
        processed_dir=processed_dir,
        config=config,
        mode=mode,
        use_weighted_sampler=train_config["class_weights"]["enabled"]
    )

    # Model
    model = build_model(model_name, config)
    model = model.to(device)
    logger.info(f"Parameter model: {model.count_parameters():,}")

    # Loss function dengan class weights
    if train_config["class_weights"]["enabled"]:
        # Hitung dari training dataset
        train_data = train_loader.dataset
        class_weights = train_data.get_class_weights().to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        logger.info(f"CrossEntropyLoss dengan class weights: {class_weights.cpu().numpy()}")
    else:
        criterion = nn.CrossEntropyLoss()

    # Optimizer
    lr = train_config["learning_rate"]
    wd = train_config["weight_decay"]
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    # Scheduler
    max_epochs = train_config["max_epochs"]
    scheduler_name = train_config["scheduler"]
    if scheduler_name == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-6)
    elif scheduler_name == "step":
        scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    else:
        scheduler = None

    # Early stopping
    patience = train_config["early_stopping"]["patience"]
    best_val_f1 = 0.0
    patience_counter = 0

    # Training log
    checkpoint_dir = config["training"]["checkpoint_dir"]
    training_log = []

    logger.info(f"Mulai training: {max_epochs} epoch, batch_size={train_config['batch_size']}, lr={lr}")

    for epoch in range(1, max_epochs + 1):
        epoch_start = time.time()

        # Training
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Validation
        val_metrics = validate(model, val_loader, criterion, device, prefix="val")

        # Scheduler step
        if scheduler is not None:
            scheduler.step()

        epoch_time = time.time() - epoch_start

        # Gabungkan metrik
        epoch_log = {
            "epoch": epoch,
            "time_sec": round(epoch_time, 2),
            **train_metrics,
            **val_metrics,
            "lr": optimizer.param_groups[0]["lr"]
        }
        training_log.append(epoch_log)

        # Log ringkasan
        logger.info(
            f"Epoch {epoch:3d}/{max_epochs} | "
            f"Train Loss: {train_metrics['train_loss']:.4f} | "
            f"Val Loss: {val_metrics['val_loss']:.4f} | "
            f"Val F1: {val_metrics.get('val_f1', 0):.4f} | "
            f"Val Acc: {val_metrics.get('val_accuracy', 0):.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Simpan checkpoint dan cek improvement
        val_f1 = val_metrics.get("val_f1", 0.0)
        is_best = val_f1 > best_val_f1

        if is_best:
            best_val_f1 = val_f1
            patience_counter = 0
        else:
            patience_counter += 1

        checkpoint_path = str(Path(checkpoint_dir) / f"epoch_{epoch:03d}.pth")
        save_checkpoint(
            model, optimizer, epoch, epoch_log, config,
            checkpoint_path, is_best=is_best
        )

        # Early stopping
        if patience_counter >= patience:
            logger.info(f"Early stopping dipicu setelah {epoch} epoch (patience={patience})")
            break

    # Simpan training log
    log_path = Path(checkpoint_dir).parent / "training_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2)

    logger.info(f"Training log tersimpan: {log_path}")
    logger.info(f"Training selesai. Best Val F1: {best_val_f1:.4f}")
    logger.info("Jalankan evaluasi: python src/evaluate.py --config configs/config.yaml")


def main():
    parser = argparse.ArgumentParser(description="Training model deteksi gempa")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument(
        "--model",
        type=str,
        choices=["cnn_1d", "cnn_spectrogram"],
        default=None,
        help="Nama model yang dilatih (default: dari config)"
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    os.chdir(script_dir.parent)

    config = load_config(args.config)
    model_name = args.model or config["model"]["default"]

    run_training(config, model_name)


if __name__ == "__main__":
    main()
