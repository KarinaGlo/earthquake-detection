"""
evaluate.py
===========
Script evaluasi model deteksi gempa bumi.

Menghasilkan:
- Semua metrik klasifikasi (Precision, Recall, F1, ROC-AUC, PR-AUC)
- Confusion matrix
- Laporan evaluasi dalam format Markdown
- Grafik (ROC curve, PR curve, confusion matrix heatmap)

Penggunaan:
    python src/evaluate.py --config configs/config.yaml --model-path models/best_model.pth
    python src/evaluate.py --config configs/config.yaml --model-path models/best_model.pth --model-name cnn_1d
"""

import os
import sys
import argparse
import logging
import yaml
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend untuk server
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Tuple
from datetime import datetime

import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)

sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("evaluate")


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray] = None,
    prefix: str = "val"
) -> Dict:
    """
    Menghitung semua metrik evaluasi.

    Args:
        y_true: Label sebenarnya (0=noise, 1=earthquake)
        y_pred: Prediksi model (0 atau 1)
        y_proba: Probabilitas kelas earthquake (P(y=1)), untuk AUC
        prefix: Prefix untuk nama metrik

    Returns:
        Dictionary metrik
    """
    metrics = {}

    metrics[f"{prefix}_accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics[f"{prefix}_precision"] = float(
        precision_score(y_true, y_pred, zero_division=0, pos_label=1)
    )
    metrics[f"{prefix}_recall"] = float(
        recall_score(y_true, y_pred, zero_division=0, pos_label=1)
    )
    metrics[f"{prefix}_f1"] = float(
        f1_score(y_true, y_pred, zero_division=0, pos_label=1)
    )

    # False Positive Rate dan False Negative Rate
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics[f"{prefix}_fpr"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    metrics[f"{prefix}_fnr"] = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    if y_proba is not None:
        try:
            metrics[f"{prefix}_roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except ValueError:
            metrics[f"{prefix}_roc_auc"] = float("nan")

        try:
            metrics[f"{prefix}_pr_auc"] = float(average_precision_score(y_true, y_proba))
        except ValueError:
            metrics[f"{prefix}_pr_auc"] = float("nan")

    return metrics


def full_evaluation(
    model,
    loader,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluasi penuh model pada satu data loader.

    Args:
        model: Model PyTorch dalam mode eval
        loader: DataLoader
        device: Device
        threshold: Threshold probabilitas untuk klasifikasi

    Returns:
        (metrics_dict, y_true, y_pred, y_proba)
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            earthquake_probs = probs[:, 1].cpu().numpy()
            preds = (earthquake_probs >= threshold).astype(int)

            all_labels.extend(y.numpy())
            all_probs.extend(earthquake_probs)
            all_preds.extend(preds)

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_proba = np.array(all_probs)

    metrics = compute_metrics(y_true, y_pred, y_proba, prefix="test")
    return metrics, y_true, y_pred, y_proba


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    model_name: str = ""
) -> str:
    """Membuat dan menyimpan heatmap confusion matrix."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    cm_normalized = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")

    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_normalized],
        ["Absolute Count", "Normalized (Row %)"],
        ["d", ".2%"]
    ):
        im = ax.imshow(data, cmap="Blues", aspect="auto")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Noise (Pred)", "Earthquake (Pred)"])
        ax.set_yticklabels(["Noise (True)", "Earthquake (True)"])
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

        for i in range(2):
            for j in range(2):
                text = f"{data[i, j]:{fmt}}"
                ax.text(j, i, text, ha="center", va="center",
                        color="white" if data[i, j] > data.max() * 0.5 else "black",
                        fontsize=12)

    plt.tight_layout()
    output_path = output_dir / f"confusion_matrix_{model_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Confusion matrix tersimpan: {output_path}")
    return str(output_path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    roc_auc: float,
    output_dir: Path,
    model_name: str = ""
) -> str:
    """Membuat dan menyimpan ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="royalblue", lw=2, label=f"ROC Curve (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    output_path = output_dir / f"roc_curve_{model_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"ROC curve tersimpan: {output_path}")
    return str(output_path)


def plot_pr_curve(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    pr_auc: float,
    output_dir: Path,
    model_name: str = ""
) -> str:
    """Membuat dan menyimpan Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    baseline = np.sum(y_true) / len(y_true)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color="darkorange", lw=2,
            label=f"PR Curve (AP = {pr_auc:.4f})")
    ax.axhline(y=baseline, color="navy", linestyle="--",
               label=f"Baseline (prevalence = {baseline:.3f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {model_name}")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])

    output_path = output_dir / f"pr_curve_{model_name}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"PR curve tersimpan: {output_path}")
    return str(output_path)


def generate_report(
    metrics: Dict,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: Path,
    figure_paths: Dict
) -> str:
    """
    Membuat laporan evaluasi dalam format Markdown.

    Args:
        metrics: Dictionary semua metrik
        y_true: Label sebenarnya
        y_pred: Prediksi model
        model_name: Nama model
        output_dir: Direktori output
        figure_paths: Path ke file gambar yang dihasilkan

    Returns:
        Path ke file laporan
    """
    report_path = output_dir / f"evaluation_report_{model_name}.md"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cr = classification_report(
        y_true, y_pred,
        target_names=["Noise", "Earthquake"],
        digits=4
    )

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"# Laporan Evaluasi Model\n\n")
        f.write(f"**Model:** {model_name}  \n")
        f.write(f"**Tanggal:** {timestamp}  \n")
        f.write(f"**Total sampel test:** {len(y_true):,}  \n\n")
        f.write(f"---\n\n")

        f.write(f"## Metrik Utama\n\n")
        f.write(f"| Metrik | Nilai |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Accuracy | {metrics.get('test_accuracy', 0):.4f} |\n")
        f.write(f"| Precision (EQ) | {metrics.get('test_precision', 0):.4f} |\n")
        f.write(f"| Recall (EQ) | {metrics.get('test_recall', 0):.4f} |\n")
        f.write(f"| F1-Score (EQ) | {metrics.get('test_f1', 0):.4f} |\n")
        f.write(f"| ROC-AUC | {metrics.get('test_roc_auc', float('nan')):.4f} |\n")
        f.write(f"| PR-AUC | {metrics.get('test_pr_auc', float('nan')):.4f} |\n")
        f.write(f"| False Positive Rate | {metrics.get('test_fpr', 0):.4f} |\n")
        f.write(f"| False Negative Rate | {metrics.get('test_fnr', 0):.4f} |\n\n")

        f.write(f"## Confusion Matrix\n\n")
        f.write(f"| | Predicted Noise | Predicted Earthquake |\n")
        f.write(f"|---|---|---|\n")
        f.write(f"| **True Noise** | {tn} (TN) | {fp} (FP) |\n")
        f.write(f"| **True Earthquake** | {fn} (FN) | {tp} (TP) |\n\n")

        f.write(f"## Classification Report (per Kelas)\n\n")
        f.write(f"```\n{cr}\n```\n\n")

        f.write(f"## Interpretasi Dampak Error\n\n")
        f.write(f"### False Positive (FP = {fp})\n")
        f.write(f"Sinyal noise diklasifikasikan sebagai gempa.\n")
        f.write(f"**Dampak:** Alarm palsu — operator seismologi perlu memverifikasi lebih banyak rekaman.\n")
        f.write(f"Dalam konteks operasional, ini meningkatkan beban kerja verifikasi manusia.\n\n")

        f.write(f"### False Negative (FN = {fn})\n")
        f.write(f"Kejadian gempa nyata terlewat dan diklasifikasikan sebagai noise.\n")
        f.write(f"**Dampak:** Gempa tidak terdeteksi — potensi kehilangan data ilmiah.\n")
        f.write(f"Dalam sistem monitoring, FN lebih berbahaya dari FP karena gempa nyata tidak tercatat.\n\n")

        f.write(f"## Keterbatasan Model\n\n")
        f.write(f"1. Model dilatih pada dataset STEAD (data global) — generalisasi ke Indonesia belum tervalidasi.\n")
        f.write(f"2. Dataset training didominasi gempa lokal (jarak < 350 km).\n")
        f.write(f"3. Probabilitas yang dihasilkan model adalah confidence score, bukan probabilitas kejadian yang dikalibrasi.\n")
        f.write(f"4. Sistem ini tidak dapat memprediksi gempa sebelum terjadi.\n\n")

        f.write(f"## Referensi\n\n")
        f.write(f"- Mousavi et al. (2019). STEAD. IEEE Access. doi:10.1109/ACCESS.2019.2947848\n")

    logger.info(f"Laporan evaluasi tersimpan: {report_path}")
    return str(report_path)


def run_evaluation(config: dict, model_path: str, model_name: Optional[str] = None) -> None:
    """
    Menjalankan evaluasi lengkap pada test set.

    Args:
        config: Konfigurasi proyek
        model_path: Path ke checkpoint model terbaik
        model_name: Nama arsitektur (jika None, dibaca dari config)
    """
    from models import load_model
    from dataset import create_dataloaders

    if model_name is None:
        model_name = config["model"]["default"]

    logger.info("=" * 60)
    logger.info(f"EVALUASI — Model: {model_name}")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode = "spectrogram" if model_name == "cnn_spectrogram" else "waveform"

    # Load model
    model = load_model(model_path, model_name, config, device=str(device))

    # DataLoaders
    processed_dir = config["preprocessing"]["processed_dir"]
    _, _, test_loader = create_dataloaders(
        processed_dir=processed_dir,
        config=config,
        mode=mode
    )

    # Evaluasi
    threshold = config["evaluation"]["threshold"]
    metrics, y_true, y_pred, y_proba = full_evaluation(model, test_loader, device, threshold)

    # Log hasil
    logger.info(f"Hasil Evaluasi (Test Set):")
    for key, val in metrics.items():
        if isinstance(val, float):
            logger.info(f"  {key}: {val:.4f}")

    # Buat direktori output
    reports_dir = Path(config["paths"]["reports"])
    figures_dir = reports_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Buat grafik
    figure_paths = {}

    # Confusion matrix
    figure_paths["confusion_matrix"] = plot_confusion_matrix(
        y_true, y_pred, figures_dir, model_name
    )

    # ROC curve
    if "test_roc_auc" in metrics and not np.isnan(metrics["test_roc_auc"]):
        figure_paths["roc_curve"] = plot_roc_curve(
            y_true, y_proba, metrics["test_roc_auc"], figures_dir, model_name
        )

    # PR curve
    if "test_pr_auc" in metrics and not np.isnan(metrics["test_pr_auc"]):
        figure_paths["pr_curve"] = plot_pr_curve(
            y_true, y_proba, metrics["test_pr_auc"], figures_dir, model_name
        )

    # Simpan metrik sebagai JSON
    metrics_path = reports_dir / f"metrics_{model_name}.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    # Buat laporan Markdown
    generate_report(metrics, y_true, y_pred, model_name, reports_dir, figure_paths)

    logger.info("Evaluasi selesai.")


def main():
    parser = argparse.ArgumentParser(description="Evaluasi model deteksi gempa")
    parser.add_argument("--config", type=str, default="configs/config.yaml")
    parser.add_argument("--model-path", type=str, default="models/best_model.pth")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Nama arsitektur ('cnn_1d' atau 'cnn_spectrogram')")
    args = parser.parse_args()

    script_dir = Path(__file__).parent
    os.chdir(script_dir.parent)

    config = load_config(args.config)
    run_evaluation(config, args.model_path, args.model_name)


if __name__ == "__main__":
    main()
