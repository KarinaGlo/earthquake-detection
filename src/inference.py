"""
inference.py
============
Modul inferensi untuk deteksi gempa bumi dari file waveform.

Mendukung:
- Input file miniSEED (.mseed, .miniseed) via ObsPy
- Input array NumPy langsung
- Batch inferensi

Digunakan oleh:
- Aplikasi Streamlit (app.py)
- Script evaluasi tambahan

Penggunaan:
    from inference import EarthquakeDetector
    detector = EarthquakeDetector("models/best_model.pth", config)
    result = detector.predict_from_file("waveform.mseed")
"""

import os
import sys
import logging
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Dict, Tuple, Union
import time

sys.path.insert(0, str(Path(__file__).parent))

from preprocessing import normalize_waveform, validate_waveform, pad_or_crop
from spectrogram import waveform_to_spectrogram

logger = logging.getLogger("inference")


class EarthquakeDetector:
    """
    Kelas utama untuk inferensi deteksi gempa bumi.

    Mendukung dua mode input:
    1. File miniSEED/SAC (via ObsPy)
    2. Array NumPy langsung

    CATATAN PENTING:
    Probabilitas yang dihasilkan adalah confidence score model,
    bukan probabilitas kejadian sebenarnya yang dikalibrasi secara statistik.
    Hasil harus diinterpretasikan dalam konteks penelitian, bukan sebagai
    sistem peringatan operasional.
    """

    LABEL_NAMES = {0: "Noise", 1: "Earthquake"}
    LABEL_COLORS = {0: "#2196F3", 1: "#F44336"}

    def __init__(
        self,
        model_path: str,
        config: dict,
        device: str = "auto"
    ):
        """
        Inisialisasi detektor.

        Args:
            model_path: Path ke file checkpoint model (.pth)
            config: Konfigurasi proyek
            device: 'auto', 'cpu', atau 'cuda'
        """
        self.config = config
        self.model_name = config["model"]["default"]
        self.waveform_config = config["dataset"]["waveform"]
        self.model_path = model_path

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        logger.info(f"EarthquakeDetector menggunakan device: {self.device}")

        # Mode inferensi
        self.mode = "spectrogram" if "spectrogram" in self.model_name else "waveform"

        # Load model
        self.model = self._load_model()
        logger.info(f"Detektor siap. Model: {self.model_name}, Mode: {self.mode}")

    def _load_model(self):
        """Memuat model dari checkpoint."""
        from models import load_model
        return load_model(
            self.model_path,
            self.model_name,
            self.config,
            device=str(self.device)
        )

    def is_model_loaded(self) -> bool:
        """Cek apakah model berhasil dimuat."""
        return self.model is not None

    def read_miniseed(self, file_path: str) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
        """
        Membaca file miniSEED menggunakan ObsPy.

        Args:
            file_path: Path ke file miniSEED

        Returns:
            (waveform_array, metadata_dict) atau (None, None) jika gagal
        """
        try:
            import obspy
        except ImportError:
            logger.error("ObsPy tidak terinstal. Jalankan: pip install obspy")
            return None, None

        file_path = Path(file_path)
        if not file_path.exists():
            logger.error(f"File tidak ditemukan: {file_path}")
            return None, None

        # Cek ekstensi yang didukung
        supported_extensions = {".mseed", ".miniseed", ".sac", ".seed"}
        if file_path.suffix.lower() not in supported_extensions:
            logger.warning(
                f"Ekstensi '{file_path.suffix}' tidak dikenal. "
                f"Didukung: {supported_extensions}"
            )

        try:
            stream = obspy.read(str(file_path))
        except Exception as e:
            logger.error(f"Gagal membaca file dengan ObsPy: {e}")
            return None, {"error": str(e)}

        if len(stream) == 0:
            logger.error("Stream kosong — file mungkin rusak atau kosong")
            return None, {"error": "Stream kosong"}

        # Metadata
        trace = stream[0]
        metadata = {
            "network": trace.stats.network,
            "station": trace.stats.station,
            "location": trace.stats.location,
            "channel": trace.stats.channel,
            "sampling_rate": float(trace.stats.sampling_rate),
            "npts": trace.stats.npts,
            "starttime": str(trace.stats.starttime),
            "endtime": str(trace.stats.endtime),
            "n_traces": len(stream),
            "duration_sec": float(trace.stats.npts / trace.stats.sampling_rate)
        }

        logger.info(f"File miniSEED dibaca: {file_path.name}")
        logger.info(f"  Network.Station: {metadata['network']}.{metadata['station']}")
        logger.info(f"  Sampling Rate: {metadata['sampling_rate']} Hz")
        logger.info(f"  Duration: {metadata['duration_sec']:.1f} s")
        logger.info(f"  N traces: {metadata['n_traces']}")

        # Validasi sampling rate
        target_sr = self.waveform_config["sampling_rate"]
        if metadata["sampling_rate"] != target_sr:
            logger.warning(
                f"Sampling rate berbeda: file={metadata['sampling_rate']} Hz, "
                f"model dilatih pada {target_sr} Hz"
            )
            metadata["warning_sr_mismatch"] = (
                f"Sampling rate tidak sesuai. File: {metadata['sampling_rate']} Hz, "
                f"Model: {target_sr} Hz. Resampling akan dilakukan."
            )

            # Resample semua trace
            target_sr_int = int(target_sr)
            for tr in stream:
                try:
                    tr.resample(target_sr_int)
                except Exception as e:
                    logger.warning(f"Gagal resample trace {tr.id}: {e}")

        # Susun waveform multi-channel
        channel_order = {"Z": 0, "N": 1, "E": 2, "1": 1, "2": 2}
        n_samples = self.waveform_config["n_samples"]
        waveform = np.zeros((3, n_samples), dtype=np.float32)

        for tr in stream:
            ch = tr.stats.channel[-1].upper()  # Ambil karakter terakhir (Z, N, E)
            idx = channel_order.get(ch, 0)
            data = tr.data.astype(np.float32)
            # Pad or crop ke n_samples
            if len(data) >= n_samples:
                waveform[idx] = data[:n_samples]
            else:
                waveform[idx, :len(data)] = data

        return waveform, metadata

    def validate_input(
        self,
        waveform: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> Tuple[bool, str]:
        """
        Memvalidasi input waveform sebelum inferensi.

        Args:
            waveform: Array numpy shape (channels, samples)
            metadata: Metadata opsional

        Returns:
            (is_valid, message)
        """
        target_sr = self.waveform_config["sampling_rate"]
        target_samples = self.waveform_config["n_samples"]
        target_channels = self.waveform_config["n_channels"]

        is_valid, reason = validate_waveform(
            waveform, target_sr, target_samples, target_channels, "input"
        )

        if not is_valid:
            return False, reason

        # Cek duration
        if metadata:
            duration = metadata.get("duration_sec", 0)
            min_duration = 5.0  # Minimal 5 detik untuk deteksi bermakna
            if duration < min_duration:
                return False, f"Durasi terlalu pendek: {duration:.1f} s (minimal {min_duration} s)"

            # Warning jika ada sr mismatch
            if "warning_sr_mismatch" in metadata:
                logger.warning(metadata["warning_sr_mismatch"])

        return True, "OK"

    def preprocess(self, waveform: np.ndarray) -> torch.Tensor:
        """
        Melakukan preprocessing dan konversi ke tensor PyTorch.

        Args:
            waveform: Array numpy shape (channels, samples)

        Returns:
            Tensor siap inferensi
        """
        # Pad/crop ke ukuran standar
        target_samples = self.waveform_config["n_samples"]
        waveform = pad_or_crop(waveform, target_samples)

        # Normalisasi
        norm_method = self.config["preprocessing"]["normalization"]["method"]
        waveform = normalize_waveform(waveform, method=norm_method)

        if self.mode == "spectrogram":
            spec = waveform_to_spectrogram(
                waveform,
                sampling_rate=self.waveform_config["sampling_rate"],
                config=self.config
            )
            tensor = torch.tensor(spec[np.newaxis], dtype=torch.float32)  # (1, 1, F, T)
        else:
            tensor = torch.tensor(waveform[np.newaxis], dtype=torch.float32)  # (1, C, S)

        return tensor

    def predict(
        self,
        waveform: np.ndarray,
        metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Melakukan prediksi pada satu waveform.

        Args:
            waveform: Array numpy shape (channels, samples)
            metadata: Metadata rekaman (opsional)

        Returns:
            Dictionary hasil prediksi:
            {
                "label": "Earthquake" atau "Noise",
                "label_id": 1 atau 0,
                "confidence": float (0-1),
                "earthquake_probability": float (0-1),
                "noise_probability": float (0-1),
                "inference_time_ms": float,
                "is_valid": bool,
                "validation_message": str,
                "warning": str (opsional)
            }
        """
        # Validasi
        is_valid, validation_msg = self.validate_input(waveform, metadata)

        result = {
            "is_valid": is_valid,
            "validation_message": validation_msg,
        }

        if not is_valid:
            result["label"] = "Unknown"
            result["label_id"] = -1
            result["confidence"] = 0.0
            result["earthquake_probability"] = 0.0
            result["noise_probability"] = 0.0
            result["inference_time_ms"] = 0.0
            result["warning"] = (
                "Input tidak valid: " + validation_msg +
                ". Pastikan file sesuai format yang didukung."
            )
            return result

        # Preprocessing
        tensor = self.preprocess(waveform).to(self.device)

        # Inferensi
        start_time = time.perf_counter()
        self.model.eval()
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

        inference_time_ms = (time.perf_counter() - start_time) * 1000

        noise_prob = float(probs[0])
        earthquake_prob = float(probs[1])
        threshold = self.config["evaluation"]["threshold"]

        label_id = 1 if earthquake_prob >= threshold else 0
        label = self.LABEL_NAMES[label_id]
        confidence = earthquake_prob if label_id == 1 else noise_prob

        result.update({
            "label": label,
            "label_id": label_id,
            "confidence": round(confidence, 6),
            "earthquake_probability": round(earthquake_prob, 6),
            "noise_probability": round(noise_prob, 6),
            "inference_time_ms": round(inference_time_ms, 2),
            "threshold_used": threshold,
        })

        logger.info(
            f"Prediksi: {label} "
            f"(P(EQ)={earthquake_prob:.4f}, "
            f"P(Noise)={noise_prob:.4f}, "
            f"waktu={inference_time_ms:.1f}ms)"
        )

        return result

    def predict_from_file(self, file_path: str) -> Dict:
        """
        Memprediksi kejadian gempa dari file waveform.

        Args:
            file_path: Path ke file miniSEED atau format lain yang didukung ObsPy

        Returns:
            Dictionary hasil prediksi (sama dengan predict())
        """
        waveform, metadata = self.read_miniseed(file_path)

        if waveform is None:
            error_msg = metadata.get("error", "Gagal membaca file") if metadata else "Gagal membaca file"
            return {
                "is_valid": False,
                "validation_message": error_msg,
                "label": "Unknown",
                "label_id": -1,
                "confidence": 0.0,
                "earthquake_probability": 0.0,
                "noise_probability": 0.0,
                "inference_time_ms": 0.0,
                "file_metadata": metadata
            }

        result = self.predict(waveform, metadata)
        result["file_metadata"] = metadata
        return result


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
