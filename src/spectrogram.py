"""
spectrogram.py
==============
Modul pembuatan spektrogram dari waveform seismik menggunakan STFT.

Fungsi utama:
- Mengubah waveform 1D menjadi representasi waktu-frekuensi 2D
- Menghasilkan log-mel spectrogram atau linear spectrogram
- Kompatibel dengan input CNN 2D

Referensi:
    Allen, R. (1982). Automatic earthquake recognition and timing from single traces.
    BSSA, 72(5), 1521-1535.
"""

import numpy as np
import logging
from typing import Optional, Tuple

try:
    from scipy.signal import stft as scipy_stft
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

logger = logging.getLogger("spectrogram")


def compute_spectrogram(
    waveform_channel: np.ndarray,
    sampling_rate: int = 100,
    n_fft: int = 256,
    hop_length: int = 10,
    win_length: int = 100,
    window: str = "hann",
    freq_min: float = 1.0,
    freq_max: float = 45.0,
    log_scale: bool = True,
    eps: float = 1e-8
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Menghitung STFT spectrogram dari satu channel waveform.

    Args:
        waveform_channel: Array 1D (samples,)
        sampling_rate: Sampling rate dalam Hz
        n_fft: Jumlah FFT points
        hop_length: Hop length dalam sampel
        win_length: Window length dalam sampel
        window: Jenis window function
        freq_min: Frekuensi minimum yang disertakan (Hz)
        freq_max: Frekuensi maksimum yang disertakan (Hz)
        log_scale: Jika True, kembalikan magnitude dalam skala log (dB)
        eps: Epsilon untuk stabilitas numerik log

    Returns:
        (spectrogram, frequencies, times)
        - spectrogram: Array 2D (freq_bins, time_frames)
        - frequencies: Array frekuensi (Hz)
        - times: Array waktu (detik)
    """
    if not SCIPY_AVAILABLE:
        raise ImportError("scipy tidak terinstal. Jalankan: pip install scipy")

    # Pastikan tipe float
    waveform_channel = waveform_channel.astype(np.float32)

    # Hitung STFT
    freqs, times, Zxx = scipy_stft(
        waveform_channel,
        fs=sampling_rate,
        window=window,
        nperseg=win_length,
        noverlap=win_length - hop_length,
        nfft=n_fft,
        padded=True,
        boundary='zeros'
    )

    # Magnitude spectrum
    magnitude = np.abs(Zxx)

    # Filter rentang frekuensi
    freq_mask = (freqs >= freq_min) & (freqs <= freq_max)
    magnitude = magnitude[freq_mask]
    freqs = freqs[freq_mask]

    # Konversi ke skala log (dB)
    if log_scale:
        magnitude = 20 * np.log10(magnitude + eps)
        # Min-max normalisasi ke [0, 1] untuk tampilan
        mag_min = magnitude.min()
        mag_max = magnitude.max()
        if mag_max - mag_min > eps:
            magnitude = (magnitude - mag_min) / (mag_max - mag_min)

    return magnitude.astype(np.float32), freqs, times


def waveform_to_spectrogram(
    waveform: np.ndarray,
    sampling_rate: int = 100,
    config: Optional[dict] = None,
    channel_idx: int = 2  # Default: komponen Z (vertikal)
) -> np.ndarray:
    """
    Mengubah waveform multi-channel menjadi spektrogram untuk input CNN.

    Args:
        waveform: Array shape (channels, samples)
        sampling_rate: Sampling rate Hz
        config: Konfigurasi preprocessing (opsional)
        channel_idx: Channel yang digunakan. Default 2 (Z-component)
                     Jika -1, gunakan semua channel (3-channel input)

    Returns:
        Spectrogram shape (1, freq_bins, time_frames) untuk CNN input
    """
    # Default parameter
    stft_params = {
        "n_fft": 256,
        "hop_length": 10,
        "win_length": 100,
        "window": "hann",
        "freq_min": 1.0,
        "freq_max": 45.0,
    }

    if config is not None:
        spec_config = config.get("preprocessing", {}).get("spectrogram", {})
        stft_params.update({k: v for k, v in spec_config.items() if k in stft_params})

    if channel_idx == -1:
        # Multi-channel spectrogram (3 channel -> 3-channel input)
        spectrograms = []
        for ch in range(min(waveform.shape[0], 3)):
            spec, _, _ = compute_spectrogram(
                waveform[ch],
                sampling_rate=sampling_rate,
                **stft_params
            )
            spectrograms.append(spec)
        # Stack: shape (3, freq_bins, time_frames)
        return np.stack(spectrograms, axis=0)

    else:
        # Single channel spectrogram
        ch_idx = min(channel_idx, waveform.shape[0] - 1)
        spec, _, _ = compute_spectrogram(
            waveform[ch_idx],
            sampling_rate=sampling_rate,
            **stft_params
        )
        # Shape: (1, freq_bins, time_frames)
        return spec[np.newaxis, :]


def batch_to_spectrogram(
    waveforms: np.ndarray,
    sampling_rate: int = 100,
    config: Optional[dict] = None,
    channel_idx: int = 2
) -> np.ndarray:
    """
    Mengubah batch waveform menjadi batch spektrogram.

    Args:
        waveforms: Array shape (batch, channels, samples)
        sampling_rate: Sampling rate Hz
        config: Konfigurasi preprocessing
        channel_idx: Channel yang digunakan

    Returns:
        Batch spektrogram shape (batch, 1_or_3, freq_bins, time_frames)
    """
    spectrograms = []
    for i in range(waveforms.shape[0]):
        spec = waveform_to_spectrogram(
            waveforms[i],
            sampling_rate=sampling_rate,
            config=config,
            channel_idx=channel_idx
        )
        spectrograms.append(spec)

    return np.array(spectrograms, dtype=np.float32)


def get_spectrogram_shape(
    n_samples: int = 6000,
    sampling_rate: int = 100,
    config: Optional[dict] = None
) -> Tuple[int, int]:
    """
    Menghitung dimensi output spektrogram tanpa menghitung aktual.

    Args:
        n_samples: Jumlah sampel input
        sampling_rate: Sampling rate Hz
        config: Konfigurasi

    Returns:
        (freq_bins, time_frames)
    """
    dummy = np.random.randn(1, n_samples).astype(np.float32)
    spec = waveform_to_spectrogram(dummy, sampling_rate=sampling_rate, config=config)
    return spec.shape[1], spec.shape[2]  # (freq_bins, time_frames)
