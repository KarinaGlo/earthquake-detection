"""
tests/test_smoke.py
===================
Smoke test untuk memverifikasi pipeline dasar tanpa dataset nyata.

Menggunakan sinyal sintetis untuk memvalidasi:
- Preprocessing pipeline
- Spectrogram computation
- Model forward pass
- Inference module

Penggunaan:
    python tests/test_smoke.py
    # atau
    python -m pytest tests/test_smoke.py -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
import yaml


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "..", "configs", "config.yaml")
    with open(config_path) as f:
        return yaml.safe_load(f)


def make_synthetic_waveform(n_channels=3, n_samples=6000, seed=42):
    """Membuat waveform sintetis untuk testing."""
    np.random.seed(seed)
    t = np.linspace(0, 60, n_samples)

    waveforms = np.zeros((n_channels, n_samples), dtype=np.float32)
    for ch in range(n_channels):
        # Kombinasi sinusoidal + noise untuk mensimulasikan sinyal seismik
        waveforms[ch] = (
            np.sin(2 * np.pi * 5 * t) * np.exp(-t / 20) +
            np.random.randn(n_samples).astype(np.float32) * 0.1
        )

    return waveforms


def test_preprocessing():
    """Test pipeline normalisasi waveform."""
    from preprocessing import normalize_waveform, validate_waveform, pad_or_crop

    print("[TEST] Preprocessing pipeline...")
    waveform = make_synthetic_waveform()

    # Test validasi
    is_valid, reason = validate_waveform(waveform, 100, 6000, 3, "test_trace")
    assert is_valid, f"Validasi gagal: {reason}"

    # Test normalisasi peak
    normalized = normalize_waveform(waveform.copy(), method="peak")
    assert np.max(np.abs(normalized)) <= 1.0 + 1e-6, "Peak normalization gagal"

    # Test normalisasi std
    normalized_std = normalize_waveform(waveform.copy(), method="std")
    assert not np.any(np.isnan(normalized_std)), "Std normalization menghasilkan NaN"

    # Test pad/crop
    short_waveform = waveform[:, :3000]
    padded = pad_or_crop(short_waveform, 6000)
    assert padded.shape == (3, 6000), f"Pad gagal: shape {padded.shape}"

    long_waveform = np.zeros((3, 8000))
    cropped = pad_or_crop(long_waveform, 6000)
    assert cropped.shape == (3, 6000), f"Crop gagal: shape {cropped.shape}"

    print("  [PASS] Preprocessing: normalize, validate, pad_or_crop")


def test_spectrogram():
    """Test pembuatan spektrogram dari waveform."""
    from spectrogram import compute_spectrogram, waveform_to_spectrogram, get_spectrogram_shape

    print("[TEST] Spectrogram computation...")
    waveform = make_synthetic_waveform()

    # Test single channel
    spec, freqs, times = compute_spectrogram(
        waveform[2],  # Z component
        sampling_rate=100,
        n_fft=256, hop_length=10, win_length=100
    )
    assert spec.ndim == 2, "Spectrogram harus 2D"
    assert len(freqs) == spec.shape[0], "Jumlah frekuensi tidak sesuai"
    assert len(times) == spec.shape[1], "Jumlah time frame tidak sesuai"
    assert np.all(np.isfinite(spec)), "Spectrogram mengandung NaN/Inf"

    # Test waveform to spectrogram
    spec_4d = waveform_to_spectrogram(waveform, sampling_rate=100)
    assert spec_4d.ndim == 3, f"Expected 3D spectrogram, got {spec_4d.ndim}D"
    assert spec_4d.shape[0] == 1, "Expected 1 channel (Z-component)"

    # Test shape getter
    freq_bins, time_frames = get_spectrogram_shape(n_samples=6000, sampling_rate=100)
    assert freq_bins > 0 and time_frames > 0, "Spectrogram shape tidak valid"

    print(f"  [PASS] Spectrogram shape: (1, {freq_bins}, {time_frames})")


def test_models():
    """Test forward pass kedua model."""
    import torch
    from models import CNN1D, CNNSpectrogram, build_model

    print("[TEST] Model architectures...")
    config = load_config()

    # Test CNN1D
    model_1d = CNN1D(in_channels=3, num_classes=2, dropout=0.1)
    model_1d.eval()

    dummy_waveform = torch.randn(4, 3, 6000)  # batch=4
    with torch.no_grad():
        output_1d = model_1d(dummy_waveform)

    assert output_1d.shape == (4, 2), f"CNN1D output shape salah: {output_1d.shape}"
    proba_1d = model_1d.predict_proba(dummy_waveform)
    assert torch.allclose(proba_1d.sum(dim=1), torch.ones(4), atol=1e-5), "Probabilitas tidak sum=1"
    n_params_1d = model_1d.count_parameters()
    print(f"  [PASS] CNN1D — output: {output_1d.shape}, params: {n_params_1d:,}")

    # Test CNNSpectrogram
    model_spec = CNNSpectrogram(in_channels=1, num_classes=2, dropout=0.1)
    model_spec.eval()

    # Gunakan shape yang sebenarnya dari spectrogram module
    from spectrogram import get_spectrogram_shape
    freq_bins, time_frames = get_spectrogram_shape(n_samples=6000, sampling_rate=100)

    dummy_spec = torch.randn(4, 1, freq_bins, time_frames)
    with torch.no_grad():
        output_spec = model_spec(dummy_spec)

    assert output_spec.shape == (4, 2), f"CNNSpec output shape salah: {output_spec.shape}"
    n_params_spec = model_spec.count_parameters()
    print(f"  [PASS] CNNSpec — input: ({1},{freq_bins},{time_frames}), output: {output_spec.shape}, params: {n_params_spec:,}")


def test_evaluate_metrics():
    """Test fungsi compute_metrics."""
    from evaluate import compute_metrics
    import numpy as np

    print("[TEST] Evaluation metrics...")

    # Buat prediksi sempurna
    y_true = np.array([0, 0, 1, 1, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 1, 0])
    y_proba = np.array([0.1, 0.2, 0.9, 0.8, 0.95, 0.15])

    metrics = compute_metrics(y_true, y_pred, y_proba, prefix="test")

    assert metrics["test_accuracy"] == 1.0, "Accuracy harus 1.0 untuk perfect prediction"
    assert metrics["test_f1"] == 1.0, "F1 harus 1.0 untuk perfect prediction"
    assert metrics["test_fpr"] == 0.0, "FPR harus 0 untuk perfect prediction"
    assert metrics["test_fnr"] == 0.0, "FNR harus 0 untuk perfect prediction"
    assert "test_roc_auc" in metrics, "ROC-AUC harus dihitung"

    # Buat prediksi buruk
    y_pred_bad = 1 - y_true  # Semua salah
    metrics_bad = compute_metrics(y_true, y_pred_bad, prefix="bad")
    assert metrics_bad["bad_accuracy"] == 0.0, "Accuracy harus 0 untuk worst prediction"

    print(f"  [PASS] Metrics: acc={metrics['test_accuracy']}, f1={metrics['test_f1']:.3f}, roc_auc={metrics['test_roc_auc']:.3f}")


def test_dataset_class():
    """Test EarthquakeDataset dengan data sintetis."""
    import torch
    from torch.utils.data import DataLoader
    import tempfile
    import numpy as np

    print("[TEST] EarthquakeDataset...")
    config = load_config()

    # Buat data sintetis dan simpan sebagai NPZ
    n_samples = 50
    n_eq = 35
    n_noise = 15
    waveforms = np.random.randn(n_samples, 3, 6000).astype(np.float32)
    labels = np.array([1] * n_eq + [0] * n_noise, dtype=np.int64)

    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        tmp_path = tmp.name
        np.savez_compressed(tmp_path, waveforms=waveforms, labels=labels)

    try:
        from dataset import EarthquakeDataset

        # Test mode waveform
        ds_wf = EarthquakeDataset(tmp_path, mode="waveform", config=config, augment=False)
        assert len(ds_wf) == n_samples, f"Dataset length salah: {len(ds_wf)}"

        x, y = ds_wf[0]
        assert isinstance(x, torch.Tensor), "Output harus Tensor"
        assert x.shape == (3, 6000), f"Waveform shape salah: {x.shape}"

        # Test DataLoader
        loader = DataLoader(ds_wf, batch_size=16, shuffle=False)
        batch_x, batch_y = next(iter(loader))
        assert batch_x.shape == (16, 3, 6000), f"Batch shape salah: {batch_x.shape}"

        # Test class weights
        weights = ds_wf.get_class_weights()
        assert len(weights) == 2, "Harus ada 2 class weights"
        assert torch.all(weights > 0), "Class weights harus positif"

        # Test mode spectrogram
        ds_spec = EarthquakeDataset(tmp_path, mode="spectrogram", config=config, augment=False)
        x_spec, y_spec = ds_spec[0]
        assert x_spec.ndim == 3, f"Spectrogram harus 3D, got {x_spec.ndim}D"
        assert x_spec.shape[0] == 1, "Spectrogram channel harus 1"

        print(f"  [PASS] Dataset — waveform shape: {x.shape}, spec shape: {x_spec.shape}")
        print(f"  [PASS] Class weights: noise={weights[0]:.3f}, eq={weights[1]:.3f}")

    finally:
        os.unlink(tmp_path)


def run_all_tests():
    """Menjalankan semua smoke test."""
    print("=" * 60)
    print("SMOKE TEST — Earthquake Detection Pipeline")
    print("=" * 60)

    tests = [
        ("Preprocessing", test_preprocessing),
        ("Spectrogram", test_spectrogram),
        ("Models", test_models),
        ("Metrics", test_evaluate_metrics),
        ("Dataset", test_dataset_class),
    ]

    passed = 0
    failed = 0

    for test_name, test_fn in tests:
        try:
            test_fn()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {test_name}: Assertion error: {e}")
            failed += 1
        except ImportError as e:
            print(f"  [SKIP] {test_name}: Modul tidak tersedia: {e}")
        except Exception as e:
            print(f"  [FAIL] {test_name}: Error: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 60)
    print(f"Hasil: {passed} PASS, {failed} FAIL")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    os.chdir(os.path.join(os.path.dirname(__file__), ".."))
    success = run_all_tests()
    sys.exit(0 if success else 1)
