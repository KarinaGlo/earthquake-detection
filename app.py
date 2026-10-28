"""
app.py
======
Aplikasi web Streamlit untuk deteksi gempa bumi berbasis deep learning.

Fitur:
- Upload file waveform miniSEED
- Visualisasi waveform domain waktu
- Visualisasi spektrogram
- Inferensi dan tampilan hasil klasifikasi
- Riwayat analisis sesi
- Halaman dokumentasi dataset, model, evaluasi, dan keterbatasan

CATATAN PENTING:
- Aplikasi ini menggunakan model yang telah dilatih
- Probabilitas yang ditampilkan adalah confidence score model, BUKAN kepastian
- Sistem ini BUKAN sistem peringatan dini resmi
- Sistem ini TIDAK dapat memprediksi gempa sebelum terjadi

Penggunaan:
    streamlit run app.py
"""

import sys
import os
import io
import json
import time
import logging
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import yaml

# Tambahkan src ke path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# =============================================
# Konfigurasi Halaman
# =============================================

st.set_page_config(
    page_title="EarthquakeDetect — Deteksi Gempa Deep Learning",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": None,
        "Report a bug": None,
        "About": "Proyek Tugas Akhir Machine Learning — Deteksi Gempa Bumi"
    }
)

# =============================================
# CSS Custom
# =============================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-title {
    font-size: 2.4rem;
    font-weight: 700;
    color: #1a1a2e;
    margin-bottom: 0.2rem;
}

.subtitle {
    font-size: 1.1rem;
    color: #555;
    margin-bottom: 1.5rem;
}

.result-earthquake {
    background: linear-gradient(135deg, #ff6b6b, #ee0979);
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 16px;
    text-align: center;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 1rem 0;
    box-shadow: 0 4px 20px rgba(238,9,121,0.3);
}

.result-noise {
    background: linear-gradient(135deg, #43e97b, #38f9d7);
    color: white;
    padding: 1.5rem 2rem;
    border-radius: 16px;
    text-align: center;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 1rem 0;
    box-shadow: 0 4px 20px rgba(67,233,123,0.3);
}

.metric-card {
    background: #f8f9fa;
    border-left: 4px solid #667eea;
    padding: 1rem 1.2rem;
    border-radius: 0 8px 8px 0;
    margin: 0.5rem 0;
}

.warning-box {
    background: #fff3cd;
    border: 1px solid #ffc107;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    color: #856404;
}

.info-box {
    background: #d1ecf1;
    border: 1px solid #bee5eb;
    border-radius: 8px;
    padding: 1rem;
    margin: 1rem 0;
    color: #0c5460;
}

.stMetric label {
    color: #6c757d !important;
}
</style>
""", unsafe_allow_html=True)


# =============================================
# Helper Functions
# =============================================

@st.cache_resource
def load_config_cached(config_path: str = "configs/config.yaml") -> dict:
    """Memuat konfigurasi (di-cache agar tidak reload setiap render)."""
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        st.error(f"File konfigurasi tidak ditemukan: {config_path}")
        return {}


@st.cache_resource
def load_detector(model_path: str, config: dict):
    """Memuat model deteksi (di-cache untuk efisiensi)."""
    try:
        from inference import EarthquakeDetector
        detector = EarthquakeDetector(model_path, config, device="auto")
        return detector, None
    except FileNotFoundError:
        return None, f"File model tidak ditemukan: {model_path}"
    except Exception as e:
        return None, f"Gagal memuat model: {str(e)}"


def read_miniseed_obspy(uploaded_bytes: bytes, filename: str):
    """
    Membaca file miniSEED dari bytes yang diunggah.

    Returns:
        (stream, metadata) atau (None, error_msg)
    """
    try:
        import obspy

        with tempfile.NamedTemporaryFile(suffix=".mseed", delete=False) as tmp:
            tmp.write(uploaded_bytes)
            tmp_path = tmp.name

        try:
            stream = obspy.read(tmp_path)
            os.unlink(tmp_path)
        except Exception as e:
            os.unlink(tmp_path)
            return None, f"Gagal membaca file miniSEED: {str(e)}"

        if len(stream) == 0:
            return None, "Stream kosong — file mungkin rusak"

        trace = stream[0]
        metadata = {
            "filename": filename,
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

        return stream, metadata

    except ImportError:
        return None, "ObsPy tidak terinstal. Jalankan: pip install obspy"
    except Exception as e:
        return None, f"Error tidak terduga: {str(e)}"


def stream_to_waveform_array(stream, target_sr: int = 100, target_samples: int = 6000) -> np.ndarray:
    """Mengubah ObsPy Stream menjadi array numpy (3, samples)."""
    channel_order = {"Z": 0, "N": 1, "E": 2, "1": 1, "2": 2, "3": 2}
    waveform = np.zeros((3, target_samples), dtype=np.float32)

    for tr in stream:
        # Resample jika perlu
        if abs(tr.stats.sampling_rate - target_sr) > 0.5:
            try:
                tr.resample(target_sr)
            except Exception:
                pass

        ch = tr.stats.channel[-1].upper()
        idx = channel_order.get(ch, 0)
        data = tr.data.astype(np.float32)
        n = min(len(data), target_samples)
        waveform[idx, :n] = data[:n]

    return waveform


def plot_waveform_plotly(waveform: np.ndarray, sampling_rate: int = 100, title: str = "") -> go.Figure:
    """Membuat plot waveform interaktif dengan Plotly."""
    n_channels, n_samples = waveform.shape
    times = np.linspace(0, n_samples / sampling_rate, n_samples)
    ch_names = ["E (East-West)", "N (North-South)", "Z (Vertical)"]
    colors = ["#667eea", "#f093fb", "#4facfe"]

    fig = go.Figure()

    for ch in range(min(n_channels, 3)):
        fig.add_trace(go.Scatter(
            x=times,
            y=waveform[ch],
            mode="lines",
            name=ch_names[ch],
            line=dict(color=colors[ch], width=1),
            opacity=0.9
        ))

    fig.update_layout(
        title=dict(text=title, font=dict(size=16, family="Inter")),
        xaxis_title="Waktu (detik)",
        yaxis_title="Amplitudo (normalized)",
        height=350,
        margin=dict(l=60, r=20, t=50, b=50),
        legend=dict(orientation="h", y=-0.2),
        hovermode="x unified",
        plot_bgcolor="#fafafa",
        paper_bgcolor="white",
        font=dict(family="Inter")
    )
    fig.update_xaxes(gridcolor="#eeeeee", showgrid=True)
    fig.update_yaxes(gridcolor="#eeeeee", showgrid=True)

    return fig


def plot_spectrogram_plotly(waveform: np.ndarray, config: dict, sampling_rate: int = 100) -> go.Figure:
    """Membuat spektrogram interaktif dengan Plotly."""
    try:
        from spectrogram import compute_spectrogram

        spec_config = config.get("preprocessing", {}).get("spectrogram", {})
        n_fft = spec_config.get("n_fft", 256)
        hop_length = spec_config.get("hop_length", 10)
        win_length = spec_config.get("win_length", 100)
        freq_min = spec_config.get("freq_min", 1.0)
        freq_max = spec_config.get("freq_max", 45.0)

        # Gunakan komponen Z (idx=2) atau yang tersedia
        ch_idx = min(2, waveform.shape[0] - 1)
        spec, freqs, times = compute_spectrogram(
            waveform[ch_idx],
            sampling_rate=sampling_rate,
            n_fft=n_fft, hop_length=hop_length, win_length=win_length,
            freq_min=freq_min, freq_max=freq_max,
            log_scale=True
        )

        fig = go.Figure(data=go.Heatmap(
            z=spec,
            x=times,
            y=freqs,
            colorscale="Viridis",
            colorbar=dict(title="Amplitude (normalized)")
        ))

        fig.update_layout(
            title=dict(text="Spektrogram (Z-Component STFT)", font=dict(size=16, family="Inter")),
            xaxis_title="Waktu (detik)",
            yaxis_title="Frekuensi (Hz)",
            height=350,
            margin=dict(l=60, r=20, t=50, b=50),
            font=dict(family="Inter")
        )
        return fig

    except Exception as e:
        fig = go.Figure()
        fig.add_annotation(text=f"Error membuat spektrogram: {str(e)}", x=0.5, y=0.5, showarrow=False)
        return fig


def display_result(result: Dict) -> None:
    """Menampilkan hasil klasifikasi dengan styling."""
    if not result.get("is_valid", False):
        st.error(f"⚠️ Input Tidak Valid: {result.get('validation_message', 'Unknown error')}")
        return

    label = result["label"]
    eq_prob = result["earthquake_probability"]
    noise_prob = result["noise_probability"]
    conf = result["confidence"]
    infer_time = result.get("inference_time_ms", 0)

    # Label utama
    if label == "Earthquake":
        st.markdown(f'<div class="result-earthquake">🌍 GEMPA TERDETEKSI</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="result-noise">✅ NOISE (Bukan Gempa)</div>', unsafe_allow_html=True)

    # Metrik detail
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("P(Earthquake)", f"{eq_prob:.4f}", help="Probabilitas model untuk kelas Earthquake")
    col2.metric("P(Noise)", f"{noise_prob:.4f}", help="Probabilitas model untuk kelas Noise")
    col3.metric("Confidence", f"{conf:.4f}", help="Kepercayaan model untuk label yang dipilih")
    col4.metric("Waktu Inferensi", f"{infer_time:.1f} ms")

    # Gauge chart probabilitas
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=eq_prob * 100,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "P(Earthquake) %", "font": {"size": 18, "family": "Inter"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#ee0979" if eq_prob >= 0.5 else "#43e97b"},
            "steps": [
                {"range": [0, 50], "color": "#e8f5e9"},
                {"range": [50, 100], "color": "#ffebee"}
            ],
            "threshold": {
                "line": {"color": "#333", "width": 3},
                "thickness": 0.75,
                "value": 50
            }
        }
    ))
    fig.update_layout(height=300, font=dict(family="Inter"), margin=dict(t=50, b=20, l=20, r=20))
    st.plotly_chart(fig, use_container_width=True)

    # Peringatan interpretasi
    st.markdown("""
    <div class="warning-box">
    ⚠️ <strong>Catatan Interpretasi:</strong> Nilai probabilitas di atas adalah <em>confidence score</em> model, 
    bukan probabilitas kejadian yang dikalibrasi secara statistik. 
    Hasil ini merupakan output sistem penelitian dan <strong>bukan sistem peringatan dini resmi</strong>.
    </div>
    """, unsafe_allow_html=True)


# =============================================
# Halaman-halaman Aplikasi
# =============================================

def page_home():
    """Halaman utama — Deteksi Gempa."""
    st.markdown('<h1 class="main-title">🌍 EarthquakeDetect</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Deteksi Kejadian Gempa Bumi Berbasis Deep Learning dari Sinyal Seismik</p>', unsafe_allow_html=True)

    st.info(
        "**Tentang Sistem Ini:**\n\n"
        "Sistem ini mengklasifikasikan rekaman waveform seismik menjadi **Earthquake** atau **Noise** "
        "menggunakan model deep learning yang dilatih pada dataset STEAD (Stanford Earthquake Dataset).\n\n"
        "⚠️ **Sistem ini BUKAN peringatan dini dan TIDAK dapat memprediksi gempa sebelum terjadi.**"
    )

    # Load config
    config = load_config_cached()
    if not config:
        st.error("Konfigurasi tidak tersedia. Pastikan file configs/config.yaml ada.")
        return

    # Status model
    model_path = config.get("inference", {}).get("model_path", "models/best_model.pth")
    st.sidebar.markdown("### 🤖 Status Model")

    if Path(model_path).exists():
        detector, error = load_detector(model_path, config)
        if detector:
            st.sidebar.success(f"✅ Model tersedia\n`{model_path}`")
        else:
            st.sidebar.error(f"❌ Gagal memuat model: {error}")
            detector = None
    else:
        st.sidebar.warning(f"⚠️ Model belum tersedia\n`{model_path}`\n\nLatih model terlebih dahulu.")
        detector = None

    st.markdown("---")

    # Upload file
    st.markdown("### 📁 Upload File Waveform")
    st.markdown("Format yang didukung: **miniSEED** (.mseed, .miniseed)")

    uploaded_file = st.file_uploader(
        "Pilih file waveform seismik",
        type=["mseed", "miniseed"],
        help="Upload file miniSEED dari stasiun seismik. Sampling rate sebaiknya 100 Hz, durasi minimal 5 detik.",
        key="waveform_upload"
    )

    if uploaded_file is not None:
        st.markdown("---")
        st.markdown("### 📊 Analisis Waveform")

        # Baca file
        file_bytes = uploaded_file.read()

        with st.spinner("Membaca file waveform..."):
            stream, metadata = read_miniseed_obspy(file_bytes, uploaded_file.name)

        if stream is None:
            st.error(f"❌ Gagal membaca file: {metadata}")
            return

        # Tampilkan metadata rekaman
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("#### 📋 Informasi Rekaman")
            meta_display = {
                "Nama File": metadata["filename"],
                "Network.Station": f"{metadata['network']}.{metadata['station']}",
                "Channel": metadata["channel"],
                "Sampling Rate": f"{metadata['sampling_rate']} Hz",
                "Durasi": f"{metadata['duration_sec']:.1f} detik",
                "Jumlah Trace": metadata["n_traces"],
                "Start Time": metadata["starttime"]
            }
            for key, val in meta_display.items():
                st.markdown(f"**{key}:** `{val}`")

        with col2:
            # Konversi ke array
            target_sr = config["dataset"]["waveform"]["sampling_rate"]
            target_samples = config["dataset"]["waveform"]["n_samples"]
            waveform_array = stream_to_waveform_array(stream, target_sr, target_samples)

            # Validasi sampling rate
            if abs(metadata["sampling_rate"] - target_sr) > 0.5:
                st.warning(
                    f"⚠️ Sampling rate file ({metadata['sampling_rate']} Hz) "
                    f"berbeda dari yang diharapkan model ({target_sr} Hz). "
                    "Resampling otomatis dilakukan."
                )

        # Waveform plot
        st.markdown("#### 📈 Waveform (Domain Waktu)")
        waveform_fig = plot_waveform_plotly(
            waveform_array, target_sr,
            title=f"Waveform: {metadata['network']}.{metadata['station']}.{metadata['channel']}"
        )
        st.plotly_chart(waveform_fig, use_container_width=True)

        # Spectrogram
        st.markdown("#### 🔊 Spektrogram")
        with st.spinner("Menghitung spektrogram..."):
            spec_fig = plot_spectrogram_plotly(waveform_array, config, target_sr)
        st.plotly_chart(spec_fig, use_container_width=True)

        # Deteksi
        st.markdown("---")
        st.markdown("### 🔍 Deteksi Gempa Bumi")

        if detector is None:
            st.warning(
                "⚠️ Model deteksi belum tersedia. "
                "Latih model terlebih dahulu dengan menjalankan:\n"
                "```bash\n"
                "python src/train.py --config configs/config.yaml --model cnn_spectrogram\n"
                "```"
            )
            return

        if st.button("🚀 Jalankan Deteksi", key="run_detection", type="primary", use_container_width=True):
            with st.spinner("Menjalankan inferensi model..."):
                result = detector.predict(waveform_array, metadata)

            st.markdown("---")
            st.markdown("#### 📊 Hasil Klasifikasi")
            display_result(result)

            # Ringkasan preprocessing
            with st.expander("📋 Ringkasan Preprocessing & Informasi Teknis"):
                st.json({
                    "Normalization Method": config["preprocessing"]["normalization"]["method"],
                    "Target Sampling Rate": f"{target_sr} Hz",
                    "Window Duration": f"{target_samples / target_sr} detik",
                    "n_samples per channel": target_samples,
                    "n_channels": waveform_array.shape[0],
                    "Model": config["model"]["default"],
                    "Threshold": config["evaluation"]["threshold"],
                })

            # Simpan ke riwayat sesi
            if "analysis_history" not in st.session_state:
                st.session_state.analysis_history = []

            history_entry = {
                "timestamp": datetime.now().strftime("%H:%M:%S"),
                "filename": metadata["filename"],
                "station": f"{metadata['network']}.{metadata['station']}",
                "duration_sec": round(metadata["duration_sec"], 1),
                "label": result.get("label", "Unknown"),
                "earthquake_prob": round(result.get("earthquake_probability", 0), 4),
                "noise_prob": round(result.get("noise_probability", 0), 4),
                "inference_ms": result.get("inference_time_ms", 0),
            }
            st.session_state.analysis_history.append(history_entry)


def page_history():
    """Halaman riwayat analisis sesi."""
    st.title("📜 Riwayat Analisis Sesi")

    if "analysis_history" not in st.session_state or not st.session_state.analysis_history:
        st.info("Belum ada analisis yang dilakukan dalam sesi ini. Upload file waveform di halaman utama.")
        return

    history = st.session_state.analysis_history
    df_history = pd.DataFrame(history)

    st.markdown(f"**Total analisis dalam sesi ini:** {len(df_history)}")

    # Color code
    def highlight_label(val):
        if val == "Earthquake":
            return "background-color: #ffebee; color: #c62828; font-weight: bold;"
        elif val == "Noise":
            return "background-color: #e8f5e9; color: #2e7d32; font-weight: bold;"
        return ""

    styled_df = df_history.style.applymap(highlight_label, subset=["label"])
    st.dataframe(styled_df, use_container_width=True, height=400)

    # Download history
    csv_data = df_history.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Riwayat (CSV)",
        data=csv_data,
        file_name=f"history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )


def page_documentation():
    """Halaman dokumentasi proyek."""
    st.title("📚 Dokumentasi Proyek")

    tab1, tab2, tab3, tab4 = st.tabs(["Dataset", "Model", "Evaluasi", "Keterbatasan"])

    with tab1:
        st.markdown("""
        ## Dataset: STEAD (STanford EArthquake Dataset)

        **Sumber:** https://github.com/smousavi05/STEAD  
        **Diakses via:** SeisBench Python library  
        **Lisensi:** Open-access untuk penelitian akademik

        ### Komposisi Dataset

        | Kategori | Jumlah (full) | Dalam Subset Eksperimen |
        |---------|--------------|------------------------|
        | Earthquake (local) | ~1,030,232 | 50,000 |
        | Noise | ~168,804 | 10,000 |
        | **Total** | **~1,200,000** | **60,000** |

        ### Karakteristik Waveform

        - **Format:** HDF5 + CSV metadata
        - **Komponen:** 3-channel (E, N, Z)
        - **Sampling Rate:** 100 Hz
        - **Durasi:** 60 detik per trace

        ### Metadata Tersedia

        - P-wave dan S-wave arrival picks
        - Magnitudo dan kedalaman gempa
        - Koordinat stasiun
        - Signal-to-Noise Ratio (SNR) per channel
        - Kode jaringan seismik

        ### Referensi Wajib

        > Mousavi, S. M., Sheng, Y., Zhu, W., & Beroza, G. C. (2019).
        > STanford EArthquake Dataset (STEAD): A Global Data Set of Seismic Signals for AI.
        > *IEEE Access*, 7, 179464–179476. https://doi.org/10.1109/ACCESS.2019.2947848
        """)

    with tab2:
        st.markdown("""
        ## Model Deep Learning

        ### Pendekatan A: CNN Berbasis Spektrogram (utama)

        **Pipeline:**
        1. Waveform → STFT Spectrogram
        2. CNN 2D dengan Residual Blocks
        3. Global Average Pooling
        4. Fully Connected → Softmax

        **Arsitektur:**
        - Input: (1, ~112, ~597) — spektrogram Z-component
        - 4 stage conv dengan MaxPool progressif
        - Residual connections untuk stabilitas training
        - ~2.1 juta parameter

        ### Pendekatan B: 1D CNN Baseline (Waveform)

        **Pipeline:**
        1. Raw waveform (3-component)
        2. 1D CNN dengan MaxPool
        3. Global Average Pooling
        4. Fully Connected → Softmax

        **Arsitektur:**
        - Input: (3, 6000) — raw waveform
        - 5 conv blocks dengan MaxPool progressif
        - ~1.2 juta parameter

        ### Konfigurasi Training

        | Parameter | Nilai |
        |-----------|-------|
        | Optimizer | Adam |
        | Learning Rate | 0.001 |
        | Scheduler | Cosine Annealing |
        | Max Epochs | 50 |
        | Early Stopping Patience | 10 |
        | Batch Size | 64 |
        | Class Weighting | Enabled |
        """)

    with tab3:
        st.markdown("""
        ## Evaluasi Model

        ### Metrik yang Digunakan

        | Metrik | Deskripsi |
        |--------|-----------|
        | **Accuracy** | Proporsi prediksi yang benar |
        | **Precision** | Dari semua yang diprediksi gempa, berapa yang benar-benar gempa |
        | **Recall** | Dari semua gempa nyata, berapa yang berhasil terdeteksi |
        | **F1-Score** | Rata-rata harmonik Precision dan Recall |
        | **ROC-AUC** | Area di bawah kurva ROC |
        | **PR-AUC** | Area di bawah kurva Precision-Recall (penting untuk kelas tidak seimbang) |
        | **FPR** | False Positive Rate |
        | **FNR** | False Negative Rate |

        ### Dampak Error

        **False Positive (FP):** Noise diklasifikasikan sebagai gempa  
        → Alarm palsu, beban verifikasi meningkat untuk operator

        **False Negative (FN):** Gempa nyata terlewat diklasifikasikan sebagai noise  
        → Gempa tidak tercatat dalam katalog otomatis (lebih berbahaya dari FP)

        > **Catatan:** Hasil evaluasi yang ditampilkan di sini hanya tersedia  
        > setelah model dilatih dan evaluasi dijalankan:  
        > `python src/evaluate.py --config configs/config.yaml`
        """)

        # Tampilkan hasil evaluasi jika tersedia
        eval_files = list(Path("reports").glob("metrics_*.json")) if Path("reports").exists() else []
        if eval_files:
            st.markdown("### Hasil Evaluasi (dari file)")
            for eval_file in eval_files:
                with open(eval_file) as f:
                    metrics = json.load(f)
                model_nm = eval_file.stem.replace("metrics_", "")
                st.markdown(f"**Model: {model_nm}**")
                metrics_df = pd.DataFrame(
                    [(k.replace("test_", ""), f"{v:.4f}" if isinstance(v, float) else v)
                     for k, v in metrics.items()],
                    columns=["Metrik", "Nilai"]
                )
                st.dataframe(metrics_df, use_container_width=True)
        else:
            st.info("Hasil evaluasi belum tersedia. Jalankan proses training dan evaluasi terlebih dahulu.")

    with tab4:
        st.markdown("""
        ## Keterbatasan Sistem

        ### Keterbatasan Dataset
        1. **STEAD adalah dataset global** — tidak secara eksplisit mencakup data dari jaringan BMKG Indonesia
        2. Didominasi oleh gempa lokal (jarak < 350 km dari stasiun)
        3. Distribusi magnitudo: lebih banyak gempa kecil (M < 2)
        4. Tidak merepresentasikan variasi sensor dan kondisi noise di jaringan Indonesia

        ### Keterbatasan Model
        1. Model dilatih pada karakteristik sinyal tertentu — dapat menurun performa pada:
           - Data dari tipe sensor berbeda
           - Kondisi noise yang sangat berbeda dari training data
           - Gempa dengan mekanisme fokal yang tidak biasa
        2. Probabilitas model **tidak terkalibrasi** secara statistik

        ### Keterbatasan Sistem
        1. **Bukan Peringatan Dini:** Sistem mendeteksi kejadian dari rekaman yang sudah ada, bukan prediksi
        2. **Bukan Sistem Resmi:** Tidak terafiliasi dengan BMKG atau lembaga seismologi manapun
        3. **Tidak Dapat Menentukan:** Magnitudo, lokasi, kedalaman, atau waktu kejadian

        ### Syarat Penggunaan di Indonesia
        Sebelum menerapkan model ini pada data stasiun Indonesia, diperlukan:
        - Validasi pada dataset lokal Indonesia (misalnya dari jaringan BMKG)
        - Fine-tuning menggunakan data Indonesia
        - Evaluasi independen oleh ahli seismologi lokal
        """)


# =============================================
# Main App
# =============================================

def main():
    # Sidebar navigasi
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/56/White_shark.jpg/1200px-White_shark.jpg",
                    caption="", use_column_width=True) if False else None  # Placeholder untuk logo

    st.sidebar.markdown("## 🌍 EarthquakeDetect")
    st.sidebar.markdown("*Deep Learning Seismic Detection*")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigasi",
        ["🏠 Deteksi Gempa", "📜 Riwayat Sesi", "📚 Dokumentasi"],
        index=0,
        key="navigation"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ Info Proyek")
    st.sidebar.markdown("""
    **Judul:**  
    Deteksi Gempa Bumi Berbasis Deep Learning

    **Dataset:**  
    STEAD (Mousavi et al., 2019)

    **Framework:**  
    PyTorch + SeisBench
    """)

    # Routing halaman
    if page == "🏠 Deteksi Gempa":
        page_home()
    elif page == "📜 Riwayat Sesi":
        page_history()
    elif page == "📚 Dokumentasi":
        page_documentation()


if __name__ == "__main__":
    main()
