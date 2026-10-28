# Earthquake Detection - Deep Learning
> Deteksi Kejadian Gempa Bumi Menggunakan Deep Learning Berbasis Analisis Sinyal Seismik  
> Tugas Akhir Mata Kuliah Machine Learning

---

## Deskripsi Proyek

Proyek ini membangun sistem deteksi otomatis kejadian gempa bumi menggunakan deep learning yang menerima potongan rekaman waveform seismik dan mengklasifikasikannya menjadi:
- **Earthquake**: sinyal yang berkaitan dengan kejadian gempa bumi
- **Noise**: sinyal yang tidak memiliki kejadian gempa yang teridentifikasi

**Catatan Penting:**
- Sistem ini adalah alat deteksi dari sinyal yang telah direkam, **bukan** sistem prediksi atau peringatan dini
- Dataset training menggunakan data global (STEAD); penerapan di Indonesia memerlukan validasi lokal
- Aplikasi ini bukan sistem resmi BMKG atau lembaga seismologi manapun

---

## Struktur Proyek

```
earthquake-detection/
├── app.py                    # Aplikasi Streamlit
├── requirements.txt          # Dependensi Python
├── README.md                 # Dokumentasi ini
├── configs/
│   └── config.yaml           # Konfigurasi eksperimen
├── data/
│   ├── raw/                  # Data mentah (tidak di-commit ke Git)
│   └── processed/            # Data terproses
├── notebooks/
│   ├── 01_eda.ipynb          # Exploratory Data Analysis
│   └── 02_experiments.ipynb  # Eksperimen model
├── src/
│   ├── data_loader.py        # Modul loading dataset
│   ├── preprocessing.py      # Pipeline preprocessing waveform
│   ├── spectrogram.py        # Pembuatan spektrogram
│   ├── dataset.py            # PyTorch Dataset class
│   ├── models.py             # Definisi arsitektur model
│   ├── train.py              # Script training
│   ├── evaluate.py           # Script evaluasi
│   └── inference.py          # Modul inferensi
├── models/                   # Model tersimpan (.pth)
├── reports/
│   ├── dataset_audit.md      # Laporan audit dataset
│   └── evaluation_report.md  # Laporan hasil evaluasi
└── tests/                    # Unit tests
```

---

## Dataset

**Dataset Primer:** STEAD (STanford EArthquake Dataset)
- **Sumber:** https://github.com/smousavi05/STEAD
- **Akses:** via SeisBench Python library
- **Lisensi:** Open-access untuk penelitian (IEEE Access)
- **Format:** HDF5 + CSV metadata
- **Sampling Rate:** 100 Hz, durasi 60 detik per trace

**Sitasi Wajib:**
```
Mousavi, S. M., Sheng, Y., Zhu, W., & Beroza, G. C. (2019).
STanford EArthquake Dataset (STEAD): A Global Data Set of Seismic Signals for AI.
IEEE Access, 7, 179464-179476. doi:10.1109/ACCESS.2019.2947848
```

---

## Instalasi

### 1. Clone Repository
```bash
git clone <repo-url>
cd earthquake-detection
```

### 2. Buat Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. Install Dependensi
```bash
pip install -r requirements.txt
```

---

## Cara Menjalankan

### Download dan Proses Dataset
```bash
python src/data_loader.py --config configs/config.yaml --phase download
python src/preprocessing.py --config configs/config.yaml
```

### Training Model
```bash
# Training CNN berbasis Spektrogram
python src/train.py --config configs/config.yaml --model cnn_spectrogram

# Training 1D CNN Baseline
python src/train.py --config configs/config.yaml --model cnn_1d
```

### Evaluasi Model
```bash
python src/evaluate.py --config configs/config.yaml --model-path models/best_model.pth
```

### Jalankan Aplikasi Web
```bash
streamlit run app.py
```

---

## Teknologi

| Komponen | Teknologi |
|---------|-----------|
| Deep Learning | PyTorch |
| Pemrosesan Seismik | ObsPy, SeisBench |
| Komputasi Numerik | NumPy, SciPy |
| Manajemen Data | Pandas, h5py |
| Visualisasi | Matplotlib, Plotly |
| Machine Learning | Scikit-learn |
| Web App | Streamlit |

---

## Referensi

1. Mousavi, S. M., et al. (2019). STEAD: A Global Data Set of Seismic Signals for AI. *IEEE Access*, 7, 179464-179476.
2. Woollam, J., et al. (2022). SeisBench-A toolbox for machine learning in seismology. *Seismological Research Letters*, 93(3), 1695-1709.
3. Michelini, A., et al. (2021). INSTANCE-the Italian seismic dataset for machine learning. *Earth System Science Data*, 13(12), 5509-5544.

---

## Keterbatasan

1. Dataset training bersumber dari data global (STEAD), bukan data Indonesia
2. Model perlu validasi lokal sebelum penerapan pada data stasiun BMKG
3. Sistem ini hanya mendeteksi kejadian dari sinyal yang telah direkam
4. Performa dapat berbeda pada kondisi noise dan tipe batuan yang berbeda dari data training
