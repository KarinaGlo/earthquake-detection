# Dataset Audit Report
## Proyek: Deteksi Kejadian Gempa Bumi Menggunakan Deep Learning

**Tanggal Audit:** 2026-10-27  
**Auditor:** Machine Learning Engineer  
**Versi:** 1.0

---

## 1. Ringkasan Eksekutif

Audit ini mengevaluasi dua sumber dataset utama yang dapat diakses secara legal untuk pelatihan model deteksi gempa bumi berbasis deep learning:

| Dataset | Status | Ukuran | Rekomendasi |
|---------|--------|--------|-------------|
| STEAD (via SeisBench) | Tersedia | ~85 GB (full), ~5-10 GB subset | PRIMER |
| INSTANCE (via SeisBench) | Tersedia | ~40 GB | Cadangan / Validasi silang |
| EQTransformer Sample | Tersedia | < 1 GB | Demo/Uji cepat |

**Keputusan akhir:** Menggunakan subset STEAD yang diakses melalui SeisBench sebagai dataset primer untuk eksperimen awal.

---

## 2. Audit Dataset STEAD (STanford EArthquake Dataset)

### 2.1 Informasi Dasar

| Atribut | Nilai |
|---------|-------|
| **Nama Dataset** | STanford EArthquake Dataset (STEAD) |
| **Sumber Resmi** | https://github.com/smousavi05/STEAD |
| **Publikasi** | Mousavi et al., 2019, IEEE Access |
| **DOI** | 10.1109/ACCESS.2019.2947848 |
| **Tahun Rilis** | 2019 (dataset), diperbarui via SeisBench |

### 2.2 Lisensi dan Ketentuan Penggunaan

| Atribut | Nilai |
|---------|-------|
| **Lisensi** | Open-access (IEEE Access, Creative Commons-compatible) |
| **Penggunaan Komersial** | Tidak diizinkan secara eksplisit; hanya untuk penelitian |
| **Persyaratan Sitasi** | Wajib mencantumkan Mousavi et al. (2019) |
| **Autentikasi** | Tidak diperlukan — dataset publik |
| **Redistribusi** | Diizinkan untuk keperluan riset dengan atribusi |

**Referensi sitasi wajib:**

Mousavi, S. M., Sheng, Y., Zhu, W., & Beroza, G. C. (2019).
STanford EArthquake Dataset (STEAD): A Global Data Set of Seismic Signals for AI.
IEEE Access, 7, 179464-179476. https://doi.org/10.1109/ACCESS.2019.2947848

### 2.3 Cakupan Geografis dan Periode Rekaman

| Atribut | Nilai |
|---------|-------|
| **Cakupan Geografis** | Global (multi-network, multi-benua) |
| **Tipe Kejadian** | Local earthquakes (jarak epicentral < 350 km) |
| **Periode Rekaman** | Multi-tahun dari berbagai jaringan seismik global |
| **Jaringan Seismik** | IRIS, FDSN, berbagai jaringan regional global |
| **Cakupan Indonesia** | TIDAK — tidak ada subset Indonesia yang teridentifikasi |

PERINGATAN: STEAD tidak secara eksplisit mencakup data dari jaringan BMKG Indonesia. Model yang dilatih pada STEAD perlu validasi tambahan untuk penerapan di konteks Indonesia.

### 2.4 Format Waveform dan Karakteristik Sinyal

| Atribut | Nilai |
|---------|-------|
| **Format Penyimpanan** | HDF5 (.hdf5) + CSV metadata |
| **Komponen Waveform** | 3 channel: E (East-West), N (North-South), Z (Vertical) |
| **Sampling Rate** | 100 Hz (100 sampel/detik) |
| **Durasi per Rekaman** | 60 detik per trace (6,000 sampel per channel) |
| **Total Sampel per Trace** | 18,000 (3 ch x 6,000) |
| **Format Pembacaan** | HDF5 via h5py atau SeisBench API |

### 2.5 Jumlah dan Komposisi Rekaman

| Kategori | Jumlah Rekaman | Persentase |
|----------|---------------|------------|
| Earthquake (local) | ~1,030,232 | ~85.9% |
| Noise | ~168,804 | ~14.1% |
| Total | ~1,200,000 | 100% |

CATATAN Ketidakseimbangan Kelas: Dataset memiliki rasio earthquake:noise sekitar 6:1. Ketidakseimbangan ini harus ditangani saat pelatihan (mis. weighted loss, stratified sampling) tetapi jangan melakukan oversampling sebelum pemisahan data.

### 2.6 Metadata dan Label yang Tersedia

| Kolom Metadata | Deskripsi | Keterangan |
|---------------|-----------|------------|
| trace_name | Identitas unik per rekaman | Kunci utama HDF5 |
| trace_category | earthquake_local atau noise | Label utama |
| source_magnitude | Magnitudo gempa | Hanya untuk earthquake |
| source_depth_km | Kedalaman hiposenter (km) | Hanya untuk earthquake |
| source_distance_km | Jarak epicentral (km) | Hanya untuk earthquake |
| p_arrival_sample | Sampel kedatangan gelombang P | Berguna untuk windowing |
| s_arrival_sample | Sampel kedatangan gelombang S | Berguna untuk windowing |
| snr_db | Signal-to-Noise Ratio (dB) | Per channel |
| station_code | Kode stasiun seismik | Untuk pemisahan dataset |
| station_network_code | Kode jaringan | Untuk pemisahan dataset |
| station_latitude/longitude | Koordinat stasiun | Analisis geografis |
| receiver_type | Tipe sensor (HH, BH, EH, dll) | Quality filtering |

### 2.7 Ukuran Dataset dan Kebutuhan Penyimpanan

| Komponen | Ukuran |
|----------|--------|
| Dataset Lengkap (merged.hdf5 + merged.csv) | ~85 GB |
| Subset 10% (~120,000 rekaman) | ~8.5 GB |
| Subset 5% (~60,000 rekaman) | ~4.25 GB |
| Subset 1% (~12,000 rekaman) | ~0.85 GB |
| RAM untuk loading batch (batch=64) | ~2-4 GB |
| RAM untuk loading metadata penuh | ~500 MB |

**Keputusan subset eksperimen awal:**
- 60,000 rekaman (50,000 earthquake + 10,000 noise setelah balancing) diakses via SeisBench streaming
- Diprioritaskan rekaman dengan SNR >= 10 dB untuk kualitas signal yang lebih baik
- Storage dibutuhkan: ~4-5 GB untuk subset lokal

### 2.8 Potensi Bias dan Keterbatasan

| Bias/Keterbatasan | Deskripsi | Mitigasi |
|-------------------|-----------|----------|
| Bias Geografis | Didominasi data AS (California, Alaska), Eropa, dan Asia Timur | Dokumentasikan; validasi terpisah |
| Tidak Ada Data Indonesia | Tidak ada rekaman dari jaringan BMKG secara eksplisit | Nyatakan dalam batasan penelitian |
| Ketidakseimbangan Kelas | Rasio EQ:Noise ~6:1 | Weighted loss / stratified sampling |
| Bias Magnitudo | Gempa kecil (M < 2) mendominasi; gempa besar jarang | Filter magnitudo minimum |
| Bias Stasiun | Stasiun dengan banyak rekaman dapat mendominasi pelatihan | Pemisahan berbasis event/stasiun |
| Label Noise | Beberapa rekaman "noise" mungkin mengandung micro-events | Gunakan SNR threshold |

---

## 3. Audit Dataset Pendukung: INSTANCE (via SeisBench)

### 3.1 Informasi Dasar

| Atribut | Nilai |
|---------|-------|
| Nama Dataset | INSTANCE (Italian Seismic Dataset) |
| Sumber | INGV (Istituto Nazionale di Geofisica e Vulcanologia) |
| Publikasi | Michelini et al., 2021, Earth System Science Data |
| Lisensi | Creative Commons Attribution 4.0 (CC BY 4.0) |
| Autentikasi | Tidak diperlukan via SeisBench |

### 3.2 Karakteristik

| Atribut | Nilai |
|---------|-------|
| Total Rekaman | ~1,159,249 (earthquake + noise) |
| Sampling Rate | 100 Hz |
| Durasi | 120 detik per trace |
| Komponen | 3-channel (HHE, HHN, HHZ) |
| Cakupan | Italia (regional) |
| Penggunaan | Validasi silang atau eksperimen tambahan |

---

## 4. Sumber Data Indonesia: Status dan Keterbatasan

### 4.1 BMKG

| Atribut | Status |
|---------|--------|
| Ketersediaan Data Waveform Publik | Terbatas; akses katalog gempa tersedia tapi waveform memerlukan koordinasi |
| Portal | https://bmkg.go.id dan FDSN endpoint BMKG |
| Autentikasi | Diperlukan untuk data waveform real-time |
| Keputusan | TIDAK digunakan sebagai dataset training; dapat digunakan untuk demo aplikasi |

### 4.2 IRIS/EarthScope

| Atribut | Status |
|---------|--------|
| Aksesibilitas | Publik via FDSN WebService |
| Data Indonesia | Stasiun jaringan IA (BMKG) sebagian tersedia |
| Format | miniSEED via obspy |
| Keputusan | Dapat digunakan untuk uji demo aplikasi post-training |

---

## 5. Keputusan Dataset Final

**Dataset dipilih untuk Training: STEAD via SeisBench**

Kriteria subset:
- Jumlah target rekaman: 60,000
  - Earthquake: 50,000 rekaman
  - Noise: 10,000 rekaman
- SNR >= 10 dB (untuk earthquake)
- Trace category: earthquake_local atau noise
- Sampling rate: 100 Hz
- Komponen Z (vertikal) minimal tersedia

Pembagian dataset:
- Training: 70% (42,000 rekaman)
- Validation: 15% (9,000 rekaman)
- Testing: 15% (9,000 rekaman)

Strategi pemisahan: Berbasis station_network_code (bukan acak per segmen)
Storage perkiraan: ~4-5 GB lokal (HDF5 subset)

### Alasan Pemilihan

1. Label terverifikasi — Setiap rekaman memiliki kategori eksplisit dari kurasi manual
2. Metadata lengkap — P/S arrival picks, SNR, koordinat, magnitudo tersedia
3. Akses mudah — SeisBench API menghilangkan kebutuhan download manual 85 GB
4. Komunitas aktif — Banyak publikasi peer-reviewed menggunakan STEAD sebagai benchmark
5. Format standar — HDF5 + CSV kompatibel dengan pipeline ML modern

---

## 6. Rencana Implementasi Berdasarkan Kapasitas Komputer

### Spesifikasi Sistem

| Komponen | Nilai |
|----------|-------|
| Python | 3.13.12 |
| RAM | 32 GB |
| Storage Tersedia (D:) | ~432 GB |
| OS | Windows |

### Fase Eksperimen

| Fase | Dataset | Tujuan |
|------|---------|--------|
| Fase 0 | ~1,000 rekaman | Validasi pipeline (smoke test) |
| Fase 1 | 10,000 rekaman | Baseline model cepat |
| Fase 2 | 60,000 rekaman | Training model utama |
| Fase 3 | Full STEAD (opsional) | Jika hasil Fase 2 perlu peningkatan |

---

## 7. Referensi

1. Mousavi, S. M., Sheng, Y., Zhu, W., & Beroza, G. C. (2019). STanford EArthquake Dataset (STEAD): A Global Data Set of Seismic Signals for AI. IEEE Access, 7, 179464-179476. https://doi.org/10.1109/ACCESS.2019.2947848

2. Woollam, J., Munchmeyer, J., Tilmann, F., Rietbrock, A., Lange, D., Betz, T., & Soto, H. (2022). SeisBench-A toolbox for machine learning in seismology. Seismological Research Letters, 93(3), 1695-1709. https://doi.org/10.1785/0220210324

3. Michelini, A., Cianetti, S., Gaviano, S., Giunchi, C., Jozinovic, D., & Lauciani, V. (2021). INSTANCE-the Italian seismic dataset for machine learning. Earth System Science Data, 13(12), 5509-5544. https://doi.org/10.5194/essd-13-5509-2021

---

*Laporan ini dibuat sebagai bagian dari Tahap 1 proyek tugas akhir mata kuliah Machine Learning.*
*Semua informasi berdasarkan dokumentasi resmi yang dapat diverifikasi.*
