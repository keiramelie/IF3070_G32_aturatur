# Ship Loading Optimizer — IF3070

Repository tugas besar **IF3070 Dasar Inteligensi Artifisial** untuk optimasi penempatan kendaraan pada dek kapal menggunakan **Hill Climbing (HC)**, **Simulated Annealing (SA)**, dan **Genetic Algorithm (GA)**.

Sistem mencari kombinasi kendaraan dan penempatannya pada grid kapal untuk **memaksimalkan total Shipping Fee**, dengan tetap memenuhi batas dek, larangan overlap, dan kapasitas berat maksimum. Kendaraan boleh tidak dimuat jika tidak dapat atau tidak dipilih untuk ditempatkan. Sistem juga menyediakan **ETA-Weighted Objective** sebagai pengembangan opsional, yaitu skor yang memprioritaskan kendaraan dengan ETA lebih kecil.

## Fitur utama

- **Algoritma optimasi:** Steepest-Ascent Hill Climbing, Hill Climbing dengan Sideways Move, Stochastic Hill Climbing, Random-Restart Hill Climbing, Simulated Annealing, dan Genetic Algorithm.
- **Objective function:** total Shipping Fee (utama) dan ETA-Weighted Shipping Fee (opsional). Skor ETA-weighted **bukan** nilai pendapatan aktual.
- **Dashboard Streamlit:** generate initial state berdasarkan seed, jalankan algoritma, lihat initial/best ship layout, dan bandingkan hasil beberapa algoritma.
- **Visualisasi:** layout kendaraan pada dek kapal dan grafik interaktif perkembangan objective/fitness serta diagnostik SA.
- **Eksperimen:** uji berulang HC/SA dan studi parameter GA (18 run), disertai output serta analisis perbandingan.

## Dataset

Dashboard menggunakan dataset sintetis di `data/synthetic_10x20/`:

- `ship.csv` — ukuran dek **10 × 20** grid dan kapasitas berat maksimum **500**.
- `vehicles.csv` — **30 kendaraan**, masing-masing memiliki ID, dimensi, Shipping Fee, berat, dan ETA.

**Data atribut kendaraan dibaca dari CSV**, bukan dihasilkan ulang setiap kali tombol ditekan. Fungsi *Generate Initial State* mengacak kendaraan yang dimuat, koordinat, serta orientasinya secara feasible; seed memungkinkan kondisi awal direproduksi. Algoritma pada tab Comparison menggunakan initial state yang sama.

## Persyaratan

- Python **3.10 atau lebih baru**
- Git
- Paket Python dalam `requirements.txt` (termasuk dependensi Streamlit, Matplotlib, Pandas, dan Plotly sesuai versi repository)

## Setup dan menjalankan program

### 1. Clone repository

```bash
git clone https://github.com/keiramelie/IF3070_G32_aturatur.git
cd IF3070_G32_aturatur
```

> Repository dapat bersifat **private**; pastikan akun GitHub memiliki akses.

### 2. Buat dan aktifkan virtual environment

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (PowerShell)**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. Jalankan dashboard Streamlit

```bash
python -m streamlit run src/app.py
```

Buka alamat lokal yang ditampilkan terminal (umumnya `http://localhost:8501`). Di sidebar, atur **Random Seed**, **Load Probability**, **Objective Function**, dan batas iterasi/generasi. Gunakan tab **Single Optimization** untuk satu algoritma atau **Algorithm Comparison** untuk membandingkan algoritma terpilih.

## Menjalankan eksperimen

Jalankan semua perintah dari **root repository**, dengan virtual environment aktif.

### Hill Climbing dan Simulated Annealing

```bash
python -m experiments.run_experiments --dataset-dir data/synthetic_10x20
```

Untuk menjalankan varian HC tambahan:

```bash
python -m experiments.run_experiments --dataset-dir data/synthetic_10x20 --bonus
```

Untuk eksperimen objective alternatif:

```bash
python -m experiments.run_experiments --dataset-dir data/synthetic_10x20 --objective eta_weighted
```

Pengaturan trial dan algoritma utama ditetapkan dalam `config.py`. Hasil utama disimpan di `output/` (misalnya `summary.csv`, `history.csv`, `sa_metrics.csv`, dan `states.json`).

### Genetic Algorithm — studi parameter 18 run

Contoh pemanggilan untuk dua studi parameter, masing-masing 3 variasi × 3 seed:

```bash
python -m experiments.run_ga_experiments \
  --dataset-dir data/synthetic_10x20 \
  --output-dir output/ga/readme_validation \
  --generation-values 50 100 150 \
  --fixed-population-size 40 \
  --population-values 20 40 60 \
  --fixed-generations 100 \
  --seeds 42 43 44 \
  --mutation-rate 0.1
```

Perintah tersebut menghasilkan **18 run**: 9 run variasi jumlah generasi dan 9 run variasi ukuran populasi. Gunakan direktori output baru bila ingin mengulang eksperimen tanpa menimpa hasil tersimpan.

Untuk membuat grafik analisis dari output GA:

```bash
python -m experiments.plot_results \
  --ga-dir output/ga/readme_validation \
  --local-dir output \
  --output-dir output/analysis/readme_validation
```

> `--local-dir output` membutuhkan hasil eksperimen HC/SA yang sesuai sudah tersedia. Hasil eksperimen yang disertakan bersama repository dapat berada di subfolder output lain.

## Testing

Jalankan seluruh unit dan integration test:

```bash
python -m pytest tests/ -q
```

Pengujian mencakup model, constraints, objective, initial state, neighbor, algoritma, serta runner/visualisasi eksperimen. Validitas state memastikan penempatan tidak overlap, tidak melewati batas dek, dan tidak melampaui kapasitas berat.

## Struktur repository

```text
IF3070_G32_aturatur/
├── data/
│   └── synthetic_10x20/           # ship.csv dan vehicles.csv
├── src/
│   ├── core/                      # Model, constraints, objective, initializer, neighbor
│   ├── algorithms/                # HC, SA, GA, dan alternative objectives
│   ├── visualization/             # Ship layout serta grafik
│   └── app.py                     # Dashboard Streamlit
├── experiments/                   # Runner eksperimen dan analisis hasil
├── tests/                         # Unit dan integration tests
├── output/                        # CSV, JSON, dan gambar hasil eksperimen
├── docs/                          # Dokumentasi pengembangan dan laporan
├── config.py                      # Konfigurasi eksperimen
├── requirements.txt               # Dependensi Python
└── README.md
```

## Pembagian tugas anggota kelompok

| Anggota | Tanggung jawab utama | Kontribusi |
|---|---|---|
| **Keira Amelie H. - 18224003** | **Core, visualisasi, integrasi** | Merancang shared core (`models`, `constraints`, `objective`, `grid`, `initializer`, `neighbor`, `results`), mengembangkan dashboard Streamlit dan visualisasi state/grafik, mengintegrasikan PR, pengujian integrasi, serta dokumentasi repository. |
| **Firanti N. Fasya - 18224119** | **Hill Climbing dan Simulated Annealing** | Implementasi Steepest-Ascent HC beserta varian bonus, SA dan random-move sampler, eksperimen HC/SA, pengolahan hasil, serta kontribusi visualisasi eksperimen. |
| **Ellaine Juvina - 18224071** | **Genetic Algorithm dan evaluasi** | Implementasi GA (seleksi, crossover, mutasi, repair), eksperimen 18 run, visualisasi hasil GA, statistik, serta analisis perbandingan algoritma. |

Seluruh anggota berkontribusi dalam review, pengujian, penyusunan laporan, dan finalisasi tugas.

## Catatan

- **Shipping Fee** adalah objective wajib; **ETA-Weighted** adalah objective tambahan untuk eksplorasi prioritas pengiriman.
- Nilai objective, runtime, dan jumlah iterasi/generasi perlu dipahami bersama: satu iterasi HC, satu iterasi SA, dan satu generasi GA memiliki biaya evaluasi yang berbeda.
- Kode pengembangan dikerjakan melalui feature branch dan Pull Request sebelum digabungkan ke `main`.
- Laporan akhir serta GitHub Release/tag disiapkan sesuai ketentuan pengumpulan mata kuliah.
