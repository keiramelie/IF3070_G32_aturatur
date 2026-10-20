# Dataset sementara untuk pengujian HC dan SA

Dataset ini dibuat untuk menguji pipeline eksperimen sebelum dataset resmi tersedia.
Angka shipping fee, berat, dimensi, dan ETA adalah data sintetis/contoh, bukan data riil.
Jangan melaporkan hasil dari dataset ini sebagai hasil final proyek.

## Struktur
- `ship.csv`: tepat satu baris berisi `width,length,max_capacity`.
- `vehicles.csv`: satu baris per kendaraan dengan kolom `vehicle_id,width,length,shipping_fee,weight,eta`.

## Saat dataset resmi tersedia
1. Buat folder baru, misalnya `data/official/`.
2. Letakkan `ship.csv` dan `vehicles.csv` dengan kolom yang sama di folder itu.
3. Jalankan `python -m experiments.run_experiments --dataset-dir data/official`.

Dengan cara ini, dataset sementara tetap tersimpan dan kode algoritma tidak perlu diubah.
