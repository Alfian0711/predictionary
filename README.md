# Aplikasi Prediksi Saham Indonesia

Aplikasi web sederhana untuk memprediksi harga saham Indonesia menggunakan Flask, yfinance, dan machine learning sederhana. Antarmuka pengguna dibangun dengan Tailwind CSS untuk tampilan yang modern dan responsif.

## Fitur

- Mengambil data historis saham Indonesia dari Yahoo Finance
- Membuat prediksi harga saham menggunakan model machine learning sederhana
- Menampilkan grafik perbandingan data historis dan prediksi
- Antarmuka yang responsif dan modern dengan Tailwind CSS
- Menampilkan prediksi tertinggi, terendah, dan potensi return

## Persyaratan

- Python 3.7+
- Flask
- yfinance
- pandas
- numpy
- scikit-learn
- matplotlib

## Instalasi

1. Buat dan aktifkan virtual environment (opsional tapi direkomendasikan):

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. Instal semua dependensi:

```bash
pip install flask yfinance pandas numpy scikit-learn matplotlib
```

3. Buat struktur folder:

```
/stock_prediction_app
    /templates
        index.html
    app.py
```

4. Salin kode dari artifact "Aplikasi Prediksi Saham Indonesia (app.py)" ke dalam file `app.py`

5. Salin kode dari artifact "Template HTML dengan Tailwind CSS" ke dalam file `templates/index.html`

## Menjalankan Aplikasi

1. Jalankan aplikasi Flask:

```bash
python app.py
```

2. Buka browser dan akses `http://127.0.0.1:5000/`

## Cara Penggunaan

1. Pilih kode saham dari dropdown menu
2. Pilih jumlah hari untuk prediksi (7, 14, 30, 60, atau 90 hari)
3. Klik tombol "Analisis & Prediksi"
4. Lihat hasil analisis yang mencakup:
   - Informasi saham terkini
   - Grafik perbandingan data historis dan prediksi
   - Tabel prediksi harga untuk periode yang dipilih
   - Ringkasan prediksi tertinggi, terendah, dan potensi return

## Catatan Penting

- Aplikasi ini menggunakan model machine learning sederhana (Linear Regression) dan hanya untuk tujuan demonstrasi.
- Prediksi saham sangat sulit dan tidak ada model yang dapat memprediksi dengan akurat secara konsisten.
- Gunakan aplikasi ini hanya untuk tujuan pendidikan dan eksperimen, bukan untuk keputusan investasi nyata.
- Kode saham di Indonesia pada Yahoo Finance umumnya diakhiri dengan `.JK`

## Meningkatkan Aplikasi

Beberapa cara untuk meningkatkan aplikasi:

1. Gunakan model machine learning yang lebih canggih (LSTM, Prophet, dll.)
2. Tambahkan lebih banyak fitur teknikal (RSI, MACD, Bollinger Bands, dll.)
3. Tambahkan analisis sentimen dari berita terkait saham
4. Implementasikan autentikasi pengguna dan penyimpanan riwayat prediksi
5. Tambahkan fitur perbandingan antar saham
6. Integrasi dengan API lain untuk data fundamental perusahaan

## Disclaimer

Aplikasi ini hanya untuk tujuan informasi dan pendidikan. Bukan merupakan rekomendasi investasi. Selalu lakukan riset Anda sendiri sebelum melakukan investasi.
