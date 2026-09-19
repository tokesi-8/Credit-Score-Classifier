# Credit Score Prediction

Repositori ini memuat proyek end-to-end Machine Learning yang dirancang untuk mengklasifikasikan skor kredit (credit score) nasabah ke dalam kategori Poor, Standard, atau Good berdasarkan data demografi, finansial, dan perilaku pembayaran. Sistem ini telah di-deploy sebagai aplikasi web menggunakan Streamlit untuk prediksi secara real-time.

# Gambaran Project

Sistem ini dibangun untuk mempercepat dan menstandardisasi penilaian risiko kredit. Dengan memprediksi profil risiko pelanggan secara otomatis, sistem ini memungkinkan analis atau institusi keuangan untuk mengambil keputusan kredit dengan lebih cepat dan objektif, serta membantu memberikan indikasi awal untuk profil pendaftar berisiko tinggi. Berdasarkan hasil evaluasi metrik Macro F1-Score pada tahapan Eksplorasi Data (EDA), model klasifikasi terbaik yang diimplementasikan dalam sistem ini adalah LightGBM.

# Arsitektur Aplikasi
Proyek ini mengadopsi arsitektur pipeline end-to-end Machine Learning:
- Data Ingestion & Preprocessing — Memuat data mentah, membersihkan data yang tidak valid, menangani missing values, dan melakukan feature engineering untuk menyesuaikan input dengan skema yang dibutuhkan oleh model.
- Machine Learning Model — Script modular digunakan untuk melakukan pelacakan (tracking) menggunakan MLflow. Model dilatih, dievaluasi, dan model terbaik (LightGBM) diserialisasi ke dalam bentuk .pkl agar dapat digunakan ulang di lingkungan production.
- Aplikasi Frontend (Streamlit) — Antarmuka web interaktif. Pengguna memasukkan data nasabah melalui form (demografi, informasi finansial, kartu kredit, dan pinjaman). Karena model telah menampung konfigurasi pelatihannya, aplikasi langsung memuat artefak model dan memproses inferensi (prediksi) tanpa memerlukan layanan backend terpisah.
- 
## Struktur Repositori

```text
Credit-Score-Classifier/
├── EksplorasiDataDanModelling.ipynb  # Notebook proses EDA, komparasi model, dan hyperparameter tuning
├── data_ingestion.py                 # Script untuk memuat dan menyiapkan data mentah
├── preprocessing.py                   # Script pembersihan data dan feature engineering
├── training.py                        # Script pelatihan model ML
├── evaluation.py                      # Script evaluasi metrik performa model
├── inference.py                       # Script pengujian inferensi model
├── pipeline.py                        # Orchestrator alur end-to-end (ingestion hingga evaluation)
├── app_streamlit.py                   # Script utama aplikasi antarmuka Streamlit
├── models/                            # Folder penyimpanan artefak model
├── mlruns/                            # Folder MLflow tracking untuk metrik dan eksperimen
└── requirements.txt                   # Daftar dependensi library Python

# Menjalankan Aplikasi (Deployment Lokal)
Aplikasi inferensi dapat berjalan di mesin lokal Anda. Pastikan artefak model (models/model_lightgbm.pkl) sudah ada sebelum menjalankan aplikasi.
"streamlit run app_streamlit.py"
Akses aplikasi melalui browser pada alamat http://localhost:8501.

# Cara Kerja Prediksi
Pada antarmuka Streamlit, pengguna akan diarahkan untuk mengisi metrik nasabah yang dibagi menjadi beberapa kategori form: Pendapatan & Demografi, Akun Bank & Kartu Kredit, serta Pinjaman & Riwayat Pembayaran.
Saat form di-submit, fungsi predict() akan mengonversi parameter input pengguna menjadi DataFrame, mencocokkan skema dengan format saat training, lalu menjalankan model.predict() beserta probabilitas kelasnya (predict_proba()). Hasil akhirnya (skor kredit Poor, Standard, atau Good) langsung ditampilkan di layar pengguna.


