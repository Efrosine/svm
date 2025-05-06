# Penjelasan Struktur Project SVM Deteksi Penyakit Apel

Dokumen ini menjelaskan struktur dan fungsi dari setiap file dalam project deteksi penyakit apel menggunakan Support Vector Machine (SVM).

## Struktur Direktori Project

```
project/svm/
├── data/                           # Data dan hasil visualisasi
│   ├── apple_disease_data_before_hyperplane.pdf
│   ├── apple_disease_data_before_hyperplane.png
│   ├── apple_disease_svm_static_results.pdf
│   ├── apple_disease_svm_static_results.png
│   ├── apple_disease_svm_static_with_test.png
│   ├── iteration_history_table.png
│   └── dataset/                    # Dataset gambar apel
│       └── apple_disease/
│           ├── apple/              # Dataset utama
│           │   ├── apple scab/     # Apel dengan penyakit apple scab
│           │   ├── black rot/      # Apel dengan penyakit black rot
│           │   ├── cedar apple rust/ # Apel dengan penyakit cedar apple rust
│           │   └── healthy/        # Apel sehat
│           └── apple_test/         # Dataset untuk pengujian
│               ├── apple scab/
│               ├── black rot/
│               ├── cedar apple rust/
│               └── healthy/
├── feature_extraction_results/     # Hasil dari proses ekstraksi fitur
│   ├── run_20250430_120247/        # Hasil run spesifik dengan timestamp
│   │   ├── apple_scab_visualization.png
│   │   ├── features.csv
│   │   ├── healthy_visualization.png
│   │   ├── README.txt
│   │   └── scatter_plot.png
│   ├── run_20250430_120524/
│   ├── run_20250430_120835/
│   └── run_20250507_000008/
├── docs/                           # Dokumentasi project
│   ├── project_structure_explanation.md  # File ini
│   ├── svm_first_epoch_calculations.md   # Dokumentasi perhitungan epoch pertama (bahasa Inggris)
│   ├── svm_perhitungan_epoch_pertama.md  # Dokumentasi perhitungan epoch pertama (bahasa Indonesia)
│   ├── svm_perhitungan_epoch_pertama.docx
│   └── svm_perhitungan_epoch_pertama_new.docx
├── src/                            # Source code project
│   ├── apple_disease_detection.py  # Deteksi penyakit apel dengan data statis
│   ├── optimized_comparative_feature_extraction.py # Versi yang dioptimalkan dari perbandingan ekstraksi fitur
│   ├── svm.py                      # Implementasi algoritma SVM dari awal
│   └── __pycache__/                # Cache Python dari file yang diimport
├── tests/                          # Direktori untuk pengujian
└── requirements.txt                # Dependency packages project
```

## Penjelasan Fungsi File Source Code (src/)

### 1. svm.py

File ini berisi implementasi algoritma Support Vector Machine (SVM) dari awal tanpa menggunakan library SVM yang sudah ada seperti sklearn. Fungsi utamanya:

- Mendefinisikan kelas `SVM` dengan metode untuk pelatihan model (`fit`) dan prediksi (`predict`)
- Implementasi optimisasi gradient descent untuk menemukan hyperplane optimal
- Mengidentifikasi support vector dan margin
- Menghitung jarak margin dan fungsi keputusan
- Melacak history iterasi dan loss selama pelatihan

### 2. apple_disease_detection.py

File ini mengimplementasikan deteksi penyakit apel dengan data statis untuk demonstrasi dan validasi konsep. Fungsi utamanya:

- Membuat dataset statis dengan fitur yang terpisah jelas (apel sehat vs berpenyakit)
- Visualisasi data sebelum penerapan SVM
- Melatih model SVM pada data statis
- Menampilkan hyperlane, margin, dan support vector
- Menyimpan hasil visualisasi untuk dokumentasi
- Pengujian model dengan sampel statis baru

### 3. optimized_comparative_feature_extraction.py

Versi yang dioptimalkan dari perbandingan metode ekstraksi fitur. Fungsi utamanya:

- Implementasi perbandingan yang lebih efisien
- Benchmark untuk berbagai metode
- Analisis kecepatan dan efektivitas berbagai pendekatan
- Visualisasi perbandingan performa

## Direktori Dokumentasi (docs/)

### 1. project_structure_explanation.md

Dokumen ini, yang menjelaskan struktur dan fungsi dari setiap file dalam project.

### 2. svm_first_epoch_calculations.md

Dokumen dalam bahasa Inggris yang menjelaskan perhitungan matematika detail untuk epoch pertama dari proses pelatihan SVM. Berisi:

- Penjelasan matematika tentang algoritma SVM
- Perhitungan langkah demi langkah untuk setiap sampel dalam epoch pertama
- Ilustrasi dan contoh dari konsep margin, support vector, dan fungsi keputusan
- Hasil visual dari model setelah epoch pertama

### 3. svm_perhitungan_epoch_pertama.md

Versi bahasa Indonesia dari dokumen svm_first_epoch_calculations.md. Berisi penjelasan yang sama dalam bahasa Indonesia untuk aksesibilitas.

### 4. svm_perhitungan_epoch_pertama.docx dan svm_perhitungan_epoch_pertama_new.docx

Versi dokumen Microsoft Word dari perhitungan epoch pertama dalam bahasa Indonesia, dengan format yang mungkin lebih mudah dibaca untuk beberapa pengguna.

## Direktori Data dan Hasil

### 1. data/dataset/

Berisi dataset gambar apel dengan berbagai kondisi (sehat dan berbagai penyakit) untuk pelatihan dan pengujian model.

### 2. data/_.png, data/_.pdf

File-file visualisasi hasil yang dihasilkan oleh program, termasuk:

- Visualisasi data sebelum fitting SVM
- Hasil SVM dengan hyperplane, margin, dan support vector
- Hasil pengujian dengan sampel baru
- Tabel histori iterasi

### 3. feature_extraction_results/

Hasil yang dihasilkan dari berbagai eksperimen ekstraksi fitur, dengan direktori terpisah untuk setiap run dengan timestamp (misalnya: run_20250430_120247/). Setiap direktori berisi:

- Visualisasi fitur dari berbagai kategori penyakit apel
- File CSV berisi fitur yang diekstrak
- Scatter plot untuk visualisasi pemisahan data
- File README yang menjelaskan detail run

## Catatan Tentang Workflow Utama

Workflow utama dari project ini menggunakan kombinasi dari file-file berikut:

1. `src/svm.py` - untuk implementasi algoritma SVM
2. `src/apple_disease_detection.py` - sebagai program demonstrasi dengan data statis
3. `src/optimized_comparative_feature_extraction.py` - untuk perbandingan metode ekstraksi fitur

Project ini juga dilengkapi dengan direktori pengujian (`tests/`) untuk pengujian dan validasi komponen-komponen software.
