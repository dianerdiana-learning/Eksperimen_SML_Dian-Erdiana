# Eksperimen_SML_Dian-Erdiana

Repository ini berisi implementasi Kriteria 1 MSML untuk eksperimen preprocessing dataset pelatihan, conversion ke script automation, dan workflow GitHub Actions untuk Advance.

## Dataset

Dataset utama yang digunakan adalah Breast Cancer Wisconsin (Diagnostic).

- Sumber primer (publik): UCI Machine Learning Repository
  https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)
- Sumber akses pada implementasi ini: loader resmi scikit-learn (`load_breast_cancer`)
  https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

Profil singkat dataset:

- 569 sampel
- 30 fitur numerik
- Tugas klasifikasi biner dengan target `0 = malignant` dan `1 = benign`

## Struktur Penting

- `preprocessing/Eksperimen_Dian-Erdiana.ipynb` - eksperimen manual sesuai template MSML.
- `preprocessing/automate_Dian-Erdiana.py` - preprocessing otomatis yang menghasilkan dataset siap latih.
- `namadataset_raw/breast_cancer_raw.csv` - dataset mentah yang dipakai notebook dan script.
- `namadataset_preprocessing/breast_cancer_preprocessed.csv` - output dataset hasil preprocessing.
- `.github/workflows/preprocess.yml` - workflow GitHub Actions untuk menjalankan preprocessing otomatis.

## Cara Menjalankan

1. Jalankan notebook `preprocessing/Eksperimen_Dian-Erdiana.ipynb` untuk melihat data loading, EDA, preprocessing, dan baseline model.
2. Jalankan `python preprocessing/automate_Dian-Erdiana.py` untuk menghasilkan dataset preprocessing secara otomatis.
3. Jalankan workflow GitHub Actions `preprocess-dataset` melalui push ke `main` atau `workflow_dispatch` untuk menghasilkan artefak dataset terbaru.

## Dependensi

Gunakan paket di `requirements.txt` jika ingin memasang dependensi secara lokal.
