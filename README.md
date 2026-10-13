# Structural Damage Classification — Gempa Gorkha 2015

Memprediksi tingkat kerusakan bangunan (`damage_grade`: Grade 1–5) dari karakteristik struktural dan lokasi
bangunan. Metrik kompetisi: **Macro F1-Score**. Model: **LightGBM** di dalam `Pipeline` + `ColumnTransformer`
(tanpa Neural Network).

## Struktur repository

```
structural-damage-classification/
├── README.md
├── .gitignore
├── requirements.txt
├── data/                 # train.csv, test.csv, sample_submission.csv, metadata.csv (tidak di-commit)
├── notebooks/
│   ├── 01_eda.ipynb          # eksplorasi data
│   ├── 02_modeling.ipynb     # split, preprocessing, baseline (OvR/OvO), training, tuning, evaluasi
│   └── 03_submission.ipynb   # membuat & memvalidasi submission
├── scripts/
│   ├── __init__.py
│   ├── preprocess.py             # load data, CategoryCaster, ColumnTransformer
│   ├── feature_engineering.py    # BuildingFeatureEngineer (fitur turunan)
│   ├── train.py                  # Pipeline, 5-fold CV, tuning bobot kelas, simpan model
│   └── predict.py                # prediksi test -> submissions/submission.csv
├── models/               # pipeline_model_lgbm.pkl, class_weights.json, oof_proba.npy
├── submissions/          # submission.csv
└── reports/figures/      # grafik EDA dan evaluasi
```

## Cara menggunakan

1. **Install dependency**
   ```bash
   pip install -r requirements.txt
   ```
2. **Unduh data** dari halaman kompetisi Kaggle (tab *Data* → *Download All*), atau lewat API:
   ```bash
   kaggle competitions download -c nama-kompetisi
   ```
   Letakkan `train.csv`, `test.csv`, `sample_submission.csv`, `metadata.csv` di folder `data/`
   (nama seperti `train(1).csv` juga terbaca). Folder `data/*.csv` di-ignore oleh git (file > 100MB).
3. **Training** (5-fold CV + tuning bobot kelas + fit model final):
   ```bash
   python -m scripts.train
   ```
4. **Prediksi & submission**:
   ```bash
   python -m scripts.predict
   ```
   Hasil: `submissions/submission.csv` (kolom `building_id,damage_grade`; label angka 1-5 sesuai deskripsi kompetisi, bukan `Grade N`).
5. Upload `submission.csv` di tab *Submit Prediction* pada Kaggle (maks. 5 submission per hari).

## Alur analisis (workflow)

Detail langkah ada di notebook; `scripts/` berisi versi modular yang sama untuk dijalankan ulang.

1. **EDA** (`01_eda.ipynb`): ukuran data, kolom yang tidak dipakai langsung, duplikasi, missing value
   (tabel + heatmap), outlier (IQR + boxplot), distribusi numerik & kategorik, imbalance target,
   hubungan fitur-target, korelasi, lokasi geografis, dan perbandingan train vs test.
2. **Split** (`02_modeling.ipynb`): holdout 80/20 stratified (+ 5-fold CV di `scripts/train.py`).
3. **Feature engineering** (`BuildingFeatureEngineer`): luas & tinggi per lantai, slenderness, proxy volume,
   umur di-clip + flag `age_unknown`, dan potongan kode lokasi dari `building_id` (`b_vdc`, `b_ward`, `b_seq`).
4. **Preprocessing** (`ColumnTransformer`): kategorik dan kode lokasi jadi dtype `category` (kategori native
   LightGBM); numerik diteruskan. Baseline linear memakai `SimpleImputer` + `StandardScaler` + `OneHotEncoder`.
5. **Baseline & strategi multi-kelas**: Dummy, Logistic Regression One-vs-Rest dan One-vs-One, RandomForest,
   dan LightGBM (native multiclass).
6. **Training** model utama: `Pipeline([features, prep, LGBMClassifier])`.
7. **Hyperparameter tuning**: `RandomizedSearchCV` (3-fold, scoring `f1_macro`) pada subset 60.000 baris.
8. **Evaluasi** pada holdout dan CV 5-fold (probabilitas out-of-fold).
9. **Optimasi Macro F1**: bobot per kelas dikalikan ke probabilitas sebelum `argmax`, dicari dengan
   coordinate ascent pada probabilitas OOF (tidak mengintip label holdout).

## Hasil

Baseline (subset 60.000 baris latih, dievaluasi pada 20.000 baris holdout):

| Model | Accuracy | Macro F1 |
|---|---|---|
| Dummy (kelas mayoritas) | 0.362 | 0.106 |
| Logistic Regression + One-vs-Rest | 0.540 | 0.475 |
| Logistic Regression + One-vs-One | 0.541 | 0.480 |
| RandomForest | 0.546 | 0.476 |
| LightGBM (pipeline akhir, 200 pohon) | 0.573 | 0.532 |

Model utama (data penuh):

| Evaluasi | Macro F1 |
|---|---|
| Holdout 20%, parameter default | 0.579 |
| Holdout 20%, hasil tuning | 0.576 (tidak lebih baik, default dipertahankan) |
| Holdout 20%, default + bobot kelas | 0.584 |
| **CV 5-fold, argmax** | **0.582** |
| **CV 5-fold, + bobot kelas (dipakai untuk submission)** | **0.587** |

Bobot kelas hasil tuning: Grade 1-5 = `[1.00, 1.29, 1.18, 1.09, 0.78]`.
Skor leaderboard Kaggle belum dicek dari sini; estimasi dari CV di sekitar 0.58-0.59.

Catatan tuning: pencarian pada subset kecil memilih pohon yang lebih kecil (`num_leaves=31`) dan hasilnya sedikit
lebih rendah di holdout data penuh, jadi parameter default dipertahankan.

## Interpretasi

- **Lokasi adalah prediktor terkuat.** Lima fitur teratas berdasarkan gain adalah `b_ward` (27%),
  `ward_id` (15%), `vdcmun_id` (14%), `b_vdc` (11%), lalu `superstructure` (10%). Rata-rata kerusakan sangat
  berbeda antar-district (episentrum di sekitar Gorkha/Sindhupalchok), sehingga bangunan di area yang sama cenderung
  rusak serupa. Kode lokasi dari `building_id` memberi informasi lokasi yang lebih halus daripada `ward_id`.
- **Material dan struktur** tetap penting: `superstructure`, `foundation_type`, dan luas/volume bangunan.
  Bangunan berfondasi dan berlantai lumpur (mud mortar) dengan atap berat cenderung Grade 4–5; bangunan RC/semen
  cenderung lebih ringan kerusakannya.
- **Kelas tengah paling sulit.** Per kelas F1: Grade 5 = 0.75, Grade 1 = 0.69, Grade 4 = 0.57,
  Grade 3 = 0.47, Grade 2 = 0.46. Grade 2–3 sering tertukar karena batas antar-tingkat kerusakan bersifat ordinal
  dan sebagian ditentukan penilaian surveyor.
- **Pembobotan kelas** menaikkan Macro F1 sekitar +0.005 karena menaikkan recall kelas minoritas (Grade 2–3)
  dengan sedikit mengorbankan Grade 5.
- **Multi-kelas: OvR vs OvO vs native.** Pada Logistic Regression, OvO sedikit lebih baik dari OvR (0.480 vs 0.475),
  selisihnya kecil. LightGBM dengan multi-kelas native paling tinggi (0.532 pada subset yang sama). Sebagian
  keunggulannya berasal dari kode lokasi halus yang tidak dipakai baseline, jadi perbandingannya bukan murni
  algoritma vs algoritma.
- **Catatan:** fitur `b_ward`/`b_seq` diturunkan dari `building_id`. Ini sah secara data kompetisi, tetapi
  memanfaatkan struktur penomoran survei, sehingga kurang bisa digeneralisasi ke gempa/survei lain.
