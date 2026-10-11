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
│   ├── 02_modeling.ipynb     # OvR vs OvO vs native multiclass, hasil CV, feature importance
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

1. **EDA** — data bersih (tanpa missing value, tanpa `building_id` yang bocor antar train/test).
   `age_building` punya kode 999 (tidak diketahui) pada ±2.200 baris. Kelas tidak seimbang
   (Grade 5 ≈ 36%, Grade 1 ≈ 10%).
2. **Feature engineering** (`BuildingFeatureEngineer`): luas & tinggi per lantai, slenderness,
   proxy volume, umur yang di-clip + flag `age_unknown`, dan potongan kode lokasi dari `building_id`
   (`b_vdc`, `b_ward`, `b_seq`).
3. **Preprocessing** (`ColumnTransformer`): kolom kategorik dan kode lokasi → dtype `category`
   (penanganan kategorik native LightGBM); kolom numerik diteruskan apa adanya (model berbasis pohon).
4. **Model** — `Pipeline([features, prep, LGBMClassifier])`, multiclass softmax native.
5. **Evaluasi** — `StratifiedKFold` 5-fold, probabilitas out-of-fold.
6. **Optimasi Macro F1** — bobot per kelas dikalikan ke probabilitas sebelum `argmax`, dicari dengan
   coordinate ascent pada probabilitas OOF (supaya tidak bocor ke data latih).

## Hasil

| Pendekatan | Macro F1 |
|---|---|
| Logistic Regression + One-vs-Rest (subset 60k, holdout) | 0.375 |
| Logistic Regression + One-vs-One (subset 60k, holdout) | 0.386 |
| LightGBM native multiclass (subset 60k, holdout) | 0.516 |
| **LightGBM full data, 5-fold CV (argmax)** | **0.582** |
| **LightGBM + bobot kelas (CV, OOF)** | **0.587** |

Bobot kelas hasil tuning: Grade 1–5 = `[1.00, 1.29, 1.18, 1.09, 0.78]`.
Skor leaderboard Kaggle belum dicek dari sini; estimasi dari CV di sekitar 0.58–0.59.

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
- **Multi-kelas: OvR vs OvO vs native.** Pada model linear, OvO sedikit lebih baik dari OvR, tapi keduanya
  jauh di bawah gradient boosting yang menangani multi-kelas secara native dan menangkap interaksi non-linear
  (material × lokasi × tinggi).
- **Catatan:** fitur `b_ward`/`b_seq` diturunkan dari `building_id`. Ini sah secara data kompetisi, tetapi
  memanfaatkan struktur penomoran survei, sehingga kurang bisa digeneralisasi ke gempa/survei lain.
