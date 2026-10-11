"""Pemuatan data dan pembentukan ColumnTransformer untuk kompetisi Structural Damage Classification."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"

TARGET = "damage_grade"
ID_COL = "building_id"
LABELS = [1, 2, 3, 4, 5]

CATEGORICAL_COLS = [
    "land_surface_condition", "foundation_type", "roof_type", "ground_floor_type",
    "other_floor_type", "position", "plan_configuration", "superstructure",
]
# Kode lokasi berkardinalitas tinggi: diperlakukan sebagai kategori native LightGBM
GEO_CAT_COLS = ["vdcmun_id", "ward_id", "b_vdc", "b_ward"]


class CategoryCaster(BaseEstimator, TransformerMixin):
    """Ubah kolom menjadi dtype ``category`` dengan daftar kategori dari data train.

    Dipakai agar LightGBM memakai penanganan kategorik native (split berdasarkan himpunan kategori).
    Kategori yang tidak pernah muncul di train menjadi NaN saat transform.
    """

    def fit(self, X: pd.DataFrame, y=None):
        """Simpan daftar kategori terurut tiap kolom."""
        self.columns_ = list(X.columns)
        self.categories_ = {c: sorted(pd.unique(X[c].dropna())) for c in self.columns_}
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Kembalikan DataFrame dengan semua kolom bertipe category."""
        out = pd.DataFrame(index=X.index)
        for c in self.columns_:
            out[c] = pd.Categorical(X[c], categories=self.categories_[c])
        return out

    def get_feature_names_out(self, input_features=None):
        """Nama kolom keluaran (sama dengan masukan)."""
        return np.array(self.columns_)


def load_data(data_dir: Path = DATA_DIR) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Muat train.csv, test.csv, dan sample_submission.csv dari ``data_dir``.

    Nama file toleran terhadap akhiran duplikat unduhan, mis. ``train(1).csv``.

    Returns
    -------
    tuple of DataFrame
        (train, test, sample_submission).
    """

    def _find(stem: str) -> Path:
        """Cari file ``{stem}*.csv`` pertama di ``data_dir``; error jika tidak ada."""
        hits = sorted(data_dir.glob(f"{stem}*.csv"))
        if not hits:
            raise FileNotFoundError(f"{stem}*.csv tidak ditemukan di {data_dir}. Lihat README untuk cara unduh data.")
        return hits[0]

    train = pd.read_csv(_find("train"))
    test = pd.read_csv(_find("test"))
    sample = pd.read_csv(_find("sample_submission"))
    return train, test, sample


def encode_target(series: pd.Series) -> np.ndarray:
    """Ubah label ``"Grade 3"`` atau ``3`` menjadi indeks kelas 0..4."""
    return series.astype(str).str.extract(r"(\d)")[0].astype(int).to_numpy() - 1


def decode_target(idx: np.ndarray, as_text: bool = True) -> list:
    """Kembalikan indeks kelas 0..4 menjadi label submission.

    Parameters
    ----------
    idx : array-like of int
        Indeks kelas (0..4).
    as_text : bool
        True -> ``"Grade N"`` (format sample_submission), False -> angka N.
    """
    return [f"Grade {i + 1}" if as_text else i + 1 for i in idx]


def build_preprocessor(numeric_cols: list[str]) -> ColumnTransformer:
    """Bangun ColumnTransformer.

    Kolom kategorik dan kode lokasi (``CATEGORICAL_COLS`` + ``GEO_CAT_COLS``) diubah menjadi dtype
    ``category`` oleh ``CategoryCaster``. Kolom numerik diteruskan apa adanya karena model akhir
    berbasis pohon sehingga tidak butuh scaling.

    Parameters
    ----------
    numeric_cols : list of str
        Kolom numerik hasil feature engineering (kolom di ``GEO_CAT_COLS`` otomatis dikeluarkan).
    """
    num = [c for c in numeric_cols if c not in GEO_CAT_COLS]
    return ColumnTransformer(
        [("cat", CategoryCaster(), CATEGORICAL_COLS + GEO_CAT_COLS),
         ("num", "passthrough", num)],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
