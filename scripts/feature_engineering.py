"""Transformer feature engineering (kompatibel scikit-learn Pipeline)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

RAW_NUMERIC = [
    "district_id", "vdcmun_id", "ward_id",
    "count_floors_pre_eq", "age_building", "plinth_area_sq_ft", "height_ft_pre_eq",
]


class BuildingFeatureEngineer(BaseEstimator, TransformerMixin):
    """Menambahkan fitur turunan struktur bangunan dan kode lokasi.

    Fitur yang dibuat:
    - ``area_per_floor`` dan ``height_per_floor``: luas dan tinggi per lantai.
    - ``slenderness``: tinggi / akar luas dasar (bangunan langsing lebih rentan).
    - ``volume_proxy``: luas dasar x tinggi.
    - ``age_clipped`` dan ``age_unknown``: umur dibatasi ``age_cap``; nilai ekstrem (mis. 999) ditandai.
    - ``b_vdc``, ``b_ward``, ``b_seq``: potongan kode lokasi dari ``building_id`` (12 digit).

    Parameters
    ----------
    age_cap : int, default=200
        Umur di atas nilai ini dianggap tidak valid.
    use_building_id : bool, default=True
        Jika True, turunkan kode lokasi dari ``building_id``.
    """

    def __init__(self, age_cap: int = 200, use_building_id: bool = True):
        """Simpan hyperparameter transformer (lihat docstring kelas)."""
        self.age_cap = age_cap
        self.use_building_id = use_building_id

    def fit(self, X: pd.DataFrame, y=None):
        """Tidak ada parameter yang dipelajari; hanya mencatat nama kolom keluaran."""
        self.feature_names_out_ = list(self.transform(X.head(5)).columns)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Kembalikan salinan ``X`` dengan fitur turunan."""
        df = X.copy()
        floors = df["count_floors_pre_eq"].clip(lower=1)
        df["area_per_floor"] = df["plinth_area_sq_ft"] / floors
        df["height_per_floor"] = df["height_ft_pre_eq"] / floors
        df["slenderness"] = df["height_ft_pre_eq"] / np.sqrt(df["plinth_area_sq_ft"].clip(lower=1))
        df["volume_proxy"] = df["plinth_area_sq_ft"] * df["height_ft_pre_eq"]
        df["age_unknown"] = (df["age_building"] > self.age_cap).astype(int)
        df["age_clipped"] = df["age_building"].clip(upper=self.age_cap)
        if self.use_building_id and "building_id" in df.columns:
            b = df["building_id"].astype("int64").astype(str).str.zfill(12)
            df["b_vdc"] = b.str[:4].astype(int)
            df["b_ward"] = b.str[:6].astype(int)
            df["b_seq"] = b.str[6:].astype(int)
        return df.drop(columns=["building_id"], errors="ignore")

    def get_feature_names_out(self, input_features=None):
        """Nama kolom keluaran (diperlukan agar kompatibel dengan ``set_output``)."""
        return np.array(self.feature_names_out_)

    def numeric_columns(self) -> list[str]:
        """Daftar kolom numerik keluaran transformer."""
        cols = RAW_NUMERIC + ["area_per_floor", "height_per_floor", "slenderness",
                              "volume_proxy", "age_unknown", "age_clipped"]
        if self.use_building_id:
            cols += ["b_vdc", "b_ward", "b_seq"]
        return cols
