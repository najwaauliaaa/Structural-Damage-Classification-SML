"""Training: Pipeline (FeatureEngineer -> ColumnTransformer -> LightGBM), CV, dan tuning bobot kelas."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from scripts.feature_engineering import BuildingFeatureEngineer
from scripts.preprocess import ROOT, build_preprocessor, encode_target, load_data

MODEL_PATH = ROOT / "models" / "pipeline_model_lgbm.pkl"
META_PATH = ROOT / "models" / "class_weights.json"

DEFAULT_PARAMS = dict(
    n_estimators=600,
    learning_rate=0.06,
    num_leaves=127,
    min_child_samples=30,
    subsample=0.8,
    subsample_freq=1,
    colsample_bytree=0.7,
    reg_lambda=1.0,
    cat_smooth=30,
    min_data_per_group=50,
    cat_l2=10,
    n_jobs=2,
    verbose=-1,
    random_state=42,
)


def build_pipeline(params: dict | None = None) -> Pipeline:
    """Rakit Pipeline end-to-end.

    Urutan: ``BuildingFeatureEngineer`` -> ``ColumnTransformer`` -> ``LGBMClassifier``.

    Parameters
    ----------
    params : dict, optional
        Hyperparameter LightGBM yang menimpa ``DEFAULT_PARAMS``.
    """
    fe = BuildingFeatureEngineer()
    prep = build_preprocessor(fe.numeric_columns())
    clf = lgb.LGBMClassifier(**{**DEFAULT_PARAMS, **(params or {})})
    return Pipeline([("features", fe), ("prep", prep), ("model", clf)])


def macro_f1(y_true: np.ndarray, proba: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Macro F1 dari probabilitas, opsional dengan pembobotan per kelas sebelum argmax."""
    w = np.ones(proba.shape[1]) if weights is None else weights
    return f1_score(y_true, (proba * w).argmax(1), average="macro")


def tune_class_weights(y_true: np.ndarray, proba: np.ndarray, n_rounds: int = 3) -> np.ndarray:
    """Cari bobot per kelas yang memaksimalkan Macro F1 (coordinate ascent pada grid log-spasi).

    Dipakai pada probabilitas out-of-fold agar tidak bocor.

    Returns
    -------
    ndarray of shape (n_classes,)
        Bobot yang dikalikan ke probabilitas sebelum argmax.
    """
    w = np.ones(proba.shape[1])
    grid = np.exp(np.linspace(np.log(0.4), np.log(3.0), 25))
    best = macro_f1(y_true, proba, w)
    for _ in range(n_rounds):
        for k in range(proba.shape[1]):
            for g in grid:
                w2 = w.copy()
                w2[k] = g
                s = macro_f1(y_true, proba, w2)
                if s > best:
                    best, w = s, w2
    return w / w[0]


def cross_val_proba(X: pd.DataFrame, y: np.ndarray, params: dict | None = None, n_splits: int = 5, seed: int = 42):
    """Probabilitas out-of-fold dengan StratifiedKFold.

    Returns
    -------
    ndarray of shape (n_samples, n_classes)
    """
    oof = np.zeros((len(X), 5))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (tr, va) in enumerate(skf.split(X, y), 1):
        pipe = build_pipeline(params).fit(X.iloc[tr], y[tr])
        oof[va] = pipe.predict_proba(X.iloc[va])
        print(f"fold {i}: macroF1 argmax = {macro_f1(y[va], oof[va]):.4f}", flush=True)
    return oof


def main(n_splits: int = 5) -> None:
    """Jalankan CV, tuning bobot kelas, latih model final pada seluruh train, simpan ke ``models/``."""
    train, _, _ = load_data()
    y = encode_target(train["damage_grade"])
    X = train.drop(columns=["damage_grade"])

    oof = cross_val_proba(X, y, n_splits=n_splits)
    raw = macro_f1(y, oof)
    w = tune_class_weights(y, oof)
    tuned = macro_f1(y, oof, w)
    print(f"CV macroF1 argmax={raw:.4f} | dengan bobot kelas={tuned:.4f} | w={np.round(w, 3)}")

    pipe = build_pipeline().fit(X, y)
    joblib.dump(pipe, MODEL_PATH)
    META_PATH.write_text(json.dumps({"class_weights": w.tolist(), "cv_macro_f1_raw": raw, "cv_macro_f1_tuned": tuned}, indent=2))
    np.save(ROOT / "models" / "oof_proba.npy", oof)
    print("Model disimpan:", MODEL_PATH)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--folds", type=int, default=5)
    main(ap.parse_args().folds)
