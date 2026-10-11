"""Prediksi data test dan penulisan submission.csv sesuai format sample_submission."""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from scripts.preprocess import ROOT, decode_target, load_data
from scripts.train import META_PATH, MODEL_PATH


def make_submission(out_path=ROOT / "submissions" / "submission.csv") -> pd.DataFrame:
    """Muat model tersimpan, prediksi test, tulis CSV dengan urutan ``sample_submission``.

    Label ditulis sebagai angka 1-5 (format yang diminta evaluator Kaggle; sample_submission memakai ``"Grade N"`` tetapi ditolak).
    """
    _, test, sample = load_data()
    pipe = joblib.load(MODEL_PATH)
    w = np.array(json.loads(META_PATH.read_text())["class_weights"])
    proba = pipe.predict_proba(test)
    pred = (proba * w).argmax(1)
    sub = pd.DataFrame({"building_id": test["building_id"], "damage_grade": decode_target(pred, as_text=False)})
    sub = sample[["building_id"]].merge(sub, on="building_id", how="left")
    assert sub["damage_grade"].notna().all() and len(sub) == len(sample)
    sub.to_csv(out_path, index=False)
    return sub


if __name__ == "__main__":
    s = make_submission()
    print(s.damage_grade.value_counts(normalize=True).sort_index())
