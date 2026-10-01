# -*- coding: utf-8 -*-
"""Uc hizli sonuc (LGBM, 5 fold, erken durma):

(a) duplikat satir analizi - ayni feature vektörü kac kez tekrar ediyor,
    etiket tutarli mi?
(b) 3 kolonlu model - sinyal kac kolonda?
(c) OOF hedef kodlamali kategori kombinasyonlari (+smoothing) kazanci?
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

BASE = r"D:\Second_Brain\🏰 300-Projects\Makine-Ogrenmesi\Predicting_Electric_Vehicle_Purchases"
TARGET = "Will_Buy_EV"
train = pd.read_csv(BASE + r"\train.csv")
y = (train["Will_Buy_EV"] == "Yes").astype(int).values
FEATS = [c for c in train.columns if c not in ("id", "Will_Buy_EV")]
CAT = [c for c in FEATS if not pd.api.types.is_numeric_dtype(train[c])]

# ---------- (a) duplikat analizi ----------
print("=== (a) DUPLIKAT ANALIZI ===")
key = train[FEATS].astype(str).agg("|".join, axis=1)
vc = key.value_counts()
dup_mask = key.isin(vc[vc > 1].index)
print(f"tekrar eden feature vektörü satir sayisi: {dup_mask.sum()} ({dup_mask.mean():.1%})")
print(f"tekrar eden benzersiz vektör sayisi: {(vc > 1).sum()}")

# tekrar yok -> etiket tutarliligi sorgusu anlamsiz, atla

# ---------- ortak LGBM kosucusu ----------
def run_lgb(feat_list, cat_list, label, seed=42, extra_num=None):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(train))
    for tr, va in skf.split(train, y):
        X_tr = train.iloc[tr][feat_list].copy()
        X_va = train.iloc[va][feat_list].copy()
        for c in cat_list:
            X_tr[c] = X_tr[c].astype("category")
            X_va[c] = X_va[c].astype("category")
        if extra_num is not None:
            X_tr = X_tr.assign(**{k: v[tr] for k, v in extra_num.items()})
            X_va = X_va.assign(**{k: v[va] for k, v in extra_num.items()})
        m = lgb.LGBMClassifier(n_estimators=10000, learning_rate=0.05,
                               random_state=seed, verbose=-1)
        m.fit(X_tr, y[tr], eval_set=[(X_va, y[va])], eval_metric="auc",
              callbacks=[lgb.early_stopping(150, verbose=False),
                         lgb.log_evaluation(period=0)])
        oof[va] = m.predict_proba(X_va)[:, 1]
    print(f"{label}: OOF AUC = {roc_auc_score(y, oof):.5f}")
    return oof

# ---------- (b) 3 kolonlu ----------
print("\n=== (b) SINYAL KAC KOLONDA? ===")
core = ["Environmental_Concern_Level", "Subsidy_Available", "Range_Anxiety_Level"]
run_lgb(core, ["Subsidy_Available", "Range_Anxiety_Level"], "3 cekirdek kolon")

# ---------- (c) OOF hedef kodlama ----------
print("\n=== (c) OOF TARGET ENCODING ===")
combos = [
    ("te_3way", ["Subsidy_Available", "Range_Anxiety_Level"]),
    ("te_sub_env", ["Subsidy_Available", "Environmental_Concern_Level"]),
    ("te_range_env", ["Range_Anxiety_Level", "Environmental_Concern_Level"]),
    ("te_sub_range_city", ["Subsidy_Available", "Range_Anxiety_Level", "City_Type"]),
]
GLOBAL = y.mean()

def oof_te(cols, smooth=20):
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    enc = np.zeros(len(train))
    for tr, va in skf.split(train, y):
        g = pd.DataFrame({"y": y[tr]}).groupby(train.iloc[tr][cols].astype(str).agg("|".join, axis=1).values)["y"].agg(["mean", "count"])
        keys = train.iloc[va][cols].astype(str).agg("|".join, axis=1).values
        m = g["mean"].reindex(keys).to_numpy()
        c = g["count"].reindex(keys).to_numpy()
        m = np.where(np.isnan(m), GLOBAL, m)
        c = np.where(np.isnan(c), 0, c)
        enc[va] = (m * c + GLOBAL * smooth) / (c + smooth)
    return enc

extra = {}
for name, cols in combos:
    extra[name] = oof_te(cols)
    # tekil degeri ne kadar ayirt edici?
    print(f"{name}: min {extra[name].min():.4f} max {extra[name].max():.4f}")

run_lgb(core, ["Subsidy_Available", "Range_Anxiety_Level"],
        "3 kolon + 4 TE kombinasyonu", extra_num=extra)
