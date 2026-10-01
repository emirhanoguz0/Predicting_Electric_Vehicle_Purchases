# -*- coding: utf-8 -*-
"""(c) OOF hedef kodlama - hizli surum (kategori kodlari ile grup anahtari)."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb

BASE = r"D:\Second_Brain\🏰 300-Projects\Makine-Ogrenmesi\Predicting_Electric_Vehicle_Purchases"
train = pd.read_csv(BASE + r"\train.csv")
y = (train["Will_Buy_EV"] == "Yes").astype(int).values
GLOBAL = y.mean()
CATCOLS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
           "Subsidy_Available", "Range_Anxiety_Level"]

# kategori kodlari (sabit, tum set uzerinden); sayisallar yuvarlanmis deger
codes = {}
for c in CATCOLS:
    codes[c] = train[c].astype("category").cat.codes.to_numpy(np.int32) + 1
codes["Environmental_Concern_Level"] = train["Environmental_Concern_Level"].fillna(-1).round(1).map(lambda v: int(v * 10) + 100).to_numpy(np.int64)

def combo_key(cols):
    k = np.zeros(len(train), np.int64)
    for c in cols:
        k = k * 1000 + codes[c]
    return k

def oof_te_fast(cols, smooth=20):
    key = combo_key(cols)
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    enc = np.zeros(len(train))
    for tr, va in skf.split(train, y):
        tr_k, va_k = key[tr], key[va]
        order = np.argsort(tr_k, kind="stable")
        sk = tr_k[order]
        sy = y[tr][order]
        u, start = np.unique(sk, return_index=True)
        cnt = np.diff(np.append(start, len(sk)))
        sums = np.add.reduceat(sy, start)
        mean_map = dict(zip(u.tolist(), sums / np.maximum(cnt, 1)))
        cnt_map = dict(zip(u.tolist(), cnt))
        m = np.array([mean_map.get(k, GLOBAL) for k in va_k])
        c = np.array([cnt_map.get(k, 0) for k in va_k], dtype=float)
        enc[va] = (m * c + GLOBAL * smooth) / (c + smooth)
    return enc

combos = [
    ("te_sub_range", ["Subsidy_Available", "Range_Anxiety_Level"]),
    ("te_sub_env", ["Subsidy_Available", "Environmental_Concern_Level"]),
    ("te_range_env", ["Range_Anxiety_Level", "Environmental_Concern_Level"]),
    ("te_sub_range_city", ["Subsidy_Available", "Range_Anxiety_Level", "City_Type"]),
]
core = ["Environmental_Concern_Level", "Subsidy_Available", "Range_Anxiety_Level"]

extra = {}
for name, cols in combos:
    extra[name] = oof_te_fast(cols)
    print(f"{name}: min {extra[name].min():.4f} max {extra[name].max():.4f}")

# (c1) 3 cekirdek kolon + TE
skf = StratifiedKFold(5, shuffle=True, random_state=42)
oof = np.zeros(len(train))
for tr, va in skf.split(train, y):
    X_tr = train.iloc[tr][core].copy()
    X_va = train.iloc[va][core].copy()
    for c in ["Subsidy_Available", "Range_Anxiety_Level"]:
        X_tr[c] = X_tr[c].astype("category")
        X_va[c] = X_va[c].astype("category")
    X_tr = X_tr.assign(**{k: v[tr] for k, v in extra.items()})
    X_va = X_va.assign(**{k: v[va] for k, v in extra.items()})
    m = lgb.LGBMClassifier(n_estimators=10000, learning_rate=0.05,
                           random_state=42, verbose=-1)
    m.fit(X_tr, y[tr], eval_set=[(X_va, y[va])], eval_metric="auc",
          callbacks=[lgb.early_stopping(150, verbose=False),
                     lgb.log_evaluation(period=0)])
    oof[va] = m.predict_proba(X_va)[:, 1]
print(f"3 kolon + 4 TE: OOF AUC = {roc_auc_score(y, oof):.5f}")

# (c2) tum kolonlar + TE
allf = [c for c in train.columns if c not in ("id", "Will_Buy_EV")]
oof = np.zeros(len(train))
for tr, va in skf.split(train, y):
    X_tr = train.iloc[tr][allf].copy()
    X_va = train.iloc[va][allf].copy()
    for c in CATCOLS:
        X_tr[c] = X_tr[c].astype("category")
        X_va[c] = X_va[c].astype("category")
    X_tr = X_tr.assign(**{k: v[tr] for k, v in extra.items()})
    X_va = X_va.assign(**{k: v[va] for k, v in extra.items()})
    m = lgb.LGBMClassifier(n_estimators=10000, learning_rate=0.05,
                           random_state=42, verbose=-1)
    m.fit(X_tr, y[tr], eval_set=[(X_va, y[va])], eval_metric="auc",
          callbacks=[lgb.early_stopping(150, verbose=False),
                     lgb.log_evaluation(period=0)])
    oof[va] = m.predict_proba(X_va)[:, 1]
print(f"TUM kolonlar + 4 TE: OOF AUC = {roc_auc_score(y, oof):.5f}")
