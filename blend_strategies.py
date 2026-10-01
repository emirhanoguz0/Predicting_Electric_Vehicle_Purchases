# -*- coding: utf-8 -*-
"""v3 state'leri uzerinde blend stratejilerini OOF'ta karsilastir.

- arithmetic: duz olasilik ortalamasi (mevcut yaklasim)
- rank: satir bazli rank ortalamasi (742332'nin onerisi)
- geo: geometrik ortalama
- weighted: OOF AUC agirlikli (blend_search ile ayni mantik, hizli tarama)
"""
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

BASE = r"D:\Second_Brain\🏰 300-Projects\Makine-Ogrenmesi\Predicting_Electric_Vehicle_Purchases"
train = pd.read_csv(BASE + r"\train.csv")
y = (train["Will_Buy_EV"] == "Yes").astype(int).values
skf = StratifiedKFold(5, shuffle=True, random_state=42)

oofs, tests, names = [], [], []
for pre, name in [("lgb", "LGBM"), ("xgb", "XGB"), ("cat", "CatBoost")]:
    oof = np.zeros(len(train))
    for f, (tr, va) in enumerate(skf.split(train, y)):
        oof[va] = np.load(BASE + f"\\.v3_state\\{pre}_fold{f}.npz")["oof"]
    oofs.append(oof)
    names.append(name)
    print(f"{name} OOF AUC: {roc_auc_score(y, oof):.5f}")

O = np.vstack(oofs)  # (3, N)

def auc(b):
    return roc_auc_score(y, b)

print("\n--- blend stratejileri (OOF) ---")
print(f"arithmetic      : {auc(O.mean(0)):.5f}")
R = np.vstack([rankdata(o) for o in O]) / len(train)
print(f"rank avg        : {auc(R.mean(0)):.5f}")
eps = 1e-6
print(f"geometric       : {auc(np.exp(np.log(np.clip(O, eps, 1 - eps)).mean(0))):.5f}")

# agirlik taramasi (2 boyutlu, 3. = 1-w1-w2)
best = (0, None)
for w1 in np.arange(0.1, 0.71, 0.05):
    for w2 in np.arange(0.1, 0.81 - w1, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.05:
            continue
        a = auc(w1 * O[0] + w2 * O[1] + w3 * O[2])
        if a > best[0]:
            best = (a, (round(w1, 2), round(w2, 2), round(w3, 2)))
print(f"en iyi agirlikli: {best[0]:.5f}  w={best[1]}  ({names[0]}/{names[1]}/{names[2]})")

# rank + agirlikli kombinasyonu
best_r = (0, None)
for w1 in np.arange(0.1, 0.71, 0.05):
    for w2 in np.arange(0.1, 0.81 - w1, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.05:
            continue
        a = auc(w1 * R[0] + w2 * R[1] + w3 * R[2])
        if a > best_r[0]:
            best_r = (a, (round(w1, 2), round(w2, 2), round(w3, 2)))
print(f"en iyi rank agir: {best_r[0]:.5f}  w={best_r[1]}")
