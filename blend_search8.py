# -*- coding: utf-8 -*-
"""v8 state'lerinde 3 model icin agirlik taramasi + rank/geometrik karsilastirma.

seed_blend8.py'nin esit agirlikli blend'i uzerine; OOF AUC'ye gore en iyi
agirliklari bulur ve submission_v8_weighted.csv uretir.
"""
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

BASE = r"D:\Second_Brain\🏰 300-Projects\Makine-Ogrenmesi\Predicting_Electric_Vehicle_Purchases"
train = pd.read_csv(BASE + r"\train.csv")
test = pd.read_csv(BASE + r"\test.csv")
y = (train["Will_Buy_EV"] == "Yes").astype(int).values

SOURCES = [(BASE + r"\.v8_state_s42", 42), (BASE + r"\.v8_state_s2024", 2024),
           (BASE + r"\.v8_state_s7", 7)]

oofs, tests_, names = [], [], []
for pre, name in [("lgb", "LGBM"), ("xgb", "XGB"), ("cat", "CatBoost")]:
    seed_oofs, seed_tests = [], []
    for sdir, seed in SOURCES:
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        oof = np.zeros(len(train)); preds = np.zeros(len(test))
        for f, (tr, va) in enumerate(skf.split(train, y)):
            d = np.load(f"{sdir}/{pre}_fold{f}.npz")
            oof[va] = d["oof"]; preds += d["test"] / 5
        seed_oofs.append(oof); seed_tests.append(preds)
    oofs.append(np.mean(seed_oofs, 0)); tests_.append(np.mean(seed_tests, 0))
    names.append(name)

def auc(b): return roc_auc_score(y, b)

print(f"esit blend: {auc(np.mean(oofs, 0)):.5f}")
R = np.vstack([rankdata(o) for o in oofs]) / len(train)
print(f"rank esit : {auc(R.mean(0)):.5f}")
eps = 1e-6
G = np.vstack([np.log(np.clip(o, eps, 1 - eps)) for o in oofs])
print(f"geo esit  : {auc(np.exp(G.mean(0))):.5f}")

best = (0, None)
for w1 in np.arange(0.05, 0.86, 0.05):
    for w2 in np.arange(0.05, 0.91 - w1, 0.05):
        w3 = 1 - w1 - w2
        if w3 < 0.05: continue
        a = auc(w1 * oofs[0] + w2 * oofs[1] + w3 * oofs[2])
        if a > best[0]: best = (a, (round(w1, 2), round(w2, 2), round(w3, 2)))
print(f"en iyi agirlikli: {best[0]:.5f} w={best[1]} ({'/'.join(names)})")

w = best[1]
blend_test = w[0] * tests_[0] + w[1] * tests_[1] + w[2] * tests_[2]
sub = pd.DataFrame({"id": test["id"], "Will_Buy_EV": blend_test})
sub.to_csv(BASE + r"\submission_v8_weighted.csv", index=False)
print("submission_v8_weighted.csv yazildi.")
