# -*- coding: utf-8 -*-
"""v8 blend + NN cesitlilik testi — NN kucuk agirlikla eklenince OOF artiyor mu?"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

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

# NN (tek seed 42)
skf = StratifiedKFold(5, shuffle=True, random_state=42)
nn_oof = np.zeros(len(train)); nn_te = np.zeros(len(test))
for f, (tr, va) in enumerate(skf.split(train, y)):
    d = np.load(BASE + f"\\.v9_state\\nn_fold{f}.npz")
    nn_oof[va] = d["oof"]; nn_te += d["test"] / 5

gbm_oof = np.mean(oofs, 0)
print(f"GBM blend (v8)      : {roc_auc_score(y, gbm_oof):.5f}")
print(f"NN tek basina       : {roc_auc_score(y, nn_oof):.5f}")
best = (0, 0)
for w in np.arange(0.02, 0.31, 0.02):
    a = roc_auc_score(y, (1 - w) * gbm_oof + w * nn_oof)
    if a > best[0]:
        best = (a, w)
    print(f"  GBM + %{int(w*100)} NN : {a:.5f}")
print(f"\nEn iyi: {best[0]:.5f} @ NN agirlik {best[1]:.2f}")
if best[0] > roc_auc_score(y, gbm_oof) + 0.00005:
    w = best[1]
    sub = pd.DataFrame({"id": test["id"],
                        "Will_Buy_EV": (1 - w) * np.mean(tests_, 0) + w * nn_te})
    sub.to_csv(BASE + r"\submission_v9.csv", index=False)
    print(f"submission_v9.csv yazildi (NN %{int(w*100)}).")
else:
    print("NN blend'e deger katmiyor (+0.00005 esigi asilmadi).")
