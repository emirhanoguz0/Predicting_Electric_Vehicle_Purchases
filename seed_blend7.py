# -*- coding: utf-8 -*-
"""Tum v6 seed state'lerini (ve .v3_state seed42'yi) toplayip blend OOF + submission.

v3_state = seed 42 (ayni format: {pre}_fold{f}.npz).
  - her model tipi icin tum seed oof'larini ortalar (model-ici seed ensemble)
  - sonra modelleri esit harmanlar
  - submission_v7.csv uretir
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

BASE = r"D:\Second_Brain\🏰 300-Projects\Makine-Ogrenmesi\Predicting_Electric_Vehicle_Purchases"
train = pd.read_csv(BASE + r"\train.csv")
test = pd.read_csv(BASE + r"\test.csv")
y = (train["Will_Buy_EV"] == "Yes").astype(int).values

SOURCES = [(BASE + r"\.v7_state_s42", 42), (BASE + r"\.v7_state_s2024", 2024),
           (BASE + r"\.v7_state_s7", 7)]

model_oofs, model_tests, names = [], [], []
for pre, name in [("lgb", "LGBM"), ("xgb", "XGB"), ("cat", "CatBoost")]:
    seed_oofs, seed_tests = [], []
    for sdir, seed in SOURCES:
        import os
        if not all(os.path.exists(f"{sdir}/{pre}_fold{f}.npz") for f in range(5)):
            print(f"{name} seed{seed}: EKSIK, atlandi.")
            continue
        skf = StratifiedKFold(5, shuffle=True, random_state=seed)
        oof = np.zeros(len(train))
        preds = np.zeros(len(test))
        for fold, (tr, va) in enumerate(skf.split(train, y)):
            d = np.load(f"{sdir}/{pre}_fold{fold}.npz")
            oof[va] = d["oof"]
            preds += d["test"] / 5
        seed_oofs.append(oof)
        seed_tests.append(preds)
        print(f"{name} seed{seed}: OOF {roc_auc_score(y, oof):.5f}")
    if seed_oofs:
        m_oof = np.mean(seed_oofs, 0)
        m_test = np.mean(seed_tests, 0)
        print(f"--> {name} seed-ensemble ({len(seed_oofs)} seed): "
              f"OOF {roc_auc_score(y, m_oof):.5f}\n")
        model_oofs.append(m_oof)
        model_tests.append(m_test)
        names.append(name)

if len(model_oofs) >= 2:
    blend = np.mean(model_oofs, 0)
    print(f"BLEND ({'+'.join(names)}) seed-ensemble: OOF {roc_auc_score(y, blend):.5f}")
    blend_test = np.mean(model_tests, 0)
    sub = pd.DataFrame({"id": test["id"], "Will_Buy_EV": blend_test})
    sub.to_csv(BASE + r"\submission_v7.csv", index=False)
    print("submission_v7.csv yazildi.")
