# -*- coding: utf-8 -*-
"""v5 - FE + tuning deneysi (sadece LGBM, tek seed, hizli A/B).

Karsilastirma:
  --fe off --tune off  -> v3 LGBM birebir (0.94166 referans)
  --fe off --tune on   -> tuning etkisi
  --fe on  --tune off  -> FE etkisi
  --fe on  --tune on   -> ikisi birlikte
"""
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

TARGET = "Will_Buy_EV"
N_SPLITS = 5
SEED = 42

RA_ORD = {"Low": 0, "Medium": 1, "High": 2}


def add_features(df):
    d = df.copy()
    d["Charging_Gap"] = d["Charging_Stations_Near_Work"] - d["Charging_Stations_Near_Home"]
    d["Total_Charging"] = d["Charging_Stations_Near_Home"] + d["Charging_Stations_Near_Work"]
    d["Age_Bin"] = pd.cut(d["Age"], bins=[0, 30, 40, 50, 60, 70, 200],
                          labels=["<30", "30-40", "40-50", "50-60", "60-70", "70+"]).astype(str)
    d["Income_Per_Commute"] = d["Annual_Income_USD"] / (d["Daily_Commute_km"] + 1.0)
    d["Subsidy_x_HomeCharging"] = ((d["Subsidy_Available"] == "Yes") &
                                   (d["Home_Charging_Possible"] == "Yes")).astype(int)
    d["Env_x_RangeAnx"] = d["Environmental_Concern_Level"] * d["Range_Anxiety_Level"].map(RA_ORD)
    d["Commute_x_Cars"] = d["Daily_Commute_km"] * d["Number_of_Cars_Owned"]
    d["EV_Readiness"] = (
        (d["Home_Charging_Possible"] == "Yes").astype(int)
        + (d["Subsidy_Available"] == "Yes").astype(int)
        + (d["Range_Anxiety_Level"] == "Low").astype(int)
        + (d["Charging_Gap"] > 0).astype(int)
        + (d["Environmental_Concern_Level"] >= 4).astype(int)
    )
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fe", choices=["on", "off"], default="on")
    ap.add_argument("--tune", choices=["on", "off"], default="on")
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")

    if args.fe == "on":
        train = add_features(train)
        test = add_features(test)

    features = [c for c in train.columns if c not in ("id", TARGET)]
    cat_cols = train[features].select_dtypes(include=["object", "category", "str"]).columns.tolist()
    y = (train[TARGET] == "Yes").astype(int).values

    params = dict(n_estimators=10000, learning_rate=0.05, random_state=args.seed, verbose=-1)
    if args.tune == "on":
        params.update(num_leaves=127, min_child_samples=40, feature_fraction=0.8,
                      bagging_fraction=0.8, bagging_freq=1, reg_lambda=1.0)

    import lightgbm as lgb
    oof = np.zeros(len(train))
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=args.seed)
    for fold, (tr, va) in enumerate(skf.split(train[features], y)):
        X_tr = train[features].iloc[tr].copy()
        X_va = train[features].iloc[va].copy()
        for c in cat_cols:
            X_tr[c] = X_tr[c].astype("category")
            X_va[c] = X_va[c].astype("category")
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y[tr], eval_set=[(X_va, y[va])], eval_metric="auc",
                  callbacks=[lgb.early_stopping(150, verbose=False),
                             lgb.log_evaluation(period=0)])
        oof[va] = model.predict_proba(X_va)[:, 1]
        print(f"  fold {fold+1}: {roc_auc_score(y[va], oof[va]):.5f} (best_iter {model.best_iteration_})")
    print(f"\nSONUC fe={args.fe} tune={args.tune} seed={args.seed}: "
          f"OOF AUC = {roc_auc_score(y, oof):.5f}")


if __name__ == "__main__":
    main()
