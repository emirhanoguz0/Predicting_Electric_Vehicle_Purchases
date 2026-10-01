# -*- coding: utf-8 -*-
"""v7 uzerine Optuna tuning — basamak ozellikleriyle hparam arama.

20 deneme, tek fold (hiz) -> sonra en iyi paramla tam 5 fold dogrulama.
"""
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

TARGET = "Will_Buy_EV"
DECIMAL_COLS = {"Daily_Commute_km", "Environmental_Concern_Level"}
DIGIT_COLS = ["Age", "Annual_Income_USD", "Daily_Commute_km",
              "Number_of_Cars_Owned", "Charging_Stations_Near_Home",
              "Charging_Stations_Near_Work", "Environmental_Concern_Level"]


def build(df):
    out = df.copy()
    for c in DIGIT_COLS:
        scale = 10 if c in DECIMAL_COLS else 1
        v = np.abs((out[c].fillna(0).astype(np.float64) * scale).round().astype(np.int64))
        for p in range(7):
            out[f"{c}_d{p}"] = ((v // (10 ** p)) % 10).astype(np.int8)
    return out


train = build(pd.read_csv("train.csv"))
features = [c for c in train.columns if c not in ("id", TARGET)]
cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(train[c])]
y = (train[TARGET] == "Yes").astype(int).values

skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(train, y))
# hiz icin tek fold (fold 0: tr=0 va=1)
tr, va = folds[0][0], folds[0][1]
X_tr = train.iloc[tr][features].copy()
X_va = train.iloc[va][features].copy()
for c in cat_cols:
    X_tr[c] = X_tr[c].astype("category")
    X_va[c] = X_va[c].astype("category")
dtr = lgb.Dataset(X_tr, y[tr], categorical_feature=cat_cols)
dva = lgb.Dataset(X_va, y[va], reference=dtr, categorical_feature=cat_cols)


def objective(trial):
    p = {
        "objective": "binary", "metric": "auc", "verbosity": -1,
        "feature_pre_filter": False,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 200),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
        "bagging_freq": 1,
        "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
        "max_depth": trial.suggest_int("max_depth", -1, 12),
    }
    m = lgb.train(p, dtr, num_boost_round=5000, valid_sets=[dva],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    return roc_auc_score(y[va], m.predict(X_va))


import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=42),
                            storage="sqlite:///optuna_v7.db",
                            study_name="v7_lgb",
                            load_if_exists=True)
if len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]) < 15:
    study.optimize(objective, n_trials=15, show_progress_bar=False)

print("\nEn iyi 5 deneme:")
for t in study.trials_dataframe().sort_values("value", ascending=False).head(5).itertuples():
    print(f"  {t.value:.5f} | lr={t.params_learning_rate:.4f} leaves={t.params_num_leaves} "
          f"mcs={t.params_min_child_samples} ff={t.params_feature_fraction:.2f} "
          f"bf={t.params_bagging_fraction:.2f} l2={t.params_lambda_l2:.3f} depth={t.params_max_depth}")
best = study.best_params
best["objective"] = "binary"
best["metric"] = "auc"
best["verbosity"] = -1
best["bagging_freq"] = 1
print("\nbest_params =", best)

# tam 5 fold dogrulama
oof = np.zeros(len(train))
for f, (tr, va) in enumerate(folds):
    X_tr = train.iloc[tr][features].copy()
    X_va = train.iloc[va][features].copy()
    for c in cat_cols:
        X_tr[c] = X_tr[c].astype("category")
        X_va[c] = X_va[c].astype("category")
    m = lgb.LGBMClassifier(n_estimators=5000, random_state=42, verbose=-1, **best)
    m.fit(X_tr, y[tr], eval_set=[(X_va, y[va])], eval_metric="auc",
          callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(period=0)])
    oof[va] = m.predict_proba(X_va)[:, 1]
    np.save(f".optuna_oof_fold{f}.npy", oof[va])
    print(f"  dogrulama fold {f+1}: {roc_auc_score(y[va], oof[va]):.5f} (best_iter {m.best_iteration_})", flush=True)
print(f"\nTUNED 5-fold OOF: {roc_auc_score(y, oof):.5f} (referans v7-LGBM: 0.94371)")
