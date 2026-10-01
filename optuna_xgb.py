# -*- coding: utf-8 -*-
"""XGB Optuna tuning — v8 ozellikleri (basamak + pairs, 76 kolon).

15 deneme fold0 uzerinde (sqlite kalici, timeout'da tekrar calistirip devam).
Sonunda en iyi paramla tam 5 fold dogrulama.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

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
        d = {}
        for p in range(7):
            name = f"{c}_d{p}"
            out[name] = ((v // (10 ** p)) % 10).astype(np.int8)
            d[p] = out[name]
        out[f"{c}_p01"] = (d[0] + 10 * d[1]).astype(np.int8)
        out[f"{c}_p23"] = (d[2] + 10 * d[3]).astype(np.int8)
    return out


train = build(pd.read_csv("train.csv"))
features = [c for c in train.columns if c not in ("id", TARGET)]
cat_cols = [c for c in features if not pd.api.types.is_numeric_dtype(train[c])]
y = (train[TARGET] == "Yes").astype(int).values

skf = StratifiedKFold(5, shuffle=True, random_state=42)
folds = list(skf.split(train, y))
tr, va = folds[0]
X_tr = train.iloc[tr][features].copy()
X_va = train.iloc[va][features].copy()
for c in cat_cols:
    X_tr[c] = X_tr[c].astype("category")
    X_va[c] = X_va[c].astype("category")


def objective(trial):
    p = {
        "n_estimators": 5000,
        "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.08, log=True),
        "max_depth": trial.suggest_int("max_depth", 4, 10),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 100),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 20.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 5.0, log=True),
        "random_state": 42, "enable_categorical": True, "tree_method": "hist",
        "early_stopping_rounds": 100, "eval_metric": "auc",
    }
    m = XGBClassifier(**p)
    m.fit(X_tr, y[tr], eval_set=[(X_va, y[va])], verbose=False)
    return roc_auc_score(y[va], m.predict_proba(X_va)[:, 1])


import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
study = optuna.create_study(direction="maximize",
                            sampler=optuna.samplers.TPESampler(seed=42),
                            storage="sqlite:///optuna_xgb.db",
                            study_name="v8_xgb",
                            load_if_exists=True)
done = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
if len(done) < 15:
    study.optimize(objective, n_trials=15, show_progress_bar=False)

print("\nEn iyi 5 deneme:")
for t in study.trials_dataframe().sort_values("value", ascending=False).head(5).itertuples():
    print(f"  {t.value:.5f} | lr={t.params_learning_rate:.4f} depth={t.params_max_depth} "
          f"mcw={t.params_min_child_weight} sub={t.params_subsample:.2f} "
          f"col={t.params_colsample_bytree:.2f} l2={t.params_reg_lambda:.3f}")
best = study.best_params
print("\nbest_params =", best)

best.update({"n_estimators": 5000, "random_state": 42, "enable_categorical": True,
             "tree_method": "hist", "early_stopping_rounds": 150, "eval_metric": "auc"})
oof = np.zeros(len(train))
for f, (tr, va) in enumerate(folds):
    fpn = f".optuna_xgb_oof_fold{f}.npy"
    if os.path.exists(fpn):
        oof[va] = np.load(fpn)
        print(f"  dogrulama fold {f+1}: ATLANDI (var) {roc_auc_score(y[va], oof[va]):.5f}", flush=True)
        continue
    X_tr = train.iloc[tr][features].copy()
    X_va = train.iloc[va][features].copy()
    for c in cat_cols:
        X_tr[c] = X_tr[c].astype("category")
        X_va[c] = X_va[c].astype("category")
    m = XGBClassifier(**best)
    m.fit(X_tr, y[tr], eval_set=[(X_va, y[va])], verbose=False)
    oof[va] = m.predict_proba(X_va)[:, 1]
    np.save(f".optuna_xgb_oof_fold{f}.npy", oof[va])
    print(f"  dogrulama fold {f+1}: {roc_auc_score(y[va], oof[va]):.5f} (best_iter {m.best_iteration})", flush=True)
print(f"\nXGB TUNED 5-fold OOF: {roc_auc_score(y, oof):.5f} (referans v8-XGB: 0.94389-0.94397)")
