# -*- coding: utf-8 -*-
"""v8 - v7 + pairs + tuned LGBM.

v7 basamak ozelliklerine iki basamakli parcalar (d0+10*d1, d2+10*d3) eklendi
(model_v8.py ablasyonunda +0.00011). LGBM Optuna ile tune edildi
(optuna_v7.py, +0.00016). State .v8_state_s{seed}.

Kullanim:
  python model_v8.py --models lgb --seed 42
  python seed_blend8.py
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

TARGET = 'Will_Buy_EV'
N_SPLITS = 5

# 1 ondalik tasiyan kolonlar (commute, concern) x10 ile tamsayi yapilir
DECIMAL_COLS = {"Daily_Commute_km", "Environmental_Concern_Level"}
DIGIT_COLS = ["Age", "Annual_Income_USD", "Daily_Commute_km",
              "Number_of_Cars_Owned", "Charging_Stations_Near_Home",
              "Charging_Stations_Near_Work", "Environmental_Concern_Level"]
N_DIGITS = 7  # birler .. milyonlar

# optuna_v7.py best (fold0) — lr dusuk, iterasyon ~1600
LGB_TUNED = dict(n_estimators=2500, learning_rate=0.0243, num_leaves=96,
                 min_child_samples=79, feature_fraction=0.7824,
                 bagging_fraction=0.9141, bagging_freq=1, lambda_l2=0.0063,
                 max_depth=6)


def add_digit_features(df):
    out = df.copy()
    for c in DIGIT_COLS:
        scale = 10 if c in DECIMAL_COLS else 1
        v = (out[c].fillna(0).astype(np.float64) * scale).round().astype(np.int64)
        v = np.abs(v)
        d = {}
        for p in range(N_DIGITS):
            name = f"{c}_d{p}"
            out[name] = ((v // (10 ** p)) % 10).astype(np.int8)
            d[p] = out[name]
        out[f"{c}_p01"] = (d[0] + 10 * d[1]).astype(np.int8)
        out[f"{c}_p23"] = (d[2] + 10 * d[3]).astype(np.int8)
    return out


def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    train = add_digit_features(train)
    test = add_digit_features(test)
    features = [c for c in train.columns if c not in ('id', TARGET)]
    cat_cols = train[features].select_dtypes(include=['object', 'category', 'str']).columns.tolist()
    y = (train[TARGET] == 'Yes').astype(int).values
    print(f"Digit+pairs features: {len(features)} kolon")
    return train, test, features, cat_cols, y


def run_lgb(train, test, features, cat_cols, y, state_dir, seed):
    import lightgbm as lgb
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(train[features], y)):
        fp = f'{state_dir}/lgb_fold{fold}.npz'
        if os.path.exists(fp):
            print(f'  LGBM fold {fold+1}/{N_SPLITS} -> ATLANDI (var)')
            d = np.load(fp)
            oof[va] = d['oof']; preds += d['test']
            continue
        X_tr = train[features].iloc[tr].copy()
        X_va = train[features].iloc[va].copy()
        for c in cat_cols:
            X_tr[c] = X_tr[c].astype('category')
            X_va[c] = X_va[c].astype('category')
        X_te = test[features].copy()
        for c in cat_cols:
            X_te[c] = X_te[c].astype('category')
        model = lgb.LGBMClassifier(random_state=seed, verbose=-1, **LGB_TUNED)
        model.fit(X_tr, y[tr], eval_set=[(X_va, y[va])], eval_metric='auc',
                  callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False),
                             lgb.log_evaluation(period=0)])
        oof_va = model.predict_proba(X_va)[:, 1]
        te = model.predict_proba(X_te)[:, 1]
        oof[va] = oof_va
        preds += te / N_SPLITS
        np.savez(fp, oof=oof_va, test=te)
        print(f'  LGBM fold {fold+1}/{N_SPLITS} | AUC: {roc_auc_score(y[va], oof_va):.5f} | '
              f'best_iter: {model.best_iteration_}')
    print(f'  LGBM OOF AUC: {roc_auc_score(y, oof):.5f}')
    return oof, preds


def run_xgb(train, test, features, cat_cols, y, state_dir, seed):
    from xgboost import XGBClassifier
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    for fold, (tr, va) in enumerate(skf.split(train[features], y)):
        fp = f'{state_dir}/xgb_fold{fold}.npz'
        if os.path.exists(fp):
            print(f'  XGB fold {fold+1}/{N_SPLITS} -> ATLANDI (var)')
            d = np.load(fp)
            oof[va] = d['oof']; preds += d['test']
            continue
        X_tr = train[features].iloc[tr].copy()
        X_va = train[features].iloc[va].copy()
        X_te = test[features].copy()
        for c in cat_cols:
            X_tr[c] = X_tr[c].astype('category')
            X_va[c] = X_va[c].astype('category')
            X_te[c] = X_te[c].astype('category')
        model = XGBClassifier(
            n_estimators=10000, learning_rate=0.05, random_state=seed,
            enable_categorical=True, tree_method='hist',
            early_stopping_rounds=150)
        model.fit(X_tr, y[tr], eval_set=[(X_va, y[va])], verbose=False)
        oof_va = model.predict_proba(X_va)[:, 1]
        te = model.predict_proba(X_te)[:, 1]
        oof[va] = oof_va
        preds += te / N_SPLITS
        np.savez(fp, oof=oof_va, test=te)
        print(f'  XGB fold {fold+1}/{N_SPLITS} | AUC: {roc_auc_score(y[va], oof_va):.5f}')
    print(f'  XGB OOF AUC: {roc_auc_score(y, oof):.5f}')
    return oof, preds


def run_cat(train, test, features, cat_cols, y, state_dir, seed, iters=1500):
    from catboost import CatBoostClassifier, Pool
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    cat_idx = [features.index(c) for c in cat_cols]
    for fold, (tr, va) in enumerate(skf.split(train[features], y)):
        fp = f'{state_dir}/cat_fold{fold}.npz'
        if os.path.exists(fp):
            print(f'  CatBoost fold {fold+1}/{N_SPLITS} -> ATLANDI (var)')
            d = np.load(fp)
            oof[va] = d['oof']; preds += d['test']
            continue
        pool_tr = Pool(train[features].iloc[tr], y[tr], cat_features=cat_idx)
        pool_va = Pool(train[features].iloc[va], y[va], cat_features=cat_idx)
        pool_te = Pool(test[features], cat_features=cat_idx)
        model = CatBoostClassifier(
            iterations=iters, learning_rate=0.1, random_seed=seed,
            verbose=0, early_stopping_rounds=100, thread_count=-1)
        model.fit(pool_tr, eval_set=pool_va, use_best_model=True)
        oof_va = model.predict_proba(pool_va)[:, 1]
        te = model.predict_proba(pool_te)[:, 1]
        oof[va] = oof_va
        preds += te / N_SPLITS
        np.savez(fp, oof=oof_va, test=te)
        print(f'  CatBoost fold {fold+1}/{N_SPLITS} | AUC: {roc_auc_score(y[va], oof_va):.5f} | '
              f'best_iter: {model.best_iteration_}')
    print(f'  CatBoost OOF AUC: {roc_auc_score(y, oof):.5f}')
    return oof, preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='lgb')
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--cat-iters', type=int, default=1500)
    args = ap.parse_args()

    state_dir = f'.v8_state_s{args.seed}'
    os.makedirs(state_dir, exist_ok=True)

    train, test, features, cat_cols, y = load_data()
    print(f'State: {state_dir} | seed={args.seed}')

    for m in args.models.split(','):
        m = m.strip()
        print(f'\n=== {m.upper()} ===')
        if m == 'lgb':
            run_lgb(train, test, features, cat_cols, y, state_dir, args.seed)
        elif m == 'xgb':
            run_xgb(train, test, features, cat_cols, y, state_dir, args.seed)
        elif m in ('cat', 'catboost'):
            run_cat(train, test, features, cat_cols, y, state_dir, args.seed, args.cat_iters)


if __name__ == '__main__':
    main()
