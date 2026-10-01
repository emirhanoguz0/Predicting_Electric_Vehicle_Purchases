# -*- coding: utf-8 -*-
"""v9 - v8 + income lookup (orijinal veri "satir hafizasi").

Kaynak: Kaggle discussion 742385 "the generator copied the original incomes
and remembers their labels" + 738968 "Magic Flags". SDGM orijinal setteki
gelir degerlerini aynen kopyalamis; gelir -> orijinal hedef orani ozelligi
LGBM icinde +~0.0017 kazanc veriyor (OOF hedefi ~0.946).

v8 (digit + pairs + tuned LGBM) uzerine 4 lookup ozelligi ekler:
  - inc_match   : gelir orijinalde var mi (0/1)
  - orig_cnt    : bu gelire sahip orijinal satir sayisi
  - orig_rate   : orijinal sette bu gelirin hedef orani (NaN -> global oran)
  - orig_rate_c2: sadece cnt>=2 ise orig_rate, yoksa NaN -> orig_rate

Kullanim:
  python model_v9.py --seed 42
  python model_v9.py --seed 42 --models lgb
Cikti: .v9_state_s{seed}/ klasorune fold state'leri + submission_v9.csv
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

TARGET = 'Will_Buy_EV'
N_SPLITS = 5

DECIMAL_COLS = {"Daily_Commute_km", "Environmental_Concern_Level"}
DIGIT_COLS = ["Age", "Annual_Income_USD", "Daily_Commute_km",
              "Number_of_Cars_Owned", "Charging_Stations_Near_Home",
              "Charging_Stations_Near_Work", "Environmental_Concern_Level"]
N_DIGITS = 7

LGB_TUNED = dict(n_estimators=2500, learning_rate=0.0243, num_leaves=96,
                 min_child_samples=79, feature_fraction=0.7824,
                 bagging_fraction=0.9141, bagging_freq=1, lambda_l2=0.0063,
                 max_depth=6)

ORIG_PATH = 'source_data/EV_Adoption_and_Range_Anxiety_Dataset.csv'


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


def add_income_lookup(train, test):
    """Gelir -> orijinal hedef orani ozellikleri (train+test'e ayni sekilde)."""
    orig = pd.read_csv(ORIG_PATH)
    g = orig.groupby('Annual_Income_USD')['Will_Buy_EV'].agg(
        orig_cnt='count',
        orig_rate=lambda s: (s == 'Yes').mean())
    global_rate = (orig['Will_Buy_EV'] == 'Yes').mean()
    g = g.reset_index()

    out_tr, out_te = [], []
    for df, name in ((train, 'train'), (test, 'test')):
        m = df[['Annual_Income_USD']].merge(g, on='Annual_Income_USD', how='left')
        m['inc_match'] = m['orig_cnt'].notna().astype(np.int8)
        m['orig_cnt'] = m['orig_cnt'].fillna(0).astype(np.int16)
        m['orig_rate'] = m['orig_rate'].fillna(global_rate).astype(np.float32)
        m['orig_rate_c2'] = np.where(m['orig_cnt'] >= 2, m['orig_rate'],
                                     np.float32(np.nan)).astype(np.float32)
        m['orig_rate_c2'] = m['orig_rate_c2'].fillna(m['orig_rate'])
        keep = ['inc_match', 'orig_cnt', 'orig_rate', 'orig_rate_c2']
        out_tr.append(m[keep]) if name == 'train' else out_te.append(m[keep])
        print(f"  {name} eslesme orani: {m['inc_match'].mean():.4f}")
    return (pd.concat([train.reset_index(drop=True), out_tr[0]], axis=1),
            pd.concat([test.reset_index(drop=True), out_te[0]], axis=1))


def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    train = add_digit_features(train)
    test = add_digit_features(test)
    train, test = add_income_lookup(train, test)
    features = [c for c in train.columns if c not in ('id', TARGET)]
    cat_cols = train[features].select_dtypes(include=['object', 'category', 'str']).columns.tolist()
    y = (train[TARGET] == 'Yes').astype(int).values
    print(f"v9 features: {len(features)} kolon")
    return train, test, features, cat_cols, y


def run_lgb(train, test, features, cat_cols, y, state_dir, seed):
    import lightgbm as lgb
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))
    # kategori listeleri train'den TEK SEFER alinir — train/valid/test tutarli olmali
    cats = {c: pd.Categorical(train[c]).categories for c in cat_cols}
    X_te = test[features].copy()
    for c in cat_cols:
        X_te[c] = pd.Categorical(X_te[c], categories=cats[c])
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
            X_tr[c] = pd.Categorical(X_tr[c], categories=cats[c])
            X_va[c] = pd.Categorical(X_va[c], categories=cats[c])
        model = lgb.LGBMClassifier(random_state=seed, verbose=-1, **LGB_TUNED)
        model.fit(X_tr, y[tr],
                  eval_set=[(X_va, y[va])],
                  eval_metric='auc',
                  callbacks=[lgb.early_stopping(200, verbose=False),
                             lgb.log_evaluation(0)])
        oof_va = model.predict_proba(X_va)[:, 1]
        oof[va] = oof_va
        te = model.predict_proba(X_te)[:, 1]
        preds += te / N_SPLITS
        np.savez_compressed(fp, oof=oof_va, test=te)
        print(f'  LGBM fold {fold+1}/{N_SPLITS} | AUC: {roc_auc_score(y[va], oof_va):.5f} | '
              f'best_iter: {model.best_iteration_}')
    auc = roc_auc_score(y, oof)
    print(f'  LGBM OOF AUC: {auc:.5f}')
    return oof, preds, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--models', default='lgb')
    args = ap.parse_args()

    state_dir = f'.v9_state_s{args.seed}'
    os.makedirs(state_dir, exist_ok=True)

    train, test, features, cat_cols, y = load_data()
    for m in args.models.split(','):
        if m == 'lgb':
            oof, preds, auc = run_lgb(train, test, features, cat_cols, y, state_dir, args.seed)
            sub = pd.DataFrame({'id': test['id'], TARGET: preds})
            # AUC metrigi icin olasilik da sakla (blend icin)
            np.savez_compressed(f'{state_dir}/lgb_full.npz', oof=oof, test=preds)
            sub.to_csv('submission_v9.csv', index=False)
            print(f'submission_v9.csv yazildi | OOF AUC: {auc:.5f}')
        else:
            raise SystemExit(f'bilinmeyen model: {m}')


if __name__ == '__main__':
    main()
