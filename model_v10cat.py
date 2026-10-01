# -*- coding: utf-8 -*-
"""v10-CatBoost: v10 taban özellikleriyle CatBoost (TE'siz, native kategori).

LGBM/XGB yolundan bilinçli sapma: TE kolonlari YOK (CatBoost ordered
boosting kendi hedef istatistiğini üretir), 6 kategori string olarak
native handling'e verilir. Farkli hata karakteri = blend'de gerçek çeşitlilik.

Kullanim: python model_v10cat.py --seed 42
Sure: ~45-70 dk. State .v10c_state_s{seed}/ ile resume.
Cikti: submission_v10cat_recipe-10fold-cat_s{seed}.csv
"""
import argparse
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
import model_v10

TARGET = 'Will_Buy_EV'
N_SPLITS = 10
CAT_PARAMS = dict(iterations=8000, learning_rate=0.03, depth=6,
                  colsample_bylevel=0.3, subsample=0.9, min_data_in_leaf=100,
                  l2_leaf_reg=3.0, cat_features=None, verbose=0,
                  allow_writing_files=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    print('veri yukleniyor...')
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    y = (train[TARGET] == 'Yes').astype(int).values
    test_id = test['id'].values

    print('taban ozellikler kuruluyor...')
    train, test = model_v10.build_base_features(train, test)
    num_cols = [c for c in train.columns
                if c not in ('id', TARGET) + tuple(model_v10.TE_KEYS) + tuple(model_v10.CAT_COLS)]
    print(f'ozellik: {len(num_cols)} sayisal + {len(model_v10.CAT_COLS)} native kategori (TE yok)')

    state_dir = f'.v10c_state_s{args.seed}'
    os.makedirs(state_dir, exist_ok=True)

    from catboost import CatBoostClassifier, Pool

    features = num_cols + list(model_v10.CAT_COLS)
    oof = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=args.seed)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        fp = f'{state_dir}/fold{fold}.npz'
        if os.path.exists(fp):
            print(f'fold {fold+1}/{N_SPLITS} -> ATLANDI (var)')
            d = np.load(fp)
            oof[va_idx] = d['oof']; test_preds += d['test'] / N_SPLITS
            continue
        A_tr = train[features].iloc[tr_idx].copy()
        A_va = train[features].iloc[va_idx].copy()
        A_te = test[features].copy()
        # kategorileri stringe cevir (catboost native)
        for c in model_v10.CAT_COLS:
            for df in (A_tr, A_va, A_te):
                df[c] = df[c].astype(str)

        pool_tr = Pool(A_tr, y[tr_idx], cat_features=list(model_v10.CAT_COLS))
        pool_va = Pool(A_va, y[va_idx], cat_features=list(model_v10.CAT_COLS))
        pool_te = Pool(A_te, cat_features=list(model_v10.CAT_COLS))

        model = CatBoostClassifier(random_seed=args.seed, eval_metric='AUC',
                                   od_type='Iter', od_wait=500, **CAT_PARAMS)
        model.fit(pool_tr, eval_set=pool_va, use_best_model=True)
        oof_va = model.predict_proba(pool_va)[:, 1]
        te_p = model.predict_proba(pool_te)[:, 1]
        oof[va_idx] = oof_va
        test_preds += te_p / N_SPLITS
        np.savez_compressed(fp, oof=oof_va, test=te_p)
        print(f'fold {fold+1}/{N_SPLITS} | AUC: {roc_auc_score(y[va_idx], oof_va):.5f} | '
              f'best_iter: {model.best_iteration_}')

    auc = roc_auc_score(y, oof)
    print(f'\nCatBoost OOF AUC: {auc:.5f}')
    out = f'submission_v10cat_recipe-10fold-cat_s{args.seed}.csv'
    pd.DataFrame({'id': test_id, TARGET: test_preds}).to_csv(out, index=False)
    np.savez_compressed(f'{state_dir}/full.npz', oof=oof, test=test_preds)
    print(f'{out} yazildi')


if __name__ == '__main__':
    main()
