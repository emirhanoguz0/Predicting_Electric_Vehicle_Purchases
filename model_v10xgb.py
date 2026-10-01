# -*- coding: utf-8 -*-
"""v10-XGB: v10 reçete özellikleriyle XGBoost (çeşitlilik üyesi).

Ayni 90 ozellik (48 sayisal + 6 kategori kodu + 30 triple TE) ile
10-fold XGBoost. Stack'te kendi LGBM'imizden farkli hata karakteri
üretmesi icin var.

Kullanim: python model_v10xgb.py --seed 42
Sure: ~45-70 dk. State .v10x_state_s{seed}/ ile resume.
Cikti: submission_v10xgb_recipe-10fold-xgb_s{seed}.csv
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
XGB_PARAMS = dict(n_estimators=8000, learning_rate=0.03, max_depth=5,
                  colsample_bytree=0.3, subsample=0.9, min_child_weight=50,
                  reg_lambda=1.0, tree_method='hist', eval_metric='auc',
                  early_stopping_rounds=500)


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
    cats = {c: pd.Categorical(train[c]).categories for c in model_v10.CAT_COLS}
    print(f'ozellik: {len(num_cols)} sayisal + {len(model_v10.CAT_COLS)} kategori + {len(model_v10.TE_KEYS)*3} TE')

    state_dir = f'.v10x_state_s{args.seed}'
    os.makedirs(state_dir, exist_ok=True)

    from sklearn.preprocessing import TargetEncoder
    import xgboost as xgb

    oof = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=args.seed)

    def assemble(X, cols):
        parts = [X[num_cols].reset_index(drop=True)]
        for c in model_v10.CAT_COLS:
            codes = pd.Categorical(X[c].reset_index(drop=True),
                                   categories=cats[c]).codes.astype(np.int16)
            parts.append(pd.Series(codes, name=c).to_frame())
        parts.append(pd.DataFrame(cols, columns=[f'te_{i}' for i in range(cols.shape[1])]))
        return pd.concat(parts, axis=1)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        fp = f'{state_dir}/fold{fold}.npz'
        if os.path.exists(fp):
            print(f'fold {fold+1}/{N_SPLITS} -> ATLANDI (var)')
            d = np.load(fp)
            oof[va_idx] = d['oof']; test_preds += d['test'] / N_SPLITS
            continue
        X_tr, X_va = train.iloc[tr_idx], train.iloc[va_idx]
        print(f'fold {fold+1}/{N_SPLITS}: triple TE hesaplaniyor...')
        tr_c, va_c, te_c = [], [], []
        for k in model_v10.TE_KEYS:
            for sm in model_v10.TE_SMOOTHS:
                enc = TargetEncoder(target_type='binary', smooth=sm)
                tr_c.append(enc.fit_transform(X_tr[[k]], y[tr_idx]).ravel().astype(np.float32))
                va_c.append(enc.transform(X_va[[k]]).ravel().astype(np.float32))
                te_c.append(enc.transform(test[[k]]).ravel().astype(np.float32))
        A_tr = assemble(X_tr, np.vstack(tr_c).T)
        A_va = assemble(X_va, np.vstack(va_c).T)
        A_te = assemble(test, np.vstack(te_c).T)

        model = xgb.XGBClassifier(random_state=args.seed, verbosity=0, **XGB_PARAMS)
        model.fit(A_tr, y[tr_idx], eval_set=[(A_va, y[va_idx])], verbose=False)
        oof_va = model.predict_proba(A_va)[:, 1]
        te_p = model.predict_proba(A_te)[:, 1]
        oof[va_idx] = oof_va
        test_preds += te_p / N_SPLITS
        np.savez_compressed(fp, oof=oof_va, test=te_p)
        print(f'fold {fold+1}/{N_SPLITS} | AUC: {roc_auc_score(y[va_idx], oof_va):.5f} | '
              f'best_iter: {model.best_iteration}')

    auc = roc_auc_score(y, oof)
    print(f'\nXGB OOF AUC: {auc:.5f}')
    out = f'submission_v10xgb_recipe-10fold-xgb_s{args.seed}.csv'
    pd.DataFrame({'id': test_id, TARGET: test_preds}).to_csv(out, index=False)
    np.savez_compressed(f'{state_dir}/full.npz', oof=oof, test=test_preds)
    print(f'{out} yazildi')


if __name__ == '__main__':
    main()
