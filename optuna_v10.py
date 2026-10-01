# -*- coding: utf-8 -*-
"""v10 reçetesi için hafif Optuna tuning (fold 0 üzerinde).

Amaç: LGBM hiperparametrelerinde (megayak ayarlari civarinda) kazanc var mi
gorup yoksa vakit kaybetmemek. Her deneme fold 0 ile sinirli (~2-3 dk).
Bulunan en iyi parametrelerle TAM kosu model_v10_tuned.py ile yapilir.

Kullanim: python optuna_v10.py --n-trials 20
Sure: ~40-60 dk (20 deneme). sqlite kalici, resume edilebilir.
"""
import argparse
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
import model_v10  # build_base_features, TE_KEYS, TE_SMOOTHS, CAT_COLS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-trials', type=int, default=20)
    args = ap.parse_args()

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    print('veri yukleniyor...')
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    y = (train['Will_Buy_EV'] == 'Yes').astype(int).values
    train, test = model_v10.build_base_features(train, test)

    num_cols = [c for c in train.columns
                if c not in ('id', 'Will_Buy_EV') + tuple(model_v10.TE_KEYS) + tuple(model_v10.CAT_COLS)]
    cats = {c: pd.Categorical(train[c]).categories for c in model_v10.CAT_COLS}

    # fold 0 bolmesi
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    tr_idx, va_idx = next(skf.split(train, y))
    X_tr, X_va = train.iloc[tr_idx], train.iloc[va_idx]

    print('triple TE (fold 0) hesaplaniyor...')
    from sklearn.preprocessing import TargetEncoder
    tr_cols, va_cols, te_cols = [], [], []
    for k in model_v10.TE_KEYS:
        for sm in model_v10.TE_SMOOTHS:
            enc = TargetEncoder(target_type='binary', smooth=sm)
            tr_cols.append(enc.fit_transform(X_tr[[k]], y[tr_idx]).ravel().astype(np.float32))
            va_cols.append(enc.transform(X_va[[k]]).ravel().astype(np.float32))
            te_cols.append(enc.transform(test[[k]]).ravel().astype(np.float32))

    def assemble(X, cols):
        parts = [X[num_cols].reset_index(drop=True)]
        for c in model_v10.CAT_COLS:
            codes = pd.Categorical(X[c].reset_index(drop=True),
                                   categories=cats[c]).codes.astype(np.int16)
            parts.append(pd.Series(codes, name=c).to_frame())
        parts.append(pd.DataFrame(cols, columns=[f'te_{i}' for i in range(cols.shape[1])]))
        return pd.concat(parts, axis=1)

    A_tr = assemble(X_tr, np.vstack(tr_cols).T)
    A_va = assemble(X_va, np.vstack(va_cols).T)

    def objective(trial):
        params = dict(
            n_estimators=8000,
            learning_rate=trial.suggest_float('learning_rate', 0.01, 0.04, log=True),
            num_leaves=trial.suggest_int('num_leaves', 16, 64),
            max_depth=trial.suggest_int('max_depth', 4, 7),
            colsample_bytree=trial.suggest_float('colsample_bytree', 0.25, 0.5),
            max_bin=1024,
            min_child_samples=trial.suggest_int('min_child_samples', 20, 120),
            lambda_l2=trial.suggest_float('lambda_l2', 1e-4, 0.1, log=True),
        )
        import lightgbm as lgb
        model = lgb.LGBMClassifier(random_state=42, verbose=-1, **params)
        model.fit(A_tr, y[tr_idx], eval_set=[(A_va, y[va_idx])], eval_metric='auc',
                  callbacks=[lgb.early_stopping(500, verbose=False), lgb.log_evaluation(0)])
        return roc_auc_score(y[va_idx], model.predict_proba(A_va)[:, 1])

    study = optuna.create_study(direction='maximize',
                                storage='sqlite:///optuna_v10.db',
                                study_name='v10_tune', load_if_exists=True)
    study.optimize(objective, n_trials=args.n_trials)

    print(f'\nEn iyi fold0 AUC: {study.best_value:.5f}')
    print('Parametreler:', study.best_params)
    ref = 0.94501  # v10 default fold0 AUC (ayni bolme) — karsilastirma bunla
    print(f'Referans (v10 default, fold0): ~0.94724 — fark: {study.best_value - ref:+.5f}')


if __name__ == '__main__':
    main()
