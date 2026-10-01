"""
Model v4 - Predicting Electric Vehicle Purchases (Playground S6E9)

v3'ten fark: orijinal 10K kaynak set (EV_Adoption_and_Range_Anxiety_Dataset.csv)
her fold'da train parcasina ekleniyor (dogrulama asla orijinal satir icermez).
Neden: SDGM sentetigi satir kopyalamiyor (nn analizi: hedef uyumu taban oraninda),
ama orijinal veri gercek etiketli ek sinyal olabilir.

Kullanim:
  python model_v4.py --models lgb,xgb,catboost
  python model_v4.py --blend
"""
import argparse
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

TARGET = 'Will_Buy_EV'
N_SPLITS = 5
SEED = 42
STATE_DIR = '.v4_state'
os.makedirs(STATE_DIR, exist_ok=True)


def load_data():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    orig = pd.read_csv('source_data/EV_Adoption_and_Range_Anxiety_Dataset.csv')
    orig = orig.drop(columns=[c for c in orig.columns if c not in train.columns])
    features = [c for c in train.columns if c not in ('id', TARGET)]
    cat_cols = train[features].select_dtypes(include=['object', 'category', 'str']).columns.tolist()
    y = (train[TARGET] == 'Yes').astype(int).values
    y_orig = (orig[TARGET] == 'Yes').astype(int).values
    print(f'Orijinal set: {orig.shape} -> her fold train parcasina eklenecek.')
    return train, test, features, cat_cols, y, orig, y_orig


def train_lgb(train, test, features, cat_cols, y, orig, y_orig):
    import lightgbm as lgb
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for fold, (tr, va) in enumerate(skf.split(train[features], y)):
        fp = f'{STATE_DIR}/lgb_fold{fold}.npz'
        if os.path.exists(fp):
            print(f'  LGBM fold {fold+1}/{N_SPLITS} -> ATLANDI (var)')
            d = np.load(fp)
            oof[va] = d['oof']; preds += d['test']
            continue
        X_tr = pd.concat([train[features].iloc[tr], orig[features]], ignore_index=True)
        y_tr = np.concatenate([y[tr], y_orig])
        X_va = train[features].iloc[va].copy()
        for c in cat_cols:
            X_tr[c] = X_tr[c].astype('category')
            X_va[c] = X_va[c].astype('category')
        X_te = test[features].copy()
        for c in cat_cols:
            X_te[c] = X_te[c].astype('category')

        model = lgb.LGBMClassifier(
            n_estimators=10000, learning_rate=0.05, random_state=SEED, verbose=-1)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y[va])], eval_metric='auc',
                  callbacks=[lgb.early_stopping(stopping_rounds=150, verbose=False),
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


def train_xgb(train, test, features, cat_cols, y, orig, y_orig):
    from xgboost import XGBClassifier
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    for fold, (tr, va) in enumerate(skf.split(train[features], y)):
        fp = f'{STATE_DIR}/xgb_fold{fold}.npz'
        if os.path.exists(fp):
            print(f'  XGB fold {fold+1}/{N_SPLITS} -> ATLANDI (var)')
            d = np.load(fp)
            oof[va] = d['oof']; preds += d['test']
            continue
        X_tr = pd.concat([train[features].iloc[tr], orig[features]], ignore_index=True)
        y_tr = np.concatenate([y[tr], y_orig])
        X_va = train[features].iloc[va].copy()
        X_te = test[features].copy()
        for c in cat_cols:
            X_tr[c] = X_tr[c].astype('category')
            X_va[c] = X_va[c].astype('category')
            X_te[c] = X_te[c].astype('category')

        model = XGBClassifier(
            n_estimators=10000, learning_rate=0.05, random_state=SEED,
            enable_categorical=True, tree_method='hist',
            early_stopping_rounds=150)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y[va])], verbose=False)
        oof_va = model.predict_proba(X_va)[:, 1]
        te = model.predict_proba(X_te)[:, 1]
        oof[va] = oof_va
        preds += te / N_SPLITS
        np.savez(fp, oof=oof_va, test=te)
        best_iter = getattr(model, 'best_iteration', None)
        if best_iter is None:
            best_iter = getattr(model, 'best_iteration_', '?')
        print(f'  XGB fold {fold+1}/{N_SPLITS} | AUC: {roc_auc_score(y[va], oof_va):.5f} | '
              f'best_iter: {best_iter}')
    print(f'  XGB OOF AUC: {roc_auc_score(y, oof):.5f}')
    return oof, preds


def train_catboost(train, test, features, cat_cols, y, orig, y_orig):
    from catboost import CatBoostClassifier, Pool
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    cat_idx = [features.index(c) for c in cat_cols]

    for fold, (tr, va) in enumerate(skf.split(train[features], y)):
        fp = f'{STATE_DIR}/cat_fold{fold}.npz'
        if os.path.exists(fp):
            print(f'  CatBoost fold {fold+1}/{N_SPLITS} -> ATLANDI (var)')
            d = np.load(fp)
            oof[va] = d['oof']; preds += d['test']
            continue
        X_tr = pd.concat([train[features].iloc[tr], orig[features]], ignore_index=True)
        y_tr = np.concatenate([y[tr], y_orig])
        pool_tr = Pool(X_tr, y_tr, cat_features=cat_idx)
        pool_va = Pool(train[features].iloc[va], y[va], cat_features=cat_idx)
        pool_te = Pool(test[features], cat_features=cat_idx)

        model = CatBoostClassifier(
            iterations=1500, learning_rate=0.1, random_seed=SEED,
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


def blend(train, test, y):
    oofs, tests, names = [], [], []
    for name, prefix in [('LGBM', 'lgb'), ('XGB', 'xgb'), ('CatBoost', 'cat')]:
        oof = np.zeros(len(train))
        preds = np.zeros(len(test))
        ok = True
        for fold in range(N_SPLITS):
            fp = f'{STATE_DIR}/{prefix}_fold{fold}.npz'
            if not os.path.exists(fp):
                print(f'EKSIK: {fp} -> {name} blend icin hazir degil.')
                ok = False
                break
        if ok:
            skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
            for fold, (tr, va) in enumerate(skf.split(train.drop(columns=[TARGET]), y)):
                d = np.load(f'{STATE_DIR}/{prefix}_fold{fold}.npz')
                oof[va] = d['oof']
                preds += d['test'] / N_SPLITS
            auc = roc_auc_score(y, oof)
            print(f'{name} OOF AUC: {auc:.5f}')
            oofs.append(oof); tests.append(preds); names.append(name)

    if len(oofs) < 2:
        print('Blend icin en az 2 model gerekli.')
        return
    blend_oof = np.mean(oofs, axis=0)
    blend_test = np.mean(tests, axis=0)
    print(f'\nBLEND ({ " + ".join(names) }) OOF AUC: {roc_auc_score(y, blend_oof):.5f}')

    sub = pd.DataFrame({'id': test['id'], TARGET: blend_test})
    sub.to_csv('submission_v4.csv', index=False)
    print('submission_v4.csv yazildi.')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='lgb,xgb,catboost')
    ap.add_argument('--blend', action='store_true')
    args = ap.parse_args()

    train, test, features, cat_cols, y, orig, y_orig = load_data()
    print(f'Train: {train.shape} | Test: {test.shape} | class weight: YOK (v4)')

    if args.blend:
        blend(train, test, y)
        return

    for m in args.models.split(','):
        m = m.strip()
        print(f'\n=== {m.upper()} ===')
        if m == 'lgb':
            train_lgb(train, test, features, cat_cols, y, orig, y_orig)
        elif m == 'xgb':
            train_xgb(train, test, features, cat_cols, y, orig, y_orig)
        elif m in ('cat', 'catboost'):
            train_catboost(train, test, features, cat_cols, y, orig, y_orig)
        else:
            print(f'Bilinmeyen model: {m}')
    print('\nEgitim tamam. Blend icin: python model_v4.py --blend')


if __name__ == '__main__':
    main()
