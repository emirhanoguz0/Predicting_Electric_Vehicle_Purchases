# -*- coding: utf-8 -*-
"""v10 - megayak/najiama reçetesi (README'den implemente).

Kaynak: public_sources/megayak_s6e9-hybrid-lgbm-oof/README.md
Hedef: tek LightGBM ile CV ~0.946 (ham veriden, blend'siz).

Ozellik aileleri:
  1) Generator leak:
     - gelir & commute basamaklari (income d0..d5, commute x10 d0..d4)
     - moduli: income % 7/9/11/13/97, commute10 % 7/9/11/13
     - quant ladder: income // 50/100/250/500/1000/2500/5000
     - deger frekansi: her kategori + gelir + commute10 icin train+test
       concat uzerinde normalize frekans (transduktif, etiketsiz)
  2) najiama V3:
     - Smooth Keys: gelin tam/str //100 //1000 + commute10 str
     - triple TE: 6 kategori + 4 smooth key x sklearn TargetEncoder
       (smooth='auto'/10/100) — her dis fold'da inner CV ile fit
     - orijinal veri hedef ortalamalari (gelir bazinda)
     - bayraklar: is_30k_spike (gelir==30000, oran %4.4 vs %18.8),
       is_env_hater (env==1, oran %0.6), is_commute5 (commute==5.0)
  3) LGBM: max_bin=1024, colsample_bytree=0.3, max_depth=5,
     num_leaves=32, lr=0.02, ES 500 (megayak ayarlari), 10-fold.

Kullanim:
  python model_v10.py --seed 42
  python model_v10.py --seed 2024
  python model_v10.py --seed 7
Sure: ~20-25 dk / seed (10 fold + TE). State .v10_state_s{seed}/ ile resume.
Cikti: submission_v10_recipe-10fold-lgb_s{seed}.csv (seed'e ozel, cakisma yok)
Sonrasinda 3-seed ortalamasi icin: python v10_seedavg.py
"""
import argparse
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

TARGET = 'Will_Buy_EV'
N_SPLITS = 10
ORIG_PATH = 'source_data/EV_Adoption_and_Range_Anxiety_Dataset.csv'

CAT_COLS = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible',
            'Subsidy_Available', 'Range_Anxiety_Level']
INCOME = 'Annual_Income_USD'
COMMUTE = 'Daily_Commute_km'
ENV = 'Environmental_Concern_Level'

LGB_PARAMS = dict(n_estimators=8000, learning_rate=0.02, num_leaves=32,
                  max_depth=5, colsample_bytree=0.3, max_bin=1024)


def build_base_features(train, test):
    """Leak ailesi + smooth keys + bayraklar + frekanslar (TE haric)."""
    tr, te = train.copy(), test.copy()
    for df in (tr, te):
        inc = df[INCOME].round().astype(np.int64)
        c10 = (df[COMMUTE] * 10).round().astype(np.int64)
        df['c10'] = c10
        # basamaklar
        for p in range(6):
            df[f'inc_d{p}'] = ((inc // 10 ** p) % 10).astype(np.int8)
        for p in range(5):
            df[f'com_d{p}'] = ((c10 // 10 ** p) % 10).astype(np.int8)
        # moduli
        for m in (7, 9, 11, 13, 97):
            df[f'inc_m{m}'] = (inc % m).astype(np.int8)
        for m in (7, 9, 11, 13):
            df[f'com_m{m}'] = (c10 % m).astype(np.int8)
        # quant ladder
        for q in (50, 100, 250, 500, 1000, 2500, 5000):
            df[f'inc_q{q}'] = (inc // q).astype(np.int32)
        # smooth keys (string kategori)
        df['sk_inc'] = inc.astype(str)
        df['sk_inc100'] = (inc // 100).astype(str)
        df['sk_inc1000'] = (inc // 1000).astype(str)
        df['sk_com'] = c10.astype(str)
        # bayraklar
        df['is_30k_spike'] = (inc == 30000).astype(np.int8)
        df['is_env_hater'] = (df[ENV] == 1).astype(np.int8)
        df['is_commute5'] = (df[COMMUTE] == 5.0).astype(np.int8)

    # frekanslar (train+test concat, etiketsiz)
    n = len(tr) + len(te)
    for c in CAT_COLS + [INCOME, 'c10']:
        freq = pd.concat([tr[c], te[c]], ignore_index=True).value_counts()
        tr[f'{c}_freq'] = tr[c].map(freq).astype(np.float32) / n
        te[f'{c}_freq'] = te[c].map(freq).astype(np.float32) / n

    # orijinal veri hedef ortalamalari (gelir bazinda)
    orig = pd.read_csv(ORIG_PATH)
    g = orig.groupby(INCOME)[TARGET].agg(orig_cnt='count',
                                         orig_rate=lambda s: (s == 'Yes').mean())
    g = g.reset_index()
    for df in (tr, te):
        m = df[[INCOME]].merge(g, on=INCOME, how='left')
        df['orig_cnt'] = m['orig_cnt'].fillna(0).astype(np.int16)
        df['orig_rate'] = m['orig_rate'].fillna(0.175).astype(np.float32)
    return tr, te


TE_KEYS = CAT_COLS + ['sk_inc', 'sk_inc100', 'sk_inc1000', 'sk_com']
TE_SMOOTHS = ['auto', 10.0, 100.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()

    print('veri yukleniyor...')
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    y = (train[TARGET] == 'Yes').astype(int).values
    train_id, test_id = train['id'].values, test['id'].values

    print('taban ozellikler kuruluyor (leak + keys + bayraklar + frekans)...')
    train, test = build_base_features(train, test)

    te_key_cols = TE_KEYS  # string kolonlar modele kategori olarak gitmez, TE'den gecer
    num_cols = [c for c in train.columns
                if c not in ('id', TARGET) + tuple(te_key_cols) + tuple(CAT_COLS)]
    cat_for_lgb = list(CAT_COLS)
    print(f'numerik/ikili ozellik: {len(num_cols)} | TE: {len(TE_KEYS)} key x {len(TE_SMOOTHS)} smooth')

    state_dir = f'.v10_state_s{args.seed}'
    os.makedirs(state_dir, exist_ok=True)

    # kategori sozlukleri (train vocab)
    cats = {c: pd.Categorical(train[c]).categories for c in cat_for_lgb}

    oof = np.zeros(len(train))
    test_preds = np.zeros(len(test))
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=args.seed)
    import lightgbm as lgb

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        fp = f'{state_dir}/fold{fold}.npz'
        if os.path.exists(fp):
            print(f'fold {fold+1}/{N_SPLITS} -> ATLANDI (var)')
            d = np.load(fp)
            oof[va_idx] = d['oof']; test_preds += d['test'] / N_SPLITS
            continue
        X_tr = train.iloc[tr_idx]
        X_va = train.iloc[va_idx]

        print(f'fold {fold+1}/{N_SPLITS}: triple TE hesaplaniyor...')
        # her key x smooth icin: train (inner-CV OOF encode), valid, test kolonlari
        from sklearn.preprocessing import TargetEncoder
        tr_cols, va_cols, te_cols = [], [], []
        for k in TE_KEYS:
            for sm in TE_SMOOTHS:
                enc = TargetEncoder(target_type='binary', smooth=sm)
                tr_cols.append(enc.fit_transform(X_tr[[k]], y[tr_idx]).ravel().astype(np.float32))
                va_cols.append(enc.transform(X_va[[k]]).ravel().astype(np.float32))
                te_cols.append(enc.transform(test[[k]]).ravel().astype(np.float32))
        te_tr = np.vstack(tr_cols).T
        te_va = np.vstack(va_cols).T
        te_test = np.vstack(te_cols).T

        def assemble(X, te_cols):
            parts = [X[num_cols].reset_index(drop=True)]
            for c in cat_for_lgb:
                codes = pd.Categorical(X[c].reset_index(drop=True),
                                       categories=cats[c]).codes.astype(np.int16)
                parts.append(pd.Series(codes, name=c).to_frame())
            parts.append(pd.DataFrame(te_cols, columns=[f'te_{i}' for i in range(te_cols.shape[1])]))
            return pd.concat(parts, axis=1)

        A_tr = assemble(X_tr, te_tr)
        A_va = assemble(X_va, te_va)
        A_te = assemble(test, te_test)

        model = lgb.LGBMClassifier(random_state=args.seed, verbose=-1, **LGB_PARAMS)
        model.fit(A_tr, y[tr_idx], eval_set=[(A_va, y[va_idx])], eval_metric='auc',
                  callbacks=[lgb.early_stopping(500, verbose=False), lgb.log_evaluation(0)])
        oof_va = model.predict_proba(A_va)[:, 1]
        te_p = model.predict_proba(A_te)[:, 1]
        oof[va_idx] = oof_va
        test_preds += te_p / N_SPLITS
        np.savez_compressed(fp, oof=oof_va, test=te_p)
        print(f'fold {fold+1}/{N_SPLITS} | AUC: {roc_auc_score(y[va_idx], oof_va):.5f} | '
              f'best_iter: {model.best_iteration_}')

    auc = roc_auc_score(y, oof)
    print(f'\nLGBM OOF AUC: {auc:.5f}')
    sub = pd.DataFrame({'id': test_id, TARGET: test_preds})
    out_name = f'submission_v10_recipe-10fold-lgb_s{args.seed}.csv'
    sub.to_csv(out_name, index=False)
    np.savez_compressed(f'{state_dir}/full.npz', oof=oof, test=test_preds)
    print(f'{out_name} yazildi')


if __name__ == '__main__':
    main()
