# -*- coding: utf-8 -*-
"""Kendi modellerimiz uzerinde residual (hata) analizi + segment kesfi.

Egitim yok — sadece diskteki OOF tahminlerini okur:
  - .v10_3seed_full.npz  (LGBM 3-tohum rank-avg, OOF ~0.94623)
  - .v10x_state_s42/full.npz (XGB, ~0.94593)
  - .v10c_state_s42/full.npz (CatBoost, ~0.94465)

Yontem:
  1) 3 modelin OOF'unu rank uzayinda esit agirlikli ortala -> kendi stack'imiz
  2) residual r = y - p hesapla (r>0 -> model satin alma olasiligini BASTIRIYOR,
     r<0 -> model SISIRIYOR)
  3) r'yi gelir dilimleri / commute bantlari / kategorilere gore kutula,
     sistematik sapma olan bolgeleri raporla
  4) derinlik-4 karar agaci ile r'yi tahmin et -> kendi segmentlerimizi kesfet
  5) her segment icin duzeltme testi: OOF olasiliklarini segment icinde
     kaydir, AUC degisimini olc (kanit: gercek sinyal mi gurultu mu)

Kullanim: python analyze_residuals.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor, export_text

BASE = os.path.dirname(os.path.abspath(__file__))
TARGET = 'Will_Buy_EV'
rank01 = lambda a: (rankdata(a) / len(a)).astype(np.float32)

NUM_FOR_TREE = ['Annual_Income_USD', 'Daily_Commute_km', 'Age',
                'Environmental_Concern_Level']


def load_own_stack():
    z = np.load(os.path.join(BASE, '.v10_3seed_full.npz'))
    lgb_o, lgb_t = rank01(z['oof']), rank01(z['test'])
    z = np.load(os.path.join(BASE, '.v10x_state_s42', 'full.npz'))
    xgb_o, xgb_t = rank01(z['oof']), rank01(z['test'])
    z = np.load(os.path.join(BASE, '.v10c_state_s42', 'full.npz'))
    cat_o, cat_t = rank01(z['oof']), rank01(z['test'])
    oof = rank01((lgb_o + xgb_o + cat_o) / 3)
    test = rank01((lgb_t + xgb_t + cat_t) / 3)
    return oof, test


def clean_bins(series, bins):
    """Bin sinirlarini veri araligina uydur: tekillestir, max'i kapa."""
    mx = float(series.max())
    bins = sorted(set([b for b in bins if b < mx] + [mx + 1]))
    return bins


def bin_report(df, y, p, col, bins, labels=None):
    """Bir sayisal kolonu kutula, her kutuda: n, gercek oran, tahmin ort, sapma."""
    if labels is None:
        labels = [f'{int(bins[i])}-{int(bins[i+1])}' for i in range(len(bins) - 1)]
    b = pd.cut(df[col], bins=bins, labels=labels, include_lowest=True)
    rows = []
    for lab, idx in b.groupby(b, observed=True).groups.items():
        idx = np.asarray(idx)
        if len(idx) < 500:
            continue
        m = y[idx].mean()
        pm = p[idx].mean()
        se = np.sqrt(m * (1 - m) / len(idx))
        rows.append((str(lab), len(idx), m, pm, m - pm, (m - pm) / max(se, 1e-9)))
    rep = pd.DataFrame(rows, columns=['bin', 'n', 'gercek', 'tahmin', 'sapma', 'z_skor'])
    return rep


def main():
    print('veri + OOF yukleniyor...')
    train = pd.read_csv(os.path.join(BASE, 'train.csv'))
    y = (train[TARGET] == 'Yes').astype(int).values
    p, _ = load_own_stack()
    print(f'kendi 3-model rank-avg OOF AUC: {roc_auc_score(y, p):.5f}')

    r = y - p
    df = train.copy()

    # ---- 1) GELIR dilimleri ----
    inc = df['Annual_Income_USD']
    print('\n=== GELIR dilimleri (sapma = gercek - tahmin; z>2 dikkat) ===')
    qs = clean_bins(inc, [0, 10000, 20000, 25000, 30000, 35000, 40000, 45000,
                          50000, 60000, 75000, 100000, 125000, 150000, 175000,
                          200000, 250000])
    rep = bin_report(df, y, p, 'Annual_Income_USD', qs)
    rep[['gercek', 'tahmin', 'sapma']] = rep[['gercek', 'tahmin', 'sapma']].round(4)
    rep['z_skor'] = rep['z_skor'].round(1)
    print(rep.to_string(index=False))

    # ---- 2) COMMUTE bantlari ----
    print('\n=== COMMUTE bantlari ===')
    com = df['Daily_Commute_km']
    qs = clean_bins(com, [0, 5, 10, 15, 20, 30, 40, 50, 60, 75, 90, 120])
    rep = bin_report(df, y, p, 'Daily_Commute_km', qs)
    rep[['gercek', 'tahmin', 'sapma']] = rep[['gercek', 'tahmin', 'sapma']].round(4)
    rep['z_skor'] = rep['z_skor'].round(1)
    print(rep.to_string(index=False))

    # ---- 3) Kategoriler ----
    print('\n=== Kategoriler (seviye basina sapma) ===')
    for col in ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible',
                'Subsidy_Available', 'Range_Anxiety_Level',
                'Environmental_Concern_Level']:
        rows = []
        for lab, idx in df.groupby(col).groups.items():
            idx = np.asarray(idx)
            m, pm = y[idx].mean(), p[idx].mean()
            se = np.sqrt(m * (1 - m) / len(idx))
            rows.append((str(lab), len(idx), round(m, 4), round(pm, 4),
                         round(m - pm, 4), round((m - pm) / max(se, 1e-9), 1)))
        print(f'-- {col}')
        print(pd.DataFrame(rows, columns=['deger', 'n', 'gercek', 'tahmin',
                                          'sapma', 'z_skor']).to_string(index=False))

    # ---- 4) Karar agaci ile segment kesfi ----
    print('\n=== Karar agaci (residual tahmini, derinlik 4) ===')
    X = pd.get_dummies(df[NUM_FOR_TREE + ['Subsidy_Available', 'City_Type']],
                       columns=['Subsidy_Available', 'City_Type'], drop_first=True)
    tree = DecisionTreeRegressor(max_depth=4, min_samples_leaf=3000,
                                 random_state=42)
    tree.fit(X, r)
    print(export_text(tree, feature_names=list(X.columns), decimals=4))

    # yaprak bazli duzeltme testi
    leaf = tree.apply(X)
    oof_auc = roc_auc_score(y, p)
    gains = []
    for lf in np.unique(leaf):
        m = leaf == lf
        if m.sum() < 3000:
            continue
        shift = r[m].mean()
        p_fix = p.copy()
        p_fix[m] = np.clip(p_fix[m] + shift, 0, 1)
        auc_fix = roc_auc_score(y, p_fix)
        gains.append((lf, int(m.sum()), round(shift, 5), round(auc_fix - oof_auc, 6)))
    gains.sort(key=lambda g: -abs(g[3]))
    print(f'\nyaprak bazli duzeltme testi (baz OOF {oof_auc:.5f}):')
    print(pd.DataFrame(gains, columns=['leaf', 'n', 'kaydirma', 'auc_kazanc'])
          .head(10).to_string(index=False))

    # kombine: tum anlamli yapraklari ayni anda duzelt
    p_fix = p.copy()
    for lf, n, shift, g in gains:
        if abs(g) > 0:
            p_fix[leaf == lf] = np.clip(p_fix[leaf == lf] + r[leaf == lf].mean(), 0, 1)
    print(f'tum yapraklar birlikte: {roc_auc_score(y, p_fix):.5f} '
          f'(delta {roc_auc_score(y, p_fix) - oof_auc:+.6f})')


if __name__ == '__main__':
    main()
