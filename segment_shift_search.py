# -*- coding: utf-8 -*-
"""Segment duzeltme grid-search — kendi kesfettigimiz anomali bolgeleri.

analyze_residuals.py'nin bulgularina dayanir:
  - 35-42k gelir bandinda model asiri tahmin yapiyor (baska notebook'ta da
    bagimsiz bulunmus -> gercek sinyal)
  - 40-50k bandinda hafif ayni yon
  - commute >= 75 ve Range_Anxiety=High kucuk ama sapmali

Her segment icin rank-uzayi kaymasini grid-search ile arar, OOF AUC kazancini
raporlar. Sadece OOF ile oynar, egitim yok.

Kullanim: python segment_shift_search.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
TARGET = 'Will_Buy_EV'
rank01 = lambda a: (rankdata(a) / len(a)).astype(np.float32)


def seg_income_3542(df):
    return (df['Annual_Income_USD'] >= 31004) & (df['Annual_Income_USD'] <= 41970)


def seg_income_4050(df):
    return (df['Annual_Income_USD'] > 41970) & (df['Annual_Income_USD'] <= 50000)


def seg_income_top(df):
    return df['Annual_Income_USD'] >= 170000


def seg_commute75(df):
    return df['Daily_Commute_km'] >= 83


def seg_anx_high(df):
    return df['Range_Anxiety_Level'] == 'High'


def seg_spike30k(df):
    return (df['Annual_Income_USD'] == 30000)


SEGMENTS = [
    ('gelir_31-42k', seg_income_3542),
    ('gelir_42-50k', seg_income_4050),
    ('gelir_170k+', seg_income_top),
    ('commute_83+', seg_commute75),
    ('anxiety_high', seg_anx_high),
    ('gelir_tam_30k', seg_spike30k),
]

SHIFTS = np.round(np.arange(-0.06, 0.061, 0.005), 3)


def main():
    print('veri + OOF yukleniyor...')
    train = pd.read_csv(os.path.join(BASE, 'train.csv'))
    y = (train[TARGET] == 'Yes').astype(int).values

    z = np.load(os.path.join(BASE, '.v10_3seed_full.npz'))
    lgb_o = rank01(z['oof'])
    z = np.load(os.path.join(BASE, '.v10x_state_s42', 'full.npz'))
    xgb_o = rank01(z['oof'])
    z = np.load(os.path.join(BASE, '.v10c_state_s42', 'full.npz'))
    cat_o = rank01(z['oof'])
    p = rank01((lgb_o + xgb_o + cat_o) / 3)
    base_auc = roc_auc_score(y, p)
    print(f'baz OOF AUC: {base_auc:.5f}')

    print(f'\n{"segment":16s} {"n":>8s} {"oran":>7s} {"tahmin":>7s} '
          f'{"en_iyi":>8s} {"auc_kazanc":>10s}')
    winners = {}
    for name, fn in SEGMENTS:
        m = fn(train).values
        n = int(m.sum())
        if n == 0:
            continue
        rate, pred = y[m].mean(), p[m].mean()
        best_s, best_a = 0.0, base_auc
        for s in SHIFTS:
            p2 = p.copy()
            p2[m] = np.clip(p2[m] + s, 0, 1)
            a = roc_auc_score(y, p2)
            if a > best_a:
                best_a, best_s = a, s
        delta = best_a - base_auc
        winners[name] = (m, best_s, delta)
        flag = ' <<<' if delta > 0.00002 else ''
        print(f'{name:16s} {n:8d} {rate:7.4f} {pred:7.4f} '
              f'{best_s:8.3f} {delta:+10.6f}{flag}')

    # kazanan segmentleri birlikte uygula
    p_fix = p.copy()
    for name, (m, s, d) in winners.items():
        if d > 0.00002 and s != 0.0:
            p_fix[m] = np.clip(p_fix[m] + s, 0, 1)
    auc_fix = roc_auc_score(y, p_fix)
    print(f'\nbirlikte uygulanan kazananlar: {auc_fix:.5f} '
          f'(delta {auc_fix - base_auc:+.6f})')

    # sinyal kararliligi: her modelin OOF'u ayri ayri test et
    print('\nmodel bazinda kararlilik (kazanan segmentler ayni yone calisiyor mu):')
    for nm, o in (('lgbm_3seed', lgb_o), ('xgb', xgb_o), ('cat', cat_o)):
        po = rank01(o)
        au = roc_auc_score(y, po)
        pf = po.copy()
        for name, (m, s, d) in winners.items():
            if d > 0.00002 and s != 0.0:
                pf[m] = np.clip(pf[m] + s, 0, 1)
        print(f'  {nm:12s} baz {au:.5f} -> duzeltmeli {roc_auc_score(y, pf):.5f} '
              f'({roc_auc_score(y, pf) - au:+.6f})')


if __name__ == '__main__':
    main()
