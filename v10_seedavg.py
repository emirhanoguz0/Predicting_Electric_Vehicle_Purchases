# -*- coding: utf-8 -*-
"""v10 3-seed rank ortalamasi (megayak'in yontemi: rank-mean of seeds).

.v10_state_s{42,2024,7}/full.npz dosyalarini okur, OOF ve test tahminlerini
rank uzayinda ortalar, OOF AUC'yi raporlar ve submission uretir.

Kullanim: python v10_seedavg.py
Cikti: submission_v10_recipe-10fold-lgb_3seed-rankavg.csv + .v10_3seed_full.npz
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
TARGET = 'Will_Buy_EV'
SEEDS = [42, 2024, 7, 1, 13, 99]
rank01 = lambda a: (rankdata(a) / len(a)).astype(np.float32)


def main():
    oofs, tests = [], []
    for s in SEEDS:
        p = os.path.join(BASE, f'.v10_state_s{s}', 'full.npz')
        if not os.path.exists(p):
            raise SystemExit(f'EKSIK: {p} — once model_v10.py --seed {s} calistir.')
        z = np.load(p)
        oofs.append(rank01(z['oof']))
        tests.append(rank01(z['test']))
        print(f'seed {s}: yuklendi')
    oof = np.mean(oofs, axis=0)
    test = np.mean(tests, axis=0)

    train = pd.read_csv(os.path.join(BASE, 'train.csv'), usecols=[TARGET])
    y = (train[TARGET] == 'Yes').astype(int).values
    auc = roc_auc_score(y, oof)
    n_seeds = len(SEEDS)
    print(f'\n{n_seeds}-seed rank-avg OOF AUC: {auc:.5f}')

    te = pd.read_csv(os.path.join(BASE, 'test.csv'), usecols=['id'])
    out = os.path.join(BASE, f'submission_v10_recipe-10fold-lgb_{n_seeds}seed-rankavg.csv')
    pd.DataFrame({'id': te['id'].values, TARGET: test}).to_csv(out, index=False)
    np.savez_compressed(os.path.join(BASE, f'.v10_{n_seeds}seed_full.npz'), oof=oof, test=test)
    print(f'{os.path.basename(out)} yazildi')


if __name__ == '__main__':
    main()
