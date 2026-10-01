# -*- coding: utf-8 -*-
"""Final own-stack blend — yalniz kendi modellerimiz.

Uyeler (hepsi kendi egitimimiz, public OOF yok):
  own_lgbm_6seed : .v10_6seed_full.npz      (OOF 0.94627)
  own_xgb        : .v10x_state_s42/full.npz (OOF 0.94593)
  own_cat        : .v10c_state_s42/full.npz (OOF 0.94465)
  own_nn         : .nn_state_s42/full.npz   (OOF 0.94094)

Yontem: rank uzayinda greedy agirlik aramasi (OOF uzerinde, iters=20).
Overfit'e karsi kisitli aday + sadece pozitif katkili uyeler.

Kullanim: python own_stack_final.py
Cikti: submission_ownstack-final.csv + .ownstack_full.npz
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
TARGET = 'Will_Buy_EV'
rank01 = lambda a: (rankdata(a) / len(a)).astype(np.float32)

MEMBERS = [
    ('lgbm_6seed', '.v10_6seed_full.npz', None),
    ('xgb', os.path.join('.v10x_state_s42', 'full.npz'), None),
    ('cat', os.path.join('.v10c_state_s42', 'full.npz'), None),
    ('nn', os.path.join('.nn_state_s42', 'full.npz'), None),
]


def load_members():
    M = {}
    for name, rel, _ in MEMBERS:
        z = np.load(os.path.join(BASE, rel))
        M[name] = (rank01(z['oof']), rank01(z['test']))
        print(f'{name:12s} yuklendi')
    return M


def main():
    train = pd.read_csv(os.path.join(BASE, 'train.csv'), usecols=[TARGET])
    y = (train[TARGET] == 'Yes').astype(int).values
    test_ids = pd.read_csv(os.path.join(BASE, 'test.csv'), usecols=['id'])['id'].values

    M = load_members()
    solo = {n: roc_auc_score(y, M[n][0]) for n in M}
    print('\nsolo OOF:')
    for n, a in sorted(solo.items(), key=lambda x: -x[1]):
        print(f'  {n:12s} {a:.5f}')

    # esit agirlik referansi
    eq = rank01(sum(M[n][0] for n in M) / len(M))
    print(f'\nesit agirlik rank-avg OOF: {roc_auc_score(y, eq):.5f}')

    # greedy: her adimda OOF'u en cok iyilestiren uyesi agirliga ekle
    names = list(M.keys())
    counts = {n: 0 for n in names}
    runsum = np.zeros(len(y), dtype=np.float64)
    k = 0
    hist = []
    for it in range(20):
        best_n, best_a = None, -1
        for n in names:
            a = roc_auc_score(y, (runsum + M[n][0]) / (k + 1))
            if a > best_a:
                best_a, best_n = a, n
        counts[best_n] += 1
        runsum += M[best_n][0]
        k += 1
        hist.append(best_a)
        print(f'  iter {it+1:2d}: + {best_n:10s} -> {best_a:.5f}')
    w = {n: c / k for n, c in counts.items() if c > 0}
    print('\ngreedy agirliklar:', {n: round(wi, 3) for n, wi in w.items()})

    # test blend
    test_blend = sum(wi * M[n][1] for n, wi in w.items())
    out = os.path.join(BASE, 'submission_ownstack-final.csv')
    pd.DataFrame({'id': test_ids, TARGET: test_blend}).to_csv(out, index=False)
    np.savez_compressed(os.path.join(BASE, '.ownstack_full.npz'),
                        oof=runsum / k, test=test_blend)
    print(f'{os.path.basename(out)} yazildi')


if __name__ == '__main__':
    main()
