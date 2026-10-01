# -*- coding: utf-8 -*-
"""OOF-stacked greedy blend — golem + najiama + megayak + kendi v9/v10'umuz.

Frozen partition: StratifiedKFold(5, shuffle, rs=42), train.csv ORIGINAL ORDER.
Tum OOF'ler pozisyonel (train.csv sirasi); id'li CSV'ler id'ye gore hizalanir.

Kullanim:
  python stack_blend.py                       # klasik: 30 iter, top-12 aday
  python stack_blend.py --iters 40 --top-k 30 --out deneme.csv
"""
import argparse
import json
import os
import glob
import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE = os.path.dirname(os.path.abspath(__file__))
PS = os.path.join(BASE, 'public_sources')
TARGET = 'Will_Buy_EV'
rank01 = lambda a: (rankdata(a) / len(a)).astype(np.float32)


def load_y_and_ids():
    tr = pd.read_csv(os.path.join(BASE, 'train.csv'), usecols=['id', TARGET])
    te = pd.read_csv(os.path.join(BASE, 'test.csv'), usecols=['id'])
    y = (tr[TARGET] == 'Yes').astype(int).values
    return y, tr['id'].values, te['id'].values, tr


def collect_members(y, train_ids, test_ids):
    """-> dict[name] = (oof_rank, test_rank)"""
    from sklearn.metrics import roc_auc_score
    M = {}

    def report(name, oof, test):
        M[name] = (rank01(oof), rank01(test))
        print(f'  {name:28s} OOF AUC: {roc_auc_score(y, oof):.5f}')

    # --- golem kutuphanesi ---
    gdir = os.path.join(PS, 'dariushafshar_s6e9-golem-oof-library')
    for f in sorted(glob.glob(os.path.join(gdir, 'oof_*.npy'))):
        name = 'golem_' + os.path.basename(f)[4:-4]
        test_f = os.path.join(gdir, 'test_' + os.path.basename(f)[4:])
        if os.path.exists(test_f):
            report(name, np.load(f), np.load(test_f))

    # --- megayak digit-leak + hybrid ---
    mdir = os.path.join(PS, 'megayak_s6e9-digit-leak-oof')
    for m in ('lgb', 'cat', 'xgb'):
        report(f'megayak_digit_{m}', np.load(os.path.join(mdir, f'oof_{m}.npy')),
               np.load(os.path.join(mdir, f'test_{m}.npy')))
    hdir = os.path.join(PS, 'megayak_s6e9-hybrid-lgbm-oof')
    report('megayak_hybrid_3seed', np.load(os.path.join(hdir, 'oof_hybrid_3seed.npy')),
           np.load(os.path.join(hdir, 'test_hybrid_3seed.npy')))

    # --- najiama (id'li) ---
    ndir = os.path.join(PS, 'najiama_s6e9-oof')
    id_to_pos = pd.Series(np.arange(len(train_ids)), index=train_ids)
    te_pos = pd.Series(np.arange(len(test_ids)), index=test_ids)
    naji_map = {
        'najiama_V1': ('Pure LGBM_V1_oof.csv', 'OOF_Pred', 'Pure LGBM_V1_test.csv'),
        'najiama_V3': ('Pure LGBM_V3_oof.csv', 'OOF_Pred', 'Pure LGBM_V3_test.csv'),
        'najiama_V5': ('Pure LGBM_V5_oof.csv', 'OOF_Pred', 'Pure LGBM_V5_test.csv'),
        'najiama_V6': ('Pure LGBM_V6_oof.csv', 'OOF_Pred', 'Pure LGBM_V6_test.csv'),
        'najiama_blend': ('01_blend_oof.csv', 'Will_Buy_EV', '01_submission.csv'),
        'najiama_sergey': ('Sergey_LGBM_oof.csv', 'WBE', 'Sergey_LGBM_submission.csv'),
    }
    for name, (oof_f, oof_col, test_f) in naji_map.items():
        o = pd.read_csv(os.path.join(ndir, oof_f))
        t = pd.read_csv(os.path.join(ndir, test_f))
        pc = 'Will_Buy_EV' if 'Will_Buy_EV' in o.columns else oof_col
        oof = np.full(len(train_ids), np.nan)
        pos = id_to_pos.loc[o['id'].values].values
        oof[pos] = o[pc].values
        assert not np.isnan(oof).any(), name
        tt = np.full(len(test_ids), np.nan)
        tpos = te_pos.loc[t['id'].values].values
        tt[tpos] = t[TARGET].values
        assert not np.isnan(tt).any(), name
        report(name, oof, tt)

    # --- kendi v9 / v10 / v10xgb / v10cat ---
    z = np.load(os.path.join(BASE, '.v9_state_s42', 'lgb_full.npz'))
    report('own_v9_s42', z['oof'], z['test'])
    p3 = os.path.join(BASE, '.v10_3seed_full.npz')
    if os.path.exists(p3):
        z = np.load(p3)
        report('own_v10_3seed', z['oof'], z['test'])
    else:
        z = np.load(os.path.join(BASE, '.v10_state_s42', 'full.npz'))
        report('own_v10_s42', z['oof'], z['test'])
    for nm, sd in (('own_xgb_s42', '.v10x_state_s42'), ('own_cat_s42', '.v10c_state_s42')):
        p = os.path.join(BASE, sd, 'full.npz')
        if os.path.exists(p):
            z = np.load(p)
            report(nm, z['oof'], z['test'])

    return M


def greedy(M, y, iters=30, top_k=12):
    from sklearn.metrics import roc_auc_score
    names = list(M.keys())
    solo = {n: roc_auc_score(y, M[n][0]) for n in names}
    cands = sorted(names, key=lambda n: -solo[n])[:top_k]
    print('\nsolo OOF sirasi (ilk %d aday):' % top_k)
    for n in cands:
        print(f'  {n:28s} {solo[n]:.5f}')

    counts = {n: 0 for n in cands}
    runsum = np.zeros(len(y), dtype=np.float32)
    k = 0
    best_hist = []
    for it in range(iters):
        best_n, best_a = None, -1
        for n in cands:
            a = roc_auc_score(y, (runsum + M[n][0]) / (k + 1))
            if a > best_a:
                best_a, best_n = a, n
        counts[best_n] += 1
        runsum += M[best_n][0]
        k += 1
        best_hist.append(best_a)
        print(f'  iter {it+1:2d}: + {best_n:24s} -> OOF {best_a:.5f}')
    w = {n: c / k for n, c in counts.items() if c > 0}
    return w, best_hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iters', type=int, default=30)
    ap.add_argument('--top-k', type=int, default=12)
    ap.add_argument('--out', default='submission_stack.csv')
    args = ap.parse_args()
    print('veri yukleniyor...')
    y, train_ids, test_ids, _ = load_y_and_ids()
    print('uyeler yukleniyor...')
    M = collect_members(y, train_ids, test_ids)
    print(f'{len(M)} uye yuklendi. greedy basliyor (iters={args.iters}, top_k={args.top_k})...')
    w, hist = greedy(M, y, iters=args.iters, top_k=args.top_k)

    test_blend = np.zeros(len(test_ids), dtype=np.float64)
    for n, wi in w.items():
        test_blend += wi * M[n][1]
    out = pd.DataFrame({'id': test_ids, TARGET: test_blend})
    outp = os.path.join(BASE, args.out)
    out.to_csv(outp, index=False)
    print(f'{os.path.basename(outp)} yazildi')
    print('agirliklar:', json.dumps({k: round(v, 3) for k, v in sorted(w.items(), key=lambda x: -x[1])}, indent=1))
    with open(os.path.join(BASE, 'stack_weights.json'), 'w') as f:
        json.dump({'weights': w, 'history': hist}, f, indent=1)


if __name__ == '__main__':
    main()
