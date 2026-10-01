# -*- coding: utf-8 -*-
"""Public kaynaklarla rank-average blend (talhatursun tarifi).

Kaynak: megayak'in "s6e9-0-94644-plus-a-from-scratch-lightgbm" notebook'undaki
gunluk blend tarifi (LB 0.94643/0.94644). Gerekli CSV'ler proje klasorunde:

  nina.csv, zoomzoom.csv, kospintr.csv, mikhail.csv,
  najiama.csv, realmlp.csv, hybrid.csv  (hepsi id,Will_Buy_EV kolonlu)

Tarif:
  base  = 0.5*rank(nina) + 0.5*rank(zoomzoom)
  E     = 0.9*base + 0.05*rank(kospintr) + 0.05*rank(mikhail)
  F     = 0.5*rank(E) + 0.5*rank(najiama)
  daily = rank(0.85*rank(F) + 0.15*rank(realmlp))
  final = rank((1-W)*daily + W*rank(hybrid))   W=0.20

Kullanim:
  python blend_public.py                # W=0.20, submission_blend_public.csv
  python blend_public.py --w 0.0        # hybrid'siz gunluk tarif
  python blend_public.py --own v9       # kendi test tahminimizi ek uye yap
"""
import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

ID, TARGET = "id", "Will_Buy_EV"
BASE = os.path.dirname(os.path.abspath(__file__))
rank01 = lambda s: rankdata(s) / len(s)


def load(name):
    p = os.path.join(BASE, name)
    if not os.path.exists(p):
        raise SystemExit(f"EKSIK DOSYA: {p} — indirip proje klasorune at.")
    df = pd.read_csv(p).sort_values(ID)
    print(f"  {name}: {len(df)} satir")
    return df[TARGET].to_numpy(), df[ID].to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--w', type=float, default=0.20, help='hybrid agirligi')
    ap.add_argument('--own', default=None, help='state klasoru oneki (orn. v9) — kendi tahminimizi uye yap')
    args = ap.parse_args()

    print("kaynaklar yukleniyor:")
    nina, test_id = load('nina.csv')
    zoomzoom, _ = load('zoomzoom.csv')
    kospintr, _ = load('kospintr.csv')
    mikhail, _ = load('mikhail.csv')
    najiama, _ = load('najiama.csv')
    realmlp, _ = load('realmlp.csv')
    hybrid, _ = load('hybrid.csv')

    base = 0.5 * rank01(nina) + 0.5 * rank01(zoomzoom)
    E = 0.9 * base + 0.05 * rank01(kospintr) + 0.05 * rank01(mikhail)
    F = 0.5 * rank01(E) + 0.5 * rank01(najiama)
    daily = rank01(0.85 * rank01(F) + 0.15 * rank01(realmlp))

    print(f"\nhybrid vs daily spearman = {spearmanr(rank01(hybrid), daily).statistic:.5f}")

    if args.own:
        import glob
        cands = sorted(glob.glob(os.path.join(BASE, f'.{args.own}_state_s*', 'lgb_full.npz')))
        if not cands:
            raise SystemExit(f'kendi state bulunamadi: .{args.own}_state_s*/lgb_full.npz')
        own_test = np.mean([np.load(c)['test'] for c in cands], axis=0)
        print(f"kendi tahminim: {len(cands)} state ortalandi | daily ile spearman = "
              f"{spearmanr(rank01(own_test), daily).statistic:.5f}")
        # kendi tahminimizi hafif uye yap (daily tarifini bozmadan)
        daily = rank01(0.9 * rank01(daily) + 0.1 * rank01(own_test))

    final = rank01((1 - args.w) * daily + args.w * rank01(hybrid))
    out = pd.DataFrame({ID: test_id, TARGET: final})
    outp = os.path.join(BASE, 'submission_blend_public.csv')
    out.to_csv(outp, index=False)
    print(f"\n{os.path.basename(outp)} yazildi | W={args.w}"
          + (" | own eklendi" if args.own else ""))


if __name__ == '__main__':
    main()
