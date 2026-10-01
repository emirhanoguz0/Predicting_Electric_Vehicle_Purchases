# -*- coding: utf-8 -*-
"""Final combo: kendi v12 blend'imiz + defiaudit champion + segment kurallari.

Yapi (defiaudit r7/r8 public deney kanitlarina dayanir):
  base   = rank01(0.5*rank01(v12) + 0.5*rank01(r6_champ_94651))
  kural  = +10 (gelir>=170537) | -10 (gelir 31004-41970)
           | -5 (commute>=83)  | -5 (gelir==30000 & teşvik yok & (env1|anx M/H))
  cikti  = base + kural (hard block: secili satirlar tum siralarin üstüne/altina oynar)

Kaynak kanit: defiaudit r8_results.csv — kurallarin tersi LB'de -3..-181 bp kaybettiriyor,
yani segmentlerde gercek sinyal var. r8_m50 (champ+mega 50/50) = 0.94657.
Kullanim: python build_final_rules.py
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE = os.path.dirname(os.path.abspath(__file__))
PS = os.path.join(BASE, 'public_sources', 'defiaudit_s6e9-realmlp-oof')
ID, TARGET = 'id', 'Will_Buy_EV'
rank01 = lambda v: rankdata(np.asarray(v, dtype=float), method='average') / len(v)


def main():
    te = pd.read_csv(os.path.join(BASE, 'test.csv')).sort_values(ID).reset_index(drop=True)
    v12 = pd.read_csv(os.path.join(BASE, 'submission_stack_v12.csv')).sort_values(ID).reset_index(drop=True)
    ch = pd.read_csv(os.path.join(PS, 'r6_rebuilt_94651.csv')).sort_values(ID).reset_index(drop=True)
    assert (te[ID].to_numpy() == v12[ID].to_numpy()).all()
    assert (te[ID].to_numpy() == ch[ID].to_numpy()).all()

    base = 0.5 * rank01(v12[TARGET].to_numpy()) + 0.5 * rank01(ch[TARGET].to_numpy())

    inc = te['Annual_Income_USD'].to_numpy(float)
    km = te['Daily_Commute_km'].to_numpy(float)
    env1 = te['Environmental_Concern_Level'].to_numpy() == 1
    no_sub = te['Subsidy_Available'].astype(str).to_numpy() == 'No'
    anx = te['Range_Anxiety_Level'].astype(str).to_numpy()
    anx_mh = np.isin(anx, ['Medium', 'High'])

    v = base.copy()
    rules = [
        ('gelir>=170537 +10', (inc >= 170537), +10.0),
        ('gelir 31004-41970 -10', ((inc >= 31004) & (inc <= 41970)), -10.0),
        ('commute>=83 -5', (km >= 83), -5.0),
        ('30k&nosub&(env1|anxMH) -5', ((inc == 30000) & no_sub & (env1 | anx_mh)), -5.0),
    ]
    for name, m, d in rules:
        print(f'  {name:28s} satir={int(m.sum()):6d}')
        v = np.where(m, v + d, v)

    out = os.path.join(BASE, 'submission_final_v13_own50_champ_rules.csv')
    pd.DataFrame({ID: te[ID].to_numpy(), TARGET: v}).to_csv(out, index=False)
    print(f'{os.path.basename(out)} yazildi')


if __name__ == '__main__':
    main()
