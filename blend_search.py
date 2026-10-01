"""
Ağırlıklı blend arama — Predicting EV Purchases
.v3_state/*.npz fold tahminlerinden OOF AUC'ye gore en iyi agirliklari bulur.
Yeniden egitim YOK, sadece harmanlama.
"""
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

TARGET = 'Will_Buy_EV'
N_SPLITS = 5
SEED = 42
STATE_DIR = sys.argv[1] if len(sys.argv) > 1 else '.v3_state'
OUT_CSV = sys.argv[2] if len(sys.argv) > 2 else 'submission_v3_weighted.csv'

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
y = (train[TARGET] == 'Yes').astype(int).values
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

models = {}
for name, prefix in [('LGBM', 'lgb'), ('XGB', 'xgb'), ('CatBoost', 'cat')]:
    oof = np.zeros(len(train))
    preds = np.zeros(len(test))
    for fold, (tr, va) in enumerate(skf.split(train.drop(columns=[TARGET]), y)):
        d = np.load(f'{STATE_DIR}/{prefix}_fold{fold}.npz')
        oof[va] = d['oof']
        preds += d['test'] / N_SPLITS
    models[name] = (oof, preds)
    print(f'{name} OOF AUC: {roc_auc_score(y, oof):.5f}')

names = list(models.keys())
best = (None, -1)
results = []
# 5%'lik adımlarla 3 model agirlik izgarasi (toplamlari 1)
for w0 in range(0, 101, 5):
    for w1 in range(0, 101 - w0, 5):
        w2 = 100 - w0 - w1
        ws = [w0 / 100, w1 / 100, w2 / 100]
        oof_blend = sum(w * models[n][0] for w, n in zip(ws, names))
        auc = roc_auc_score(y, oof_blend)
        results.append((auc, tuple(ws)))
        if auc > best[1]:
            best = (tuple(ws), auc)

results.sort(reverse=True)
print('\nEn iyi 10 kombinasyon:')
for auc, ws in results[:10]:
    print(f'  AUC {auc:.5f} | LGBM {ws[0]:.2f} + XGB {ws[1]:.2f} + CatBoost {ws[2]:.2f}')

ws, auc = best
test_blend = sum(w * models[n][1] for w, n in zip(ws, names))
sub = pd.DataFrame({'id': test['id'], TARGET: test_blend})
sub.to_csv(OUT_CSV, index=False)
print(f'\nEn iyi: LGBM {ws[0]:.2f} + XGB {ws[1]:.2f} + CatBoost {ws[2]:.2f} | OOF AUC {auc:.5f}')
print(f'{OUT_CSV} yazildi.')
