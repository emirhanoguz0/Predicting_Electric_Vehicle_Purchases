"""
EDA - Predicting Electric Vehicle Purchases (Playground S6E9)
Keşif analizi: hedef dağılımı, eksik değerler, kategorik kardinallik,
sayısal özet, hedefle ilişki, tek değişkenli ayırt edicilik.
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

pd.set_option('display.width', 160)
pd.set_option('display.max_columns', 50)

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
target_col = 'Will_Buy_EV'

print('=' * 70)
print('1. GENEL BAKIS')
print('=' * 70)
print(f'Train şekli: {train.shape} | Test şekli: {test.shape}')
print(f'\nSütun tipleri:\n{train.dtypes.value_counts()}')

print('\n' + '=' * 70)
print('2. HEDEF DEGISKEN DAGILIMI')
print('=' * 70)
vc = train[target_col].value_counts(normalize=True).round(4)
print(vc)
print(f'Sinif dengesi: {vc.max()/vc.min():.2f}x (1.0 = tamamen dengeli)')

print('\n' + '=' * 70)
print('3. EKSIK DEGERLER')
print('=' * 70)
miss = train.isna().sum()
miss = miss[miss > 0].sort_values(ascending=False)
if len(miss):
    print(pd.DataFrame({'eksik_sayi': miss, 'oran_%': (miss/len(train)*100).round(2)}))
else:
    print('Eksik deger YOK.')

print('\n' + '=' * 70)
print('4. KATEGORIK KOLONLAR (kardinallik)')
print('=' * 70)
cat_cols = train.select_dtypes(include=['object', 'category']).columns.tolist()
cat_cols = [c for c in cat_cols if c != target_col]
card = train[cat_cols].nunique().sort_values(ascending=False)
print(pd.DataFrame({'unique_sayi': card, 'ornek_deger': [str(train[c].unique()[:4]) for c in card.index]}))

print('\n' + '=' * 70)
print('5. SAYISAL KOLONLAR (ozet)')
print('=' * 70)
num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
num_cols = [c for c in num_cols if c != 'id']
desc = train[num_cols].describe().T.round(3)
print(desc[['mean', 'std', 'min', '25%', '50%', '75%', 'max']])

print('\n' + '=' * 70)
print('6. HEDEFLE ILISKI — KATEGORIK (target mean farki)')
print('=' * 70)
rows = []
for c in cat_cols:
    g = train.groupby(c)[target_col].apply(lambda s: (s == 'Yes').mean())
    diff = g.max() - g.min()
    rows.append((c, len(g), round(diff, 4), g.idxmax(), round(g.max(), 4), g.idxmin(), round(g.min(), 4)))
cat_rel = pd.DataFrame(rows, columns=['kolon', 'kategori_sayi', 'max_min_fark', 'en_yuksek_kat', 'en_yuksek_oran', 'en_dusuk_kat', 'en_dusuk_oran'])
print(cat_rel.sort_values('max_min_fark', ascending=False).head(15).to_string(index=False))

print('\n' + '=' * 70)
print('7. HEDEFLE ILISKI — SAYISAL (tek degiskenli AUC)')
print('=' * 70)
y_bin = (train[target_col] == 'Yes').astype(int)
rows = []
for c in num_cols:
    auc = roc_auc_score(y_bin, train[c])
    auc = max(auc, 1 - auc)  # yonunden bagimsiz
    rows.append((c, round(auc, 4), round(train.loc[y_bin==1, c].mean(), 3), round(train.loc[y_bin==0, c].mean(), 3)))
num_rel = pd.DataFrame(rows, columns=['kolon', 'tek_degiskenli_AUC', 'Yes_ort', 'No_ort'])
print(num_rel.sort_values('tek_degiskenli_AUC', ascending=False).head(15).to_string(index=False))

print('\n' + '=' * 70)
print('8. TRAIN-TEST DAGILIM KONTROLU (drift sinyali)')
print('=' * 70)
rows = []
for c in num_cols:
    m_tr, m_te = train[c].mean(), test[c].mean()
    s_tr = train[c].std() + 1e-9
    drift = abs(m_tr - m_te) / s_tr
    rows.append((c, round(m_tr, 3), round(m_te, 3), round(drift, 3)))
drift_df = pd.DataFrame(rows, columns=['kolon', 'train_ort', 'test_ort', 'drift_z'])
print(drift_df.sort_values('drift_z', ascending=False).head(10).to_string(index=False))

print('\nEDA tamamlandi.')
