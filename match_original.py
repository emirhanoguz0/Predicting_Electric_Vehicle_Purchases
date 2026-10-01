# -*- coding: utf-8 -*-
"""Sentetik train/test satirlarini orijinal 10K kaynak setle eslestir.

SDGM sentetik veri uretirken orijinalin sayisal/dijital kaliplarini
koruyabiliyor. Tam eslesen satirlarin orijinal hedefini geri kazaniyoruz.
"""
import pandas as pd
import numpy as np

BASE = r"D:\Second_Brain\🏰 300-Projects\Makine-Ogrenmesi\Predicting_Electric_Vehicle_Purchases"

train = pd.read_csv(BASE + r"\train.csv")
test = pd.read_csv(BASE + r"\test.csv")
orig = pd.read_csv(BASE + r"\source_data\EV_Adoption_and_Range_Anxiety_Dataset.csv")

print("orig:", orig.shape, "| train:", train.shape, "| test:", test.shape)
print("orig hedef dagilimi:\n", orig["Will_Buy_EV"].value_counts(normalize=True))

FEATURES = [c for c in train.columns if c not in ("id", "Will_Buy_EV")]
print("\nortak kolonlar:", all(c in orig.columns for c in FEATURES))

# Eslesme anahtari: tum feature kolonlarinin string konkatenasyonu
def key(df):
    d = df[FEATURES].copy()
    for c in FEATURES:
        d[c] = d[c].map(lambda x: "<NA>" if pd.isna(x) else str(x))
    return d.agg("|".join, axis=1)

orig_key = key(orig)
train_key = key(train)
test_key = key(test)

# Orijinal satirlar tekil mi?
print("orig tekil key:", orig_key.nunique(), "/", len(orig_key))

orig_target = orig.set_index(orig_key)["Will_Buy_EV"]

tr_match = train_key.map(orig_target)
te_match = test_key.map(orig_target)

print(f"\nTAM ESLESME  train: {tr_match.notna().sum()}/{len(train)} ({tr_match.notna().mean():.1%})")
print(f"TAM ESLESME  test : {te_match.notna().sum()}/{len(test)} ({te_match.notna().mean():.1%})")

# Eslesen satirlarda sentetik hedef ile orijinal hedef tutarliligi
tr_m = tr_match.notna()
if tr_m.sum() > 0:
    agree = (train.loc[tr_m, "Will_Buy_EV"].values == tr_match[tr_m].values).mean()
    print(f"train eslesenlerde hedef uyumu: {agree:.2%}  (1.0 = sentetik hedef = orijinal hedef)")

    # Kac farkli orijinal satira kac sentetik satir dusuyor?
    matched_orig_keys = train_key[tr_m].value_counts()
    print("bir orijinal satira dusen max sentetik sayisi:", matched_orig_keys.max())
    print("eslesen orijinal satir sayisi:", matched_orig_keys.nunique())

# Test tarafinda: eslesenlerin orijinal hedef dagilimi
if te_match.notna().sum() > 0:
    print("\ntest eslesenler orijinal hedef dagilimi:\n", te_match.value_counts(normalize=True))
