# -*- coding: utf-8 -*-
"""Kolon bazinda uyum + alt-kume eslesme aramasi."""
import pandas as pd
import numpy as np

BASE = r"D:\Second_Brain\🏰 300-Projects\Makine-Ogrenmesi\Predicting_Electric_Vehicle_Purchases"
train = pd.read_csv(BASE + r"\train.csv")
test = pd.read_csv(BASE + r"\test.csv")
orig = pd.read_csv(BASE + r"\source_data\EV_Adoption_and_Range_Anxiety_Dataset.csv")

FEATURES = [c for c in train.columns if c not in ("id", "Will_Buy_EV")]

print("=== Kolon bazinda: sentetik degerlerin orijinalde gorulme orani ===")
for c in FEATURES:
    if train[c].dtype == object:
        o = set(orig[c].dropna().astype(str))
        tr = train[c].astype(str).isin(o).mean()
        te = test[c].astype(str).isin(o).mean()
        print(f"{c:35s} kategorik  train {tr:.1%}  test {te:.1%}")
    else:
        o = set(orig[c].dropna().round(1))
        tr = train[c].round(1).isin(o).mean()
        te = test[c].round(1).isin(o).mean()
        print(f"{c:35s} sayisal    train {tr:.1%}  test {te:.1%}")

# Alt-kume eslesme: sadece kategorik kolonlar
cat = [c for c in FEATURES if train[c].dtype == object]
print("\nkategorik kolonlar:", cat)

def catkey(df, cols):
    d = df[cols].copy()
    for c in cols:
        d[c] = d[c].map(lambda x: "<NA>" if pd.isna(x) else str(x))
    return d.agg("|".join, axis=1)

# Orijinal kombinasyonlar tekil mi?
ok = catkey(orig, cat)
print("orig kategorik kombinasyon tekil mi:", ok.nunique(), "/", len(ok))

om = orig.assign(_k=ok).drop_duplicates("_k").set_index("_k")
trm = train.assign(_k=catkey(train, cat))["_k"].map(om["Will_Buy_EV"])
tem = test.assign(_k=catkey(test, cat))["_k"].map(om["Will_Buy_EV"])
print(f"kategorik tam eslesme train: {trm.notna().sum()} ({trm.notna().mean():.1%})")
print(f"kategorik tam eslesme test : {tem.notna().sum()} ({tem.notna().mean():.1%})")

if trm.notna().sum() > 100:
    m = trm.notna()
    agree = (train.loc[m, "Will_Buy_EV"].values == trm[m].values).mean()
    print(f"train eslesenlerde hedef uyumu: {agree:.2%}")
    print("eslesenler orijinal hedef dagilimi:", trm[m].value_counts(normalize=True).to_dict())
