# -*- coding: utf-8 -*-
"""Sentetik satirlara en yakin orijinal satiri bul (ornekleme ile).

Kodlama: kategorikler one-hot (ayni kategori => 0 mesafe), sayisallar
z-score. Nearest neighbor mesafesi kucukse ve hedef uyumu taban oranin
cok ustundeyse, sentetik satirlar orijinalin gurultulu kopyasidir ve
orijinal hedefi gercek etiket olarak kullanabiliriz.
"""
import pandas as pd
import numpy as np

rng = np.random.default_rng(42)
BASE = r"D:\Second_Brain\🏰 300-Projects\Makine-Ogrenmesi\Predicting_Electric_Vehicle_Purchases"
train = pd.read_csv(BASE + r"\train.csv")
test = pd.read_csv(BASE + r"\test.csv")
orig = pd.read_csv(BASE + r"\source_data\EV_Adoption_and_Range_Anxiety_Dataset.csv")

FEATURES = [c for c in train.columns if c not in ("id", "Will_Buy_EV")]
NUM = [c for c in FEATURES if pd.api.types.is_numeric_dtype(train[c])]
CAT = [c for c in FEATURES if c not in NUM]
print("sayisal:", NUM)
print("kategorik:", CAT)

def encode(df):
    parts = []
    for c in NUM:
        mu, sd = train[c].mean(), train[c].std()
        parts.append(((df[c].fillna(train[c].median()) - mu) / sd).to_numpy(np.float32).reshape(-1, 1))
    for c in CAT:
        cats = sorted(orig[c].dropna().unique().tolist())
        m = df[c].map(lambda x: cats.index(x) if x in cats else -1).to_numpy()
        oh = np.zeros((len(df), len(cats)), np.float32)
        oh[np.arange(len(df)), np.clip(m, 0, len(cats) - 1)] = 1.0
        oh[m < 0] = 0.0
        parts.append(oh)
    return np.hstack(parts)

Xo = encode(orig)                      # (10000, D)
Xtr_all = encode(train)
Xte_all = encode(test)

y_orig = (orig["Will_Buy_EV"] == "Yes").to_numpy(np.int8)
y_tr = (train["Will_Buy_EV"] == "Yes").to_numpy(np.int8)

def nearest(Xq, chunk=2000):
    """Her satir icin en yakin orijinal satir indeksi + mesafe (L2^2)."""
    n = Xq.shape[0]
    idx = np.empty(n, np.int32)
    dist = np.empty(n, np.float32)
    sq_o = (Xo ** 2).sum(1)
    for i in range(0, n, chunk):
        X = Xq[i:i + chunk]
        d = (X ** 2).sum(1, keepdims=True) + sq_o[None, :] - 2.0 * (X @ Xo.T)
        j = d.argmin(1)
        idx[i:i + chunk] = j
        dist[i:i + chunk] = np.sqrt(np.maximum(d[np.arange(len(X)), j], 0))
    return idx, dist

N_SAMP = 40000
samp_tr = rng.choice(len(train), N_SAMP, replace=False)
samp_te = rng.choice(len(test), N_SAMP, replace=False)

for ad, Xall, samp, y in [("train", Xtr_all, samp_tr, y_tr), ("test", Xte_all, samp_te, None)]:
    idx, dist = nearest(Xall[samp])
    print(f"\n=== {ad} (n={N_SAMP}) ===")
    print("mesafe persentiller:", np.percentile(dist, [1, 5, 25, 50, 75, 95, 99]).round(3))
    # farkli mesafe esiginde: kac satir esigin altinda + (train icin) hedef uyumu
    for t in [0.5, 1.0, 2.0, 4.0]:
        m = dist < t
        if y is not None:
            agree = (y[samp][m] == y_orig[idx][m]).mean() if m.sum() else np.nan
            base = y[samp][m].mean() if m.sum() else np.nan
            print(f"esik {t}: {m.sum():6d} satir ({m.mean():.1%}) | hedef uyumu {agree:.2%} | orijinal-Yes orani {base:.1%}")
        else:
            print(f"esik {t}: {m.sum():6d} satir ({m.mean():.1%}) | orijinal-Yes orani {y_orig[idx][m].mean():.1%}")

np.save(BASE + r"\nn_probe.npy", np.array([1]))
