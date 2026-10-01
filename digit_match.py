# -*- coding: utf-8 -*-
"""Basamak uzayinda orijinal <-> sentetik eslesme avi.

Ham degerlerde tam eslesme yoktu. Ama SDGM basamak kaliplarini koruyorsa,
sayisal kolonlarin BASAMAK IMZASI uzerinden orijinal satirlari bulabiliriz.
Eslesme orani yuksek ve etiket uyumu taban oraninin ustunde cikarsa,
test satirlari icin orijinal etiket sinyali uretilebilir.
"""
import numpy as np
import pandas as pd

BASE = r"D:\Second_Brain\🏰 300-Projects\Makine-Ogrenmesi\Predicting_Electric_Vehicle_Purchases"
train = pd.read_csv(BASE + r"\train.csv")
test = pd.read_csv(BASE + r"\test.csv")
orig = pd.read_csv(BASE + r"\source_data\EV_Adoption_and_Range_Anxiety_Dataset.csv")

DECIMAL_COLS = {"Daily_Commute_km", "Environmental_Concern_Level"}
NUM = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
       "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
       "Environmental_Concern_Level"]
N_DIGITS = 7

def digit_sig(df):
    """(n, 7*len(NUM)) basamak matrisi + eksik maskesi."""
    mats = []
    for c in NUM:
        scale = 10 if c in DECIMAL_COLS else 1
        v = (df[c].fillna(0).astype(np.float64) * scale).round().astype(np.int64)
        v = np.abs(v)
        d = np.stack([((v // (10 ** p)) % 10) for p in range(N_DIGITS)], axis=1)
        mats.append(d.astype(np.int8))
    return np.hstack(mats)

So = digit_sig(orig)      # (10000, 49)
Str_ = digit_sig(train)
Ste = digit_sig(test)
print("imza boyutlari:", So.shape, Str_.shape, Ste.shape)

def match_count(Sq, label):
    # her sentetik satir icin orijinalde ayni basamak imzasini tasiyan var mi
    set_o = set(map(tuple, So.tolist()))
    sigs = map(tuple, Sq.tolist())
    hit = np.array([s in set_o for s in sigs])
    print(f"{label}: tam basamak eslesmesi {hit.sum()}/{len(Sq)} ({hit.mean():.1%})")
    return hit

ho = match_count(Str_, "train")
he = match_count(Ste, "test")

# eslesen train satirlarinda etiket uyumu
if ho.sum() > 0:
    # orijinal imza -> etiket eslemesi (tekillik kontrolu)
    from collections import defaultdict
    sig2lab = defaultdict(list)
    for s, l in zip(map(tuple, So.tolist()), (orig["Will_Buy_EV"] == "Yes").astype(int)):
        sig2lab[s].append(l)
    multi = {k: v for k, v in sig2lab.items() if len(v) > 1}
    print(f"orijinalde ayni imzayi tasiyan satir cifti sayisi: {len(multi)}")
    tr_idx = np.where(ho)[0]
    tr_sig = list(map(tuple, Str_[tr_idx].tolist()))
    lab_orig = np.array([np.mean(sig2lab[s]) for s in tr_sig])
    lab_syn = (train["Will_Buy_EV"].iloc[tr_idx] == "Yes").astype(int).to_numpy()
    print(f"eslesenlerde orijinal Yes orani ort: {lab_orig.mean():.3f}, sentetik Yes orani: {lab_syn.mean():.3f}")
    agree = ((lab_orig > 0.5).astype(int) == lab_syn).mean()
    print(f"eslesenlerde etiket uyumu (cogunluk): {agree:.2%}")

# kismi eslesme: ilk k basamak pozisyonu (dusuk basamaklar gurultuye duyarli)
for k in [2, 3]:
    set_o_k = set(map(tuple, So[:, :k * len(NUM) // 1][:, :k].tolist())) if False else set(map(tuple, So[:, :k].tolist()))
    # Not: ilk k kolon = ilk kolonun k basamagi; tum kolonlarin ilk k basamagi icin ayir
print("\n--- tum kolonlarin ilk K basamagi ile kismi eslesme ---")
for k in [1, 2, 3, 4]:
    set_o_k = set(map(tuple, So.reshape(len(So), 7, 7)[:, :k, :].reshape(len(So), -1).tolist()))
    tr_hit = np.mean([s in set_o_k for s in map(tuple, Str_.reshape(len(Str_), 7, 7)[:, :k, :].reshape(len(Str_), -1).tolist())])
    te_hit = np.mean([s in set_o_k for s in map(tuple, Ste.reshape(len(Ste), 7, 7)[:, :k, :].reshape(len(Ste), -1).tolist())])
    print(f"ilk {k} basamak | orijinalde varlik | train {tr_hit:.1%} | test {te_hit:.1%}")
