# -*- coding: utf-8 -*-
"""v9 - MLP + entity embeddings (PyTorch CPU).

Sayisal kolonlarin basamaklari (49 pozisyon, her biri 0-9) + 6 kategorik
kolon icin embedding + 7 ham sayisal -> kucuk MLP. GBM'lerden farkli hata
profili hedefleniyor; amaç blend cesitliligi.

State: .v9_state/nn_fold{f}.npz  (diger modellerle ayni format)

Kullanim:
  python model_nn.py --seed 42
  python seed_blend9.py   # NN'i blende kat (seed_blend8'i genislet)
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

TARGET = "Will_Buy_EV"
N_SPLITS = 5
STATE_DIR = ".v9_state"
os.makedirs(STATE_DIR, exist_ok=True)

DECIMAL_COLS = {"Daily_Commute_km", "Environmental_Concern_Level"}
DIGIT_COLS = ["Age", "Annual_Income_USD", "Daily_Commute_km",
              "Number_of_Cars_Owned", "Charging_Stations_Near_Home",
              "Charging_Stations_Near_Work", "Environmental_Concern_Level"]
CAT_COLS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
            "Subsidy_Available", "Range_Anxiety_Level"]
EMB_DIM_DIGIT = 6
EMB_DIM_CAT = 4
HID = 64
BATCH = 32768
EPOCHS = 15
PATIENCE = 3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)


def build(df):
    out = df.copy()
    digits = np.zeros((len(out), len(DIGIT_COLS) * 7), np.int64)
    for i, c in enumerate(DIGIT_COLS):
        scale = 10 if c in DECIMAL_COLS else 1
        v = np.abs((out[c].fillna(0).astype(np.float64) * scale).round().astype(np.int64))
        for p in range(7):
            digits[:, i * 7 + p] = (v // (10 ** p)) % 10
    nums = out[DIGIT_COLS].fillna(0).astype(np.float32).to_numpy()
    mu, sd = nums.mean(0), nums.std(0) + 1e-6
    return digits, ((nums - mu) / sd).astype(np.float32), out[CAT_COLS]


class Net(nn.Module):
    def __init__(self, n_digit=49, n_cat=6, n_num=7, cat_cards=None):
        super().__init__()
        # tum basamak pozisyonlari TEK embedding tablosu paylasir (hiz)
        self.emb_d = nn.Embedding(10, EMB_DIM_DIGIT)
        self.emb_c = nn.ModuleList([nn.Embedding(card, EMB_DIM_CAT) for card in cat_cards])
        in_dim = n_digit * EMB_DIM_DIGIT + n_cat * EMB_DIM_CAT + n_num
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, HID), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(HID, HID), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(HID, 1))

    def forward(self, xd, xc, xn):
        ed = self.emb_d(xd).reshape(len(xd), -1)
        ec = torch.cat([e(xc[:, i]) for i, e in enumerate(self.emb_c)], dim=1)
        return self.mlp(torch.cat([ed, ec, xn], dim=1)).squeeze(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    train = pd.read_csv("train.csv")
    test = pd.read_csv("test.csv")
    y = (train[TARGET] == "Yes").astype(np.int32).to_numpy()

    dtr, ntr, ctr = build(train)
    dte, nte, cte = build(test)
    # kategorik kodlama (train kategori sirasi)
    cat_codes = {}
    for c in CAT_COLS:
        cats = sorted(train[c].dropna().unique().tolist())
        cat_codes[c] = {v: i for i, v in enumerate(cats)}
    def enc_cat(df):
        return np.stack([df[c].map(cat_codes[c]).fillna(0).astype(np.int64).to_numpy()
                         for c in CAT_COLS], axis=1)
    xctr = enc_cat(train)
    xcte = enc_cat(test)
    cat_cards = [max(2, len(cat_codes[c])) for c in CAT_COLS]

    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=args.seed)
    oof = np.zeros(len(train))
    preds_te = np.zeros(len(test))
    for fold, (tr, va) in enumerate(skf.split(train, y)):
        fp = f"{STATE_DIR}/nn_fold{fold}.npz"
        if os.path.exists(fp):
            print(f"  NN fold {fold+1}/{N_SPLITS} -> ATLANDI (var)", flush=True)
            d = np.load(fp)
            oof[va] = d["oof"]; preds_te += d["test"]
            continue
        net = Net(cat_cards=cat_cards)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
        lossf = nn.BCEWithLogitsLoss()
        xd_tr = torch.from_numpy(dtr[tr]); xd_va = torch.from_numpy(dtr[va])
        xc_tr = torch.from_numpy(xctr[tr]); xc_va = torch.from_numpy(xctr[va])
        xn_tr = torch.from_numpy(ntr[tr]); xn_va = torch.from_numpy(ntr[va])
        yb_tr = torch.from_numpy(y[tr].astype(np.float32))
        yb_va = torch.from_numpy(y[va].astype(np.float32))

        best_auc, best_state, wait = 0, None, 0
        for ep in range(EPOCHS):
            net.train()
            perm = torch.randperm(len(tr))
            tot = 0
            for i in range(0, len(perm), BATCH):
                idx = perm[i:i + BATCH]
                opt.zero_grad()
                out = net(xd_tr[idx], xc_tr[idx], xn_tr[idx])
                loss = lossf(out, yb_tr[idx])
                loss.backward(); opt.step()
                tot += loss.item() * len(idx)
            net.eval()
            with torch.no_grad():
                pv = []
                for i in range(0, len(va), BATCH * 4):
                    pv.append(torch.sigmoid(net(xd_va[i:i + BATCH * 4], xc_va[i:i + BATCH * 4],
                                                xn_va[i:i + BATCH * 4])).numpy())
                pv = np.concatenate(pv)
            auc = roc_auc_score(y[va], pv)
            if auc > best_auc:
                best_auc, wait = auc, 0
                best_state = {k: v.clone() for k, v in net.state_dict().items()}
            else:
                wait += 1
                if wait >= PATIENCE:
                    break
        net.load_state_dict(best_state)
        net.eval()
        with torch.no_grad():
            pv = []
            for i in range(0, len(va), BATCH * 4):
                pv.append(torch.sigmoid(net(xd_va[i:i + BATCH * 4], xc_va[i:i + BATCH * 4],
                                            xn_va[i:i + BATCH * 4])).numpy())
            pt = []
            for i in range(0, len(test), BATCH * 4):
                pt.append(torch.sigmoid(net(torch.from_numpy(dte[i:i + BATCH * 4]),
                                            torch.from_numpy(xcte[i:i + BATCH * 4]),
                                            torch.from_numpy(nte[i:i + BATCH * 4]))).numpy())
        oof_va = np.concatenate(pv)
        te = np.concatenate(pt)
        oof[va] = oof_va
        preds_te += te / N_SPLITS
        np.savez(fp, oof=oof_va, test=te)
        print(f"  NN fold {fold+1}/{N_SPLITS} | AUC: {roc_auc_score(y[va], oof_va):.5f} | "
              f"ep {ep+1} best {best_auc:.5f}", flush=True)
    print(f"  NN OOF AUC: {roc_auc_score(y, oof):.5f}")


if __name__ == "__main__":
    main()
