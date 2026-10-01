# -*- coding: utf-8 -*-
"""NN v2 — MLP + basamak/kategori embedding'leri (GPU varsa CUDA).

Amac: GBM'lerden (LGBM/XGB/Cat) farkli hata profili ureterek blend
cesitliligi kazandirmak. v9'daki eski NN'in sorunlari duzeltildi:
  - GPU kullanimi (3060), buyuk batch
  - fold + epoch seviyesinde resume: kapanma olursa ayni komut tekrar
    calistirilir, en iyi epoch'dan devam eder
  - submission + full.npz ciktisi (diger modellerle ayni format)

Ozellikler: 7 sayisal kolonun basamaklari (49 pozisyon, paylasimli
embedding) + 6 kategori embedding + standardize 7 ham sayisal.

Kullanim: python model_nn_v2.py --seed 42
Sure: GPU ~30-45 dk | CPU ~1,5-2 sa
Cikti: submission_nn_v2-5fold-mlp_s{seed}.csv + .nn_state_s{seed}/full.npz
"""
import argparse
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

TARGET = 'Will_Buy_EV'
N_SPLITS = 5
BATCH = 32768
MAX_EPOCHS = 30
PATIENCE = 5
EMB_DIM_DIGIT = 8
EMB_DIM_CAT = 4

DECIMAL_COLS = {'Daily_Commute_km', 'Environmental_Concern_Level'}
DIGIT_COLS = ['Age', 'Annual_Income_USD', 'Daily_Commute_km',
              'Number_of_Cars_Owned', 'Charging_Stations_Near_Home',
              'Charging_Stations_Near_Work', 'Environmental_Concern_Level']
CAT_COLS = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible',
            'Subsidy_Available', 'Range_Anxiety_Level']


def build(df):
    digits = np.zeros((len(df), len(DIGIT_COLS) * 7), np.int64)
    for i, c in enumerate(DIGIT_COLS):
        scale = 10 if c in DECIMAL_COLS else 1
        v = np.abs((df[c].fillna(0).astype(np.float64) * scale).round().astype(np.int64))
        for p in range(7):
            digits[:, i * 7 + p] = (v // (10 ** p)) % 10
    nums = df[DIGIT_COLS].fillna(0).astype(np.float32).to_numpy()
    return digits, nums, df[CAT_COLS]


class Net(nn.Module):
    def __init__(self, cat_cards, n_digit=49, n_cat=6, n_num=7):
        super().__init__()
        self.emb_d = nn.Embedding(10, EMB_DIM_DIGIT)
        self.emb_c = nn.ModuleList([nn.Embedding(c, EMB_DIM_CAT) for c in cat_cards])
        in_dim = n_digit * EMB_DIM_DIGIT + n_cat * EMB_DIM_CAT + n_num
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.25),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.15),
            nn.Linear(64, 1))

    def forward(self, xd, xc, xn):
        ed = self.emb_d(xd).reshape(len(xd), -1)
        ec = torch.cat([e(xc[:, i]) for i, e in enumerate(self.emb_c)], dim=1)
        return self.mlp(torch.cat([ed, ec, xn], dim=1)).squeeze(1)


@torch.no_grad()
def predict(net, xd, xc, xn, bs=131072):
    net.eval()
    out = []
    for i in range(0, len(xd), bs):
        out.append(torch.sigmoid(net(xd[i:i + bs], xc[i:i + bs], xn[i:i + bs])))
    return torch.cat(out).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'cihaz: {device}' + (f' ({torch.cuda.get_device_name(0)})'
                                if device == 'cuda' else ''), flush=True)

    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    y = (train[TARGET] == 'Yes').astype(np.int32).to_numpy()
    test_id = test['id'].values

    dtr, ntr, ctr = build(train)
    dte, nte, cte = build(test)
    cat_codes = {}
    for c in CAT_COLS:
        cats = sorted(train[c].dropna().unique().tolist())
        cat_codes[c] = {v: i for i, v in enumerate(cats)}

    def enc_cat(df):
        return np.stack([df[c].map(cat_codes[c]).fillna(0).astype(np.int64).to_numpy()
                         for c in CAT_COLS], axis=1)

    xctr, xcte = enc_cat(train), enc_cat(test)
    cat_cards = [max(2, len(cat_codes[c])) for c in CAT_COLS]

    state_dir = f'.nn_state_s{args.seed}'
    os.makedirs(state_dir, exist_ok=True)

    # tum tensorlari cihaza tasi (600k satir rahat sigar)
    dtr_t = torch.from_numpy(dtr).to(device)
    dte_t = torch.from_numpy(dte).to(device)
    xctr_t = torch.from_numpy(xctr).to(device)
    xcte_t = torch.from_numpy(xcte).to(device)
    y_t = torch.from_numpy(y.astype(np.float32)).to(device)

    oof = np.zeros(len(train), dtype=np.float64)
    test_preds = np.zeros(len(test), dtype=np.float64)
    skf = StratifiedKFold(N_SPLITS, shuffle=True, random_state=args.seed)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(train, y)):
        fp = f'{state_dir}/fold{fold}.npz'
        if os.path.exists(fp):
            print(f'fold {fold+1}/{N_SPLITS} -> ATLANDI (var)', flush=True)
            d = np.load(fp)
            oof[va_idx] = d['oof']
            test_preds += d['test'] / N_SPLITS
            continue

        ckpt = f'{state_dir}/fold{fold}_ckpt.pt'
        net = Net(cat_cards).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=1.5e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=MAX_EPOCHS)
        lossf = nn.BCEWithLogitsLoss()

        # sayisallari fold-train istatistigiyle standardize (sizinti yok)
        mu, sd = ntr[tr_idx].mean(0), ntr[tr_idx].std(0) + 1e-6
        ntr_t = torch.from_numpy(((ntr - mu) / sd).astype(np.float32)).to(device)
        nte_t = torch.from_numpy(((nte - mu) / sd).astype(np.float32)).to(device)

        start_ep, best_auc, wait = 0, 0.0, 0
        if os.path.exists(ckpt):
            c = torch.load(ckpt, map_location=device, weights_only=False)
            net.load_state_dict(c['model'])
            opt.load_state_dict(c['opt'])
            sched.load_state_dict(c['sched'])
            start_ep, best_auc, wait = c['epoch'] + 1, c['best_auc'], c['wait']
            print(f'fold {fold+1}/{N_SPLITS}: ckpt var, epoch {start_ep} '
                  f'best {best_auc:.5f} -> devam', flush=True)

        tr_t = torch.from_numpy(tr_idx).to(device)
        va_t = torch.from_numpy(va_idx).to(device)

        for ep in range(start_ep, MAX_EPOCHS):
            net.train()
            perm = torch.randperm(len(tr_t), device=device)
            for i in range(0, len(perm), BATCH):
                idx = tr_t[perm[i:i + BATCH]]
                opt.zero_grad()
                out = net(dtr_t[idx], xctr_t[idx], ntr_t[idx])
                loss = lossf(out, y_t[idx])
                loss.backward()
                opt.step()
            sched.step()

            pv = predict(net, dtr_t[va_t], xctr_t[va_t], ntr_t[va_t])
            auc = roc_auc_score(y[va_idx], pv)
            if auc > best_auc:
                best_auc, wait = auc, 0
                torch.save({'model': net.state_dict(), 'opt': opt.state_dict(),
                            'sched': sched.state_dict(), 'epoch': ep,
                            'best_auc': best_auc, 'wait': wait}, ckpt)
            else:
                wait += 1
                if wait >= PATIENCE:
                    print(f'  fold {fold+1}: epoch {ep+1} erken durma '
                          f'(best {best_auc:.5f})', flush=True)
                    break
            print(f'  fold {fold+1}/{N_SPLITS} | epoch {ep+1:2d} | '
                  f'val AUC {auc:.5f} | best {best_auc:.5f}', flush=True)

        # en iyi agirliklarla fold ciktisi
        c = torch.load(ckpt, map_location=device, weights_only=False)
        net.load_state_dict(c['model'])
        oof_va = predict(net, dtr_t[va_t], xctr_t[va_t], ntr_t[va_t])
        te_p = predict(net, dte_t, xcte_t, nte_t)
        oof[va_idx] = oof_va
        test_preds += te_p / N_SPLITS
        np.savez_compressed(fp, oof=oof_va, test=te_p)
        os.remove(ckpt)
        print(f'fold {fold+1}/{N_SPLITS} | AUC: {roc_auc_score(y[va_idx], oof_va):.5f}',
              flush=True)

    auc = roc_auc_score(y, oof)
    print(f'\nNN v2 OOF AUC: {auc:.5f}', flush=True)
    out = f'submission_nn_v2-5fold-mlp_s{args.seed}.csv'
    pd.DataFrame({'id': test_id, TARGET: test_preds}).to_csv(out, index=False)
    np.savez_compressed(f'{state_dir}/full.npz', oof=oof, test=test_preds)
    print(f'{out} yazildi', flush=True)


if __name__ == '__main__':
    main()
