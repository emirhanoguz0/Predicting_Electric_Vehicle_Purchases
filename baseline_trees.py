import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

def main():
    print("Veri yükleniyor...")
    try:
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
    except FileNotFoundError:
        print("HATA: 'train.csv' veya 'test.csv' bulunamadı.")
        print("Lütfen veri setini Kaggle'dan indirip (train.csv ve test.csv) bu klasöre yerleştir.")
        return

    target_col = 'Will_Buy_EV'
    if target_col not in train.columns:
        print(f"HATA: Hedef sütun '{target_col}' train.csv içinde bulunamadı.")
        return

    # 'id' kolonu submission için gerekli genelde
    has_id = 'id' in train.columns
    drop_cols = ['id'] if has_id else []
    
    features = [c for c in train.columns if c not in drop_cols + [target_col]]

    X = train[features]
    # XGBoost hedef değişkenin [0, 1] olmasını bekler, bu yüzden çeviriyoruz
    y = train[target_col].map({'No': 0, 'Yes': 1}).fillna(train[target_col])
    if y.dtype == 'object' or y.dtype.name == 'category':
        y = (train[target_col] == 'Yes').astype(int)
    
    X_test = test[features].copy()

    # Kategorik kolonları tespit et ve tipini 'category' yap (Ağaç modelleri için avantajlı)
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        X[col] = X[col].astype('category')
        X_test[col] = X_test[col].astype('category')

    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_lgb = np.zeros(len(train))
    oof_xgb = np.zeros(len(train))
    preds_lgb = np.zeros(len(test))
    preds_xgb = np.zeros(len(test))

    print(f"Model eğitimi başlıyor ({n_splits} Fold)...")
    print(f"Öznitelik Sayısı: {len(features)} | Kategorik Öznitelik Sayısı: {len(cat_cols)}\n")
    
    for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[valid_idx], y.iloc[valid_idx]
        
        # 1. LightGBM Modeli
        model_lgb = LGBMClassifier(n_estimators=1000, learning_rate=0.05, random_state=42, verbose=-1)
        model_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)])
        
        valid_preds_lgb = model_lgb.predict_proba(X_va)[:, 1]
        oof_lgb[valid_idx] = valid_preds_lgb
        preds_lgb += model_lgb.predict_proba(X_test)[:, 1] / n_splits
        
        # 2. XGBoost Modeli (enable_categorical=True özelliğiyle kategorikleri yerel destekler)
        model_xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05, random_state=42, 
                                  enable_categorical=True, tree_method='hist')
        model_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        
        valid_preds_xgb = model_xgb.predict_proba(X_va)[:, 1]
        oof_xgb[valid_idx] = valid_preds_xgb
        preds_xgb += model_xgb.predict_proba(X_test)[:, 1] / n_splits
        
        fold_auc_lgb = roc_auc_score(y_va, valid_preds_lgb)
        fold_auc_xgb = roc_auc_score(y_va, valid_preds_xgb)
        print(f"Fold {fold+1} | LGBM AUC: {fold_auc_lgb:.5f} | XGB AUC: {fold_auc_xgb:.5f}")

    print("\n--- Genel OOF (Out-Of-Fold) Skorları ---")
    auc_lgb = roc_auc_score(y, oof_lgb)
    auc_xgb = roc_auc_score(y, oof_xgb)
    print(f"LGBM Toplam OOF AUC: {auc_lgb:.5f}")
    print(f"XGB Toplam OOF AUC: {auc_xgb:.5f}")
    
    # Basit Blending (%50 LGBM + %50 XGBoost)
    oof_blend = (oof_lgb + oof_xgb) / 2
    preds_blend = (preds_lgb + preds_xgb) / 2
    auc_blend = roc_auc_score(y, oof_blend)
    print(f"Blended (Ensemble) OOF AUC: {auc_blend:.5f}")

    # Submission oluşturma
    if has_id:
        sub = pd.DataFrame({'id': test['id'], target_col: preds_blend})
        sub.to_csv('submission.csv', index=False)
        print("\n'submission.csv' başarıyla oluşturuldu!")
    else:
        print("\nUyarı: 'test.csv' içinde 'id' bulunamadı, varsayılan indekslerle kaydediliyor.")
        sub = pd.DataFrame({target_col: preds_blend})
        sub.to_csv('submission.csv', index=False)

if __name__ == "__main__":
    main()
