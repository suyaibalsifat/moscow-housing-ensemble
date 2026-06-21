import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import RidgeCV
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb

def run_multi_objective_stacking(X, X_num, X_test, X_test_num, y_raw, y_ppm, cat_cols, train_df, test_df, random_state=42, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    oof_preds = {k: np.zeros(len(X)) for k in ['cat_ppm', 'cat_raw', 'lgb_ppm', 'lgb_raw', 'xgb_ppm', 'xgb_raw']}
    test_preds = {k: np.zeros(len(X_test)) for k in ['cat_ppm', 'cat_raw', 'lgb_ppm', 'lgb_raw', 'xgb_ppm', 'xgb_raw']}

    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y_ppm), 1):
        X_tr_cat, X_val_cat = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr_ppm, y_val_ppm = y_ppm.iloc[tr_idx], y_ppm.iloc[val_idx]
        y_tr_raw, y_val_raw = y_raw.iloc[tr_idx], y_raw.iloc[val_idx]
        
        # --- CatBoost Engine ---
        cat_ppm = CatBoostRegressor(iterations=3500, learning_rate=0.025, depth=8, l2_leaf_reg=5, loss_function='RMSE', random_seed=random_state, task_type='GPU', verbose=0)
        cat_ppm.fit(X_tr_cat, y_tr_ppm, cat_features=cat_cols, eval_set=(X_val_cat, y_val_ppm), early_stopping_rounds=150)
        oof_preds['cat_ppm'][val_idx] = cat_ppm.predict(X_val_cat)
        test_preds['cat_ppm'] += cat_ppm.predict(X_test) / n_splits
        
        cat_raw = CatBoostRegressor(iterations=3500, learning_rate=0.025, depth=8, l2_leaf_reg=5, loss_function='RMSE', random_seed=random_state, task_type='GPU', verbose=0)
        cat_raw.fit(X_tr_cat, y_tr_raw, cat_features=cat_cols, eval_set=(X_val_cat, y_val_raw), early_stopping_rounds=150)
        oof_preds['cat_raw'][val_idx] = cat_raw.predict(X_val_cat)
        test_preds['cat_raw'] += cat_raw.predict(X_test) / n_splits

        # --- LightGBM Engine (Robust Huber) ---
        lgb_tr_ppm = lgb.Dataset(X_num.iloc[tr_idx], y_tr_ppm)
        lgb_va_ppm = lgb.Dataset(X_num.iloc[val_idx], y_val_ppm, reference=lgb_tr_ppm)
        lgb_ppm_mdl = lgb.train({"objective": "huber", "metric": "rmse", "learning_rate": 0.025, "num_leaves": 80, "device": "gpu", "seed": random_state, "verbosity": -1},
                                lgb_tr_ppm, num_boost_round=3500, valid_sets=[lgb_va_ppm], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof_preds['lgb_ppm'][val_idx] = lgb_ppm_mdl.predict(X_num.iloc[val_idx])
        test_preds['lgb_ppm'] += lgb_ppm_mdl.predict(X_test_num) / n_splits
        
        lgb_tr_raw = lgb.Dataset(X_num.iloc[tr_idx], y_tr_raw)
        lgb_va_raw = lgb.Dataset(X_num.iloc[val_idx], y_val_raw, reference=lgb_tr_raw)
        lgb_raw_mdl = lgb.train({"objective": "huber", "metric": "rmse", "learning_rate": 0.025, "num_leaves": 80, "device": "gpu", "seed": random_state, "verbosity": -1},
                                lgb_tr_raw, num_boost_round=3500, valid_sets=[lgb_va_raw], callbacks=[lgb.early_stopping(150, verbose=False)])
        oof_preds['lgb_raw'][val_idx] = lgb_raw_mdl.predict(X_num.iloc[val_idx])
        test_preds['lgb_raw'] += lgb_raw_mdl.predict(X_test_num) / n_splits

        # --- XGBoost Engine ---
        xgb_ppm_mdl = xgb.XGBRegressor(n_estimators=3500, learning_rate=0.025, max_depth=7, subsample=0.8, colsample_bytree=0.8, tree_method="hist", device="cuda", random_state=random_state)
        xgb_ppm_mdl.fit(X_num.iloc[tr_idx], y_tr_ppm, eval_set=[(X_num.iloc[val_idx], y_val_ppm)], verbose=False)
        oof_preds['xgb_ppm'][val_idx] = xgb_ppm_mdl.predict(X_num.iloc[val_idx])
        test_preds['xgb_ppm'] += xgb_ppm_mdl.predict(X_test_num) / n_splits
        
        xgb_raw_mdl = xgb.XGBRegressor(n_estimators=3500, learning_rate=0.025, max_depth=7, subsample=0.8, colsample_bytree=0.8, tree_method="hist", device="cuda", random_state=random_state)
        xgb_raw_mdl.fit(X_num.iloc[tr_idx], y_tr_raw, eval_set=[(X_num.iloc[val_idx], y_val_raw)], verbose=False)
        oof_preds['xgb_raw'][val_idx] = xgb_raw_mdl.predict(X_num.iloc[val_idx])
        test_preds['xgb_raw'] += xgb_raw_mdl.predict(X_test_num) / n_splits

    area_train = train_df['area'].to_numpy()
    area_test = test_df['area'].to_numpy()

    meta_oof = np.vstack([oof_preds['cat_ppm'] * area_train, oof_preds['cat_raw'], oof_preds['lgb_ppm'] * area_train, oof_preds['lgb_raw'], oof_preds['xgb_ppm'] * area_train, oof_preds['xgb_raw']]).T
    meta_test = np.vstack([test_preds['cat_ppm'] * area_test, test_preds['cat_raw'], test_preds['lgb_ppm'] * area_test, test_preds['lgb_raw'], test_preds['xgb_ppm'] * area_test, test_preds['xgb_raw']]).T

    scaler = StandardScaler()
    meta_oof_scaled = scaler.fit_transform(meta_oof)
    meta_test_scaled = scaler.transform(meta_test)

    meta_learner = RidgeCV(alphas=np.logspace(-3, 4, 100), cv=5)
    meta_learner.fit(meta_oof_scaled, y_raw.values)
    
    return np.clip(meta_learner.predict(meta_test_scaled), 0, None)
