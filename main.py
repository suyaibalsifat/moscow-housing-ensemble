import warnings
warnings.filterwarnings("ignore")
from pathlib import Path
import pandas as pd
import numpy as np

from src.engineering import add_derived, add_group_stats, clean_column_names
from src.models import run_multi_objective_stacking

DATA_DIR = Path('/kaggle/input/competitions/ne-vsos-ii-new-moscow')
TARGET = 'Бюджет объекта_тыс'

if __name__ == "__main__":
    print("[INIT] Loading production data grids...")
    train_raw = pd.read_csv(DATA_DIR / 'train.csv')
    test_raw = pd.read_csv(DATA_DIR / 'test.csv')
    y_raw = train_raw[TARGET].astype(float)

    print("[PROCESSING] Generating derived context and out-of-fold target metrics...")
    train = add_derived(train_raw)
    test = add_derived(test_raw)
    train, test = add_group_stats(train, test, y_raw)
    
    y_ppm = y_raw / train['area'].replace(0, np.nan)
    y_ppm = y_ppm.replace([np.inf, -np.inf], np.nan).fillna(y_ppm.median())
    
    X = train.drop(columns=[TARGET, 'id', 'ID лота'], errors='ignore')
    X_test = test.drop(columns=[TARGET, 'id', 'ID лота'], errors='ignore').reindex(columns=X.columns)
    
    cat_cols = [c for c in X.columns if X[c].dtype == 'object']
    for col in cat_cols:
        X[col] = X[col].fillna('__NA__').astype(str)
        X_test[col] = X_test[col].fillna('__NA__').astype(str)

    X_num = X.copy()
    X_test_num = X_test.copy()
    for c in cat_cols:
        freq = X[c].value_counts().to_dict()
        X_num[c + '_freq'] = X_num[c].map(freq).fillna(0)
        X_test_num[c + '_freq'] = X_test_num[c].map(freq).fillna(0)

    X_num = X_num.drop(columns=cat_cols).apply(pd.to_numeric, errors='coerce').fillna(0)
    X_test_num = X_test_num.drop(columns=cat_cols).apply(pd.to_numeric, errors='coerce').fillna(0)

    X_num = clean_column_names(X_num)
    X_test_num = clean_column_names(X_test_num)

    print("[TRAINING] Fitting dual-target stacked gradient boosting models...")
    final_preds = run_multi_objective_stacking(X, X_num, X_test, X_test_num, y_raw, y_ppm, cat_cols, train, test)
    
    print("[EXPORT] Writing final robust predictions file...")
    submission = pd.DataFrame({'id': test_raw['id'], TARGET: final_preds})
    submission.to_csv('submission_multitarget_ensemble.csv', index=False)
    print("[SUCCESS] Process absolute. File written.")
