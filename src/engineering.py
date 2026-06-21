import re
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold

GROUP_KEYS = [
    ['ID корпуса'],
    ['ID корпуса', 'rooms_clean'],
    ['ID корпуса', 'rooms_clean', 'Дата договора_year'],
    ['ID корпуса', 'rooms_clean', 'floor_num'],
    ['Проект', 'rooms_clean', 'Дата договора_year'],
    ['Девелопер', 'rooms_clean', 'Дата договора_year'],
]

def as_percent_number(series):
    return pd.to_numeric(
        series.astype(str)
        .str.replace('%', '', regex=False)
        .str.replace(',', '.', regex=False)
        .str.replace('н/д', '', regex=False)
        .str.strip(),
        errors='coerce',
    )

def as_number(series):
    return pd.to_numeric(
        series.astype(str).replace({'н/д': np.nan, 'nan': np.nan}),
        errors='coerce',
    )

def add_derived(raw):
    out = raw.copy()
    out['area'] = pd.to_numeric(out['Площадь согласно ПД'], errors='coerce')
    out['rooms_clean'] = (
        out['Количество комнат в прайс-листе, типология bnmap.pro']
        .astype(str)
        .str.replace('ст', '0', regex=False)
        .replace({'nan': np.nan, '-': np.nan})
    )
    out['rooms_num'] = pd.to_numeric(out['rooms_clean'], errors='coerce')
    out['floor_num'] = pd.to_numeric(out['Этаж'], errors='coerce')
    out['section_num'] = pd.to_numeric(out['Секция'], errors='coerce')
    
    out['area_x_floor'] = out['area'] * out['floor_num']
    out['area_per_room_plus1'] = out['area'] / (out['rooms_num'] + 1).replace(0, np.nan)
    out['area_round1'] = out['area'].round(1).astype(str)
    out['room_size_index'] = out['area'] / out['rooms_num'].replace(0, 0.5)
    out['living_ratio_sqm'] = out['area'] * (out['floor_num'] + 2)

    for col in ['Рост цены за 1 кв.м за период экспонирования', 'Рост бюджета покупки за период экспонирования']:
        out[f'{col}_num'] = as_percent_number(out[col])

    exposure = 'Срок в экспозиции до момента сделки, дней'
    out[f'{exposure}_num'] = as_number(out[exposure])

    date_cols = ['Заявленный срок ввода в эксплуатацию', 'Старт продаж', 'Дата договора', 'Дата регистрации']
    for col in date_cols:
        if col in out.columns:
            dt = pd.to_datetime(out[col], errors='coerce')
            out[f'{col}_ord'] = (dt - pd.Timestamp('2017-01-01')).dt.days
            out[f'{col}_month'] = dt.dt.month
            out[f'{col}_quarter'] = dt.dt.quarter
            out[f'{col}_year'] = dt.dt.year
            out[f'{col}_ym'] = dt.dt.to_period('M').astype(str)

    for left, right in [('Дата договора', 'Старт продаж'), ('Заявленный срок ввода в эксплуатацию', 'Дата договора'), ('Дата регистрации', 'Дата договора')]:
        if left in out.columns and right in out.columns:
            left_dt = pd.to_datetime(out[left], errors='coerce')
            right_dt = pd.to_datetime(out[right], errors='coerce')
            out[f'days_{left}_minus_{right}'] = (left_dt - right_dt).dt.days

    return out

def key_index(df, keys):
    key_df = df[keys].astype(str).fillna('__NA__')
    if len(keys) == 1: 
        return pd.Index(key_df[keys[0]])
    return pd.MultiIndex.from_frame(key_df)

def group_stats(df, keys, value_col):
    key_df = df[keys].astype(str).fillna('__NA__').copy()
    key_df[value_col] = df[value_col].to_numpy()
    return key_df.groupby(keys, dropna=False)[value_col].agg(['mean', 'median', 'std'])

def add_group_stats(train, test, y, random_state=42):
    train = train.copy()
    test = test.copy()
    ppm = y / train['area'].replace(0, np.nan)
    train['_target_tmp'] = y.to_numpy()
    train['_ppm_tmp'] = ppm.to_numpy()

    kf = KFold(n_splits=5, shuffle=True, random_state=random_state)
    global_target, global_ppm = float(y.mean()), float(ppm.mean())
    new_train_cols, new_test_cols = {}, {}

    for keys in GROUP_KEYS:
        name = '__'.join(keys).replace(' ', '_')
        for stat_target, default in [('_target_tmp', global_target), ('_ppm_tmp', global_ppm)]:
            for agg in ['mean', 'median', 'std']:
                new_train_cols[f'grp_{name}_{stat_target}_{agg}'] = np.full(len(train), np.nan)
                new_test_cols[f'grp_{name}_{stat_target}_{agg}'] = np.full(len(test), np.nan)

            test_idx = key_index(test, keys)
            full_stats = group_stats(train, keys, stat_target)
            for agg in ['mean', 'median', 'std']:
                new_test_cols[f'grp_{name}_{stat_target}_{agg}'] = full_stats[agg].reindex(test_idx).to_numpy()

            for fit_idx, val_idx in kf.split(train):
                fit, val = train.iloc[fit_idx], train.iloc[val_idx]
                stats = group_stats(fit, keys, stat_target)
                val_key = key_index(val, keys)
                for agg in ['mean', 'median', 'std']:
                    new_train_cols[f'grp_{name}_{stat_target}_{agg}'][val_idx] = stats[agg].reindex(val_key).to_numpy()

            for agg in ['mean', 'median']:
                col = f'grp_{name}_{stat_target}_{agg}'
                new_train_cols[col] = pd.Series(new_train_cols[col]).fillna(default).to_numpy()
                new_test_cols[col] = pd.Series(new_test_cols[col]).fillna(default).to_numpy()
            for agg in ['std']:
                col = f'grp_{name}_{stat_target}_{agg}'
                new_train_cols[col] = pd.Series(new_train_cols[col]).fillna(0).to_numpy()
                new_test_cols[col] = pd.Series(new_test_cols[col]).fillna(0).to_numpy()

    train = pd.concat([train, pd.DataFrame(new_train_cols, index=train.index)], axis=1)
    test = pd.concat([test, pd.DataFrame(new_test_cols, index=test.index)], axis=1)
    return train.drop(columns=['_target_tmp', '_ppm_tmp']), test

def clean_column_names(df):
    clean_cols = [re.sub(r'[^a-zA-Z0-9_]', '', c) for c in df.columns]
    df.columns = [f"num_feat_{i}_{name}" if name else f"num_feat_{i}" for i, name in enumerate(clean_cols)]
    return df
