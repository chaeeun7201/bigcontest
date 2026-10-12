"""18번: 동일한 04번 Y1/Week+Episode 분할에서 회귀 알고리즘 비교.

사용: python modeling/18_compare_regression_algorithms.py
같은 폴더에 04, 11, 16번 스크립트와 modeling_dataset_v2_filtered.csv 필요.
Validation MAE로만 선택. Test는 선택 후 기존 GB와 선택 모델만 평가.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
OUT = HERE / 'weather_algorithm_comparison_18'
SEED = 20261003
WINNERS_16 = {'heatwave': 'all_added__original_04', 'cold_wave': 'all_added__subsample'}


def import_file(filename, alias):
    path = HERE / filename
    if not path.is_file():
        raise FileNotFoundError(f'필수 파일이 없습니다: {path}')
    spec = importlib.util.spec_from_file_location(alias, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_preprocessor(numeric, categories):
    return ColumnTransformer([
        ('num', Pipeline([
            ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
            ('scale', StandardScaler()),
        ]), numeric),
        ('cat', Pipeline([
            ('impute', SimpleImputer(strategy='most_frequent')),
            ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
        ]), categories),
    ], sparse_threshold=0)


def make_model(name, numeric, categories, gb_params):
    if name == 'GradientBoosting_16':
        reg = GradientBoostingRegressor(**gb_params, random_state=SEED)
    elif name == 'HistGradientBoosting':
        reg = HistGradientBoostingRegressor(
            max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
            min_samples_leaf=15, l2_regularization=1.0,
            early_stopping=False, random_state=SEED)
    elif name == 'ExtraTrees':
        reg = ExtraTreesRegressor(
            n_estimators=300, max_features=0.8, min_samples_leaf=3,
            max_depth=12, n_jobs=-1, random_state=SEED)
    else:
        raise ValueError(name)
    return Pipeline([('prep', make_preprocessor(numeric, categories)), ('model', reg)])


def evaluate(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        'MAE_pp': float(mean_absolute_error(y, pred)),
        'RMSE_pp': float(np.sqrt(mean_squared_error(y, pred))),
        'R2': float(r2_score(y, pred)),
        'direction_accuracy': float(np.mean((y < 0) == (pred < 0))),
    }


def main():
    print('18번: 동일 분할·동일 피처에서 GB / HistGB / ExtraTrees 비교')
    m11 = import_file('11_improve_original_gb_features.py', 'exp11')
    m16 = import_file('16_tune_gradient_boosting.py', 'exp16')
    original = m11.load_original()
    df, dates, episodes = m11.prepare(original)
    OUT.mkdir(parents=True, exist_ok=True)
    dates.to_csv(OUT / 'split_by_date.csv', index=False, encoding='utf-8-sig')
    episodes.to_csv(OUT / 'episodes.csv', index=False, encoding='utf-8-sig')

    numeric, categories = m16.ALL_ADDED, m16.KEY
    feature_cols = numeric + categories
    algorithms = ('GradientBoosting_16', 'HistGradientBoosting', 'ExtraTrees')
    records, test_predictions, selected_rows = [], [], []

    for hazard, winner16 in WINNERS_16.items():
        target = f'Y1_{hazard}'
        work = df.loc[df[hazard].eq(1) & df[target].notna()].copy()
        parts = {s: work.loc[work['split'].eq(s)].copy()
                 for s in ('train', 'validation', 'test')}
        if any(part.empty for part in parts.values()):
            raise ValueError(f'{hazard}: train/validation/test 중 빈 분할이 있습니다.')
        gb_params = m16.CONFIGS[winner16.split('__', 1)[1]]
        print(f'\n[{hazard}] ' + ', '.join(f'{k}={len(v):,}' for k, v in parts.items()))
        print(f'기준 GB: {winner16} / Validation MAE 기준 비교')
        fitted = {}
        for name in algorithms:
            model = make_model(name, numeric, categories, gb_params)
            model.fit(parts['train'][feature_cols], parts['train'][target])
            fitted[name] = model
            pred = model.predict(parts['validation'][feature_cols])
            metric = evaluate(parts['validation'][target], pred)
            records.append({'hazard': hazard, 'algorithm': name, 'split': 'validation',
                            'n': len(pred), **metric})
            print(f'  {name:24s} Val MAE={metric["MAE_pp"]:.3f}pp '
                  f'R²={metric["R2"]:.3f} 방향={metric["direction_accuracy"]:.3f}')

        validation = [r for r in records if r['hazard'] == hazard and r['split'] == 'validation']
        selected = min(validation, key=lambda r: r['MAE_pp'])['algorithm']
        selected_rows.append({'hazard': hazard, 'selected_algorithm': selected,
                              'selection_rule': 'validation_MAE_pp'})
        print(f'Validation 선택: {selected}')
        for name in dict.fromkeys(('GradientBoosting_16', selected)):
            pred = fitted[name].predict(parts['test'][feature_cols])
            metric = evaluate(parts['test'][target], pred)
            records.append({'hazard': hazard, 'algorithm': name, 'split': 'test',
                            'n': len(pred), **metric})
            print(f'  TEST {name:19s} MAE={metric["MAE_pp"]:.3f}pp '
                  f'R²={metric["R2"]:.3f} 방향={metric["direction_accuracy"]:.3f}')
            frame = parts['test'][['date', *categories, target]].copy()
            frame['hazard'] = hazard
            frame['algorithm'] = name
            frame['predicted_Y1'] = pred
            test_predictions.append(frame)

    pd.DataFrame(records).to_csv(OUT / 'model_comparison.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(selected_rows).to_csv(OUT / 'selected_models.csv', index=False, encoding='utf-8-sig')
    pd.concat(test_predictions, ignore_index=True).to_csv(
        OUT / 'test_predictions.csv', index=False, encoding='utf-8-sig')
    (OUT / 'config.json').write_text(json.dumps({
        'seed': SEED, 'target': '04번 Train-only Y1',
        'split': '04번 Week+Episode split (11번 prepare 재사용)',
        'features': {'numeric': numeric, 'categorical': categories},
        'GB_winners_16': WINNERS_16,
        'algorithms': list(algorithms),
        'selection': 'Validation MAE only; Test used only after selection',
        'limitations': '독립 Test 지역-일이 재난별 2개뿐이며, 인과 피해액/사전 예측이 아님',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n완료: {OUT}')
    print('주의: 재난별 Test 독립 지역·날짜가 2개뿐이므로 점수 차이를 일반화하지 마세요.')
    print('주의: 동일한 Test를 반복 확인하는 실험은 최종 성능의 낙관적 해석을 유발할 수 있습니다.')


if __name__ == '__main__':
    main()
