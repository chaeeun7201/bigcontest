"""17번: 16번 선택 GB의 극단 매출변화율 대응 실험.
실행: python modeling/17_robust_gb_training.py
필요: 04_train_weather_impact_models.py, 11_improve_original_gb_features.py,
      16_tune_gradient_boosting.py, modeling_dataset_v2_filtered.csv
16번과 동일한 Y1/Week+Episode 분할. 학습 타깃만 조정하며 평가 타깃은 원본 유지.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

HERE = Path(__file__).resolve().parent
OUT = HERE / 'weather_robust_training_17'


def import_file(name, alias):
    path = HERE / name
    if not path.exists():
        raise FileNotFoundError(f'필요한 파일이 없습니다: {path}')
    spec = importlib.util.spec_from_file_location(alias, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metrics(y, p):
    y, p = np.asarray(y, dtype=float), np.asarray(p, dtype=float)
    return {'MAE_pp': float(mean_absolute_error(y, p)),
            'R2': float(r2_score(y, p)),
            'direction_accuracy': float(np.mean((y < 0) == (p < 0)))}


def main():
    print('17번: 극단 매출변화율 완화 및 가중치 학습 실험')
    m11 = import_file('11_improve_original_gb_features.py', 'gb11')
    m16 = import_file('16_tune_gradient_boosting.py', 'gb16')
    original = m11.load_original()
    df, dates, episodes = m11.prepare(original)
    OUT.mkdir(exist_ok=True, parents=True)
    dates.to_csv(OUT / 'split_by_date.csv', index=False, encoding='utf-8-sig')
    episodes.to_csv(OUT / 'episodes.csv', index=False, encoding='utf-8-sig')

    # 16번 Validation에서 선택한 후보를 고정한다. Test 성능으로 후보를 선택하지 않는다.
    winners = {'heatwave': 'all_added__original_04',
               'cold_wave': 'all_added__subsample'}
    rows, predictions = [], []
    for hazard, winner in winners.items():
        target = f'Y1_{hazard}'
        work = df[df[hazard].eq(1) & df[target].notna()].copy()
        parts = {s: work.loc[work['split'].eq(s)].copy()
                 for s in ('train', 'validation', 'test')}
        if any(p.empty for p in parts.values()):
            raise ValueError(f'{hazard}: 빈 데이터 분할이 있습니다.')
        params = m16.CONFIGS[winner.split('__', 1)[1]]
        numeric = m16.ALL_ADDED
        cols = numeric + m16.KEY
        train_y = parts['train'][target].astype(float)
        lower, upper = np.quantile(train_y, [0.01, 0.99])
        clipped = train_y.clip(lower, upper)
        scale = max(float(np.median(np.abs(train_y - np.median(train_y)))), 1.0)
        # 관측치별 가중치는 train Y1만으로 계산한다. 너무 작은 가중치는 0.2로 제한.
        weights = np.clip(1.0 / (1.0 + (np.abs(train_y) / (3 * scale)) ** 2), 0.2, 1.0)
        variants = {
            'baseline_16': (train_y, None),
            'winsor_1pct': (clipped, None),
            'robust_weight': (train_y, weights),
            'winsor_plus_weight': (clipped, weights),
        }
        print(f'\n[{hazard}] train={len(parts["train"]):,} '
              f'validation={len(parts["validation"]):,} test={len(parts["test"]):,}')
        print(f'16번 고정 설정: {winner} / Train Y1 clip=[{lower:.2f}, {upper:.2f}]pp')
        fitted = {}
        for name, (y_fit, sample_weight) in variants.items():
            model = m16.make_pipeline(numeric, params)
            fit_args = {} if sample_weight is None else {'model__sample_weight': sample_weight}
            model.fit(parts['train'][cols], y_fit, **fit_args)
            fitted[name] = model
            pred = model.predict(parts['validation'][cols])
            result = metrics(parts['validation'][target], pred)
            rows.append({'hazard': hazard, 'variant': name, 'split': 'validation',
                         'n': len(pred), **result})
            print(f'  {name:22s} Val MAE={result["MAE_pp"]:.3f}pp '
                  f'R²={result["R2"]:.3f} 방향={result["direction_accuracy"]:.3f}')
        candidates = [r for r in rows if r['hazard'] == hazard and r['split'] == 'validation']
        selected = min(candidates, key=lambda r: r['MAE_pp'])['variant']
        print(f'Validation 선택: {selected}')
        # Test는 선택 후 기준 모델과 선택 모델만 확인한다.
        for name in dict.fromkeys(['baseline_16', selected]):
            pred = fitted[name].predict(parts['test'][cols])
            result = metrics(parts['test'][target], pred)
            rows.append({'hazard': hazard, 'variant': name, 'split': 'test',
                         'n': len(pred), **result})
            print(f'  TEST {name:18s} MAE={result["MAE_pp"]:.3f}pp '
                  f'R²={result["R2"]:.3f} 방향={result["direction_accuracy"]:.3f}')
            p = parts['test'][['date', *m16.KEY, target]].copy()
            p['hazard'] = hazard
            p['variant'] = name
            p['predicted_Y1'] = pred
            predictions.append(p)
    pd.DataFrame(rows).to_csv(OUT / 'model_comparison.csv', index=False, encoding='utf-8-sig')
    pd.concat(predictions, ignore_index=True).to_csv(OUT / 'test_predictions.csv', index=False, encoding='utf-8-sig')
    (OUT / 'config.json').write_text(json.dumps({
        'base': '16번 Validation 선택 모델',
        'split': '04번 Week+Episode split / 11번 prepare',
        'target': '04번 Train-only Y1, validation/test unchanged',
        'clip_quantiles': [0.01, 0.99],
        'weight_rule': 'max(0.2, 1/(1+(abs(Y1)/(3*train_MAD))**2))',
        'selection': 'validation MAE only',
        'caveat': 'Only 2 independent test region-days per hazard; observational response not causal damage'
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n완료: {OUT}')
    print('주의: Test 사건 수가 매우 적으므로 작은 점수 차이를 일반화하면 안 됩니다.')


if __name__ == '__main__':
    main()
