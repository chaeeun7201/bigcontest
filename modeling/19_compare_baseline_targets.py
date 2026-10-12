"""19번: Train-only 기준매출 개선 실험.

실행: python modeling/19_compare_baseline_targets.py
필요: 04_train_weather_impact_models.py, 11_improve_original_gb_features.py,
      16_tune_gradient_boosting.py, modeling_dataset_v2_filtered.csv

중요: Y1 정의가 바뀌면 Y1 MAE는 직접 비교 불가.
동일 Test 행의 원래 매출 TS_AT 복원 MAE로 비교한다.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

HERE = Path(__file__).resolve().parent
OUT = HERE / 'weather_baseline_experiment_19'
KEY = ['MCT_SGG_CD', 'MCT_RY_CD']
HAZARDS = {'heatwave': (7, 8, 9), 'cold_wave': (11, 12)}
SHRINK_K = 5  # 요일별 평시 관측 수가 적으면 기존 전체 평균에 가깝게 수축


def load_py(name, module_name):
    path = HERE / name
    if not path.exists():
        raise FileNotFoundError(f'필요 파일 없음: {path}')
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_weekday_baseline(df, hazard, months):
    """재난 0인 Train 계절 관측만으로 지역×업종×요일 평균을 추정."""
    eligible = (df['split'].eq('train') & df['month'].isin(months)
                & df[hazard].eq(0))
    history = df.loc[eligible, KEY + ['weekday', 'TS_AT']]
    overall = history.groupby(KEY)['TS_AT'].agg(['mean', 'size']).reset_index()
    overall = overall.rename(columns={'mean': 'base_all', 'size': 'n_all'})
    by_day = history.groupby(KEY + ['weekday'])['TS_AT'].agg(['mean', 'size']).reset_index()
    by_day = by_day.rename(columns={'mean': 'base_day', 'size': 'n_day'})
    result = df[KEY + ['weekday']].merge(overall, on=KEY, how='left', validate='many_to_one')
    result = result.merge(by_day, on=KEY + ['weekday'], how='left', validate='many_to_one')
    n = result['n_day'].fillna(0)
    # n=0이면 전체 평균, 관측 수가 늘수록 해당 요일 평균의 비중 증가
    baseline = (n * result['base_day'].fillna(0) + SHRINK_K * result['base_all']) / (n + SHRINK_K)
    return baseline.to_numpy(), n.to_numpy()


def metrics(actual_sales, predicted_sales, actual_y, predicted_y):
    return {
        'sales_MAE': float(mean_absolute_error(actual_sales, predicted_sales)),
        'sales_WAPE_pct': float(100 * np.abs(actual_sales - predicted_sales).sum() / np.abs(actual_sales).sum()) if np.abs(actual_sales).sum() else np.nan,
        'Y1_MAE_pp_within_definition': float(mean_absolute_error(actual_y, predicted_y)),
        'Y1_R2_within_definition': float(r2_score(actual_y, predicted_y)) if len(actual_y) > 1 else np.nan,
        'sales_direction_accuracy': float(np.mean((actual_y < 0) == (predicted_y < 0))),
    }


def main():
    print('19번: 기준매출 변경에 따른 Gradient Boosting 비교')
    print('주의: Y1 MAE는 정의별로 비교하지 않고, 동일 관측의 실제 매출 MAE로 비교')
    m11 = load_py('11_improve_original_gb_features.py', 'm11_19')
    m16 = load_py('16_tune_gradient_boosting.py', 'm16_19')
    original = m11.load_original()
    df, dates, episodes = m11.prepare(original)
    OUT.mkdir(exist_ok=True, parents=True)
    dates.to_csv(OUT / 'split_by_date.csv', index=False, encoding='utf-8-sig')
    episodes.to_csv(OUT / 'episodes.csv', index=False, encoding='utf-8-sig')
    results, predictions, selected = [], [], []

    for hazard, months in HAZARDS.items():
        target = f'Y1_{hazard}'
        base_old = f'baseline_{hazard}'
        base_new, n_day = make_weekday_baseline(df, hazard, months)
        work = df.loc[df[hazard].eq(1) & df[target].notna()].copy()
        work['baseline_new'] = base_new[work.index]
        work['n_weekday_history'] = n_day[work.index]
        if work['baseline_new'].isna().any() or work['baseline_new'].le(0).any():
            raise ValueError(f'{hazard}: 새로운 기준매출이 누락되었거나 0 이하입니다.')
        work['Y1_weekday'] = 100 * (work['TS_AT'] / work['baseline_new'] - 1)
        parts = {s: work.loc[work['split'].eq(s)].copy() for s in ('train', 'validation', 'test')}
        if any(x.empty for x in parts.values()):
            raise ValueError(f'{hazard}: 빈 분할 존재')
        print(f'\n[{hazard}] ' + ', '.join(f'{s}={len(x):,}' for s, x in parts.items()))
        print(f"요일 기준매출 학습 관측 수 중앙값: {work['n_weekday_history'].median():.1f}")
        # 16번 선택된 고정 설정을 두 목표변수에 똑같이 적용.
        config = 'original_04' if hazard == 'heatwave' else 'subsample'
        params = m16.CONFIGS[config]
        feature_cols = m16.ALL_ADDED
        trained = {}
        for variant, y_col, base_col in [
            ('original_Y1', target, base_old),
            ('weekday_shrink_Y1', 'Y1_weekday', 'baseline_new'),
        ]:
            model = m16.make_pipeline(feature_cols, params)
            model.fit(parts['train'][feature_cols + KEY], parts['train'][y_col])
            trained[variant] = (model, y_col, base_col)
            val = parts['validation']
            pred_y = model.predict(val[feature_cols + KEY])
            pred_sales = val[base_col].to_numpy() * (1 + pred_y / 100)
            score = metrics(val['TS_AT'].to_numpy(), pred_sales, val[y_col].to_numpy(), pred_y)
            results.append({'hazard': hazard, 'variant': variant, 'split': 'validation', 'n': len(val), **score})
            print(f"  {variant:19s} Validation 매출 MAE={score['sales_MAE']:.3f} "
                  f"WAPE={score['sales_WAPE_pct']:.2f}% / 정의 내 Y1 MAE={score['Y1_MAE_pp_within_definition']:.3f}pp")
        candidates = [r for r in results if r['hazard'] == hazard and r['split'] == 'validation']
        winner = min(candidates, key=lambda r: r['sales_MAE'])['variant']
        selected.append({'hazard': hazard, 'selected_by_validation_sales_MAE': winner,
                         'fixed_gb_config': config})
        print('  Validation 매출 MAE 선택:', winner)
        # 두 후보 Test 출력은 진단 목적으로만 기록. 재선택하지 않음.
        test = parts['test']
        for variant, (model, y_col, base_col) in trained.items():
            pred_y = model.predict(test[feature_cols + KEY])
            pred_sales = test[base_col].to_numpy() * (1 + pred_y / 100)
            score = metrics(test['TS_AT'].to_numpy(), pred_sales, test[y_col].to_numpy(), pred_y)
            results.append({'hazard': hazard, 'variant': variant, 'split': 'test', 'n': len(test), **score})
            print(f"  TEST {variant:19s} 매출 MAE={score['sales_MAE']:.3f} "
                  f"WAPE={score['sales_WAPE_pct']:.2f}% / 정의 내 Y1 MAE={score['Y1_MAE_pp_within_definition']:.3f}pp")
            pred_frame = test[['date', *KEY, 'TS_AT', 'split']].copy()
            pred_frame['hazard'] = hazard
            pred_frame['variant'] = variant
            pred_frame['baseline'] = test[base_col].to_numpy()
            pred_frame['actual_Y1'] = test[y_col].to_numpy()
            pred_frame['predicted_Y1'] = pred_y
            pred_frame['predicted_sales'] = pred_sales
            predictions.append(pred_frame)

    pd.DataFrame(results).to_csv(OUT / 'comparison.csv', index=False, encoding='utf-8-sig')
    pd.DataFrame(selected).to_csv(OUT / 'validation_selection.csv', index=False, encoding='utf-8-sig')
    pd.concat(predictions, ignore_index=True).to_csv(OUT / 'test_predictions.csv', index=False, encoding='utf-8-sig')
    (OUT / 'experiment_config.json').write_text(json.dumps({
        'original_baseline': '04번 Train-only 해당 재난 0인 계절 지역×업종 전체 평균',
        'alternative_baseline': '같은 Train-only 표본의 지역×업종×요일 평균을 n/(n+5)로 수축',
        'weekday_shrink_k': SHRINK_K,
        'selection_metric': 'Validation 원매출 TS_AT MAE; 동일 매출 척도',
        'test_usage': '진단만; Test 기반 후보 재선택 금지',
        'split': '04번 Week+Episode (시간순 분할 아님)',
        'limitations': '재난별 독립 Test 지역-일 2개; 인과 피해 추정 또는 미래 예측 아님',
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'\n완료: {OUT}')
    print('주의: 목표변수 정의가 다르므로 Y1 MAE/R² 수치를 서로 우열 비교하지 마세요.')
    print('주의: 원매출 MAE 비교도 독립 Test 사건 수가 적어 불확실성이 큽니다.')


if __name__ == '__main__':
    main()
