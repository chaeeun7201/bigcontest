"""20번: 19번 기준매출 개선의 사건(Episode)별 안정성 점검.

실행: python modeling/20_check_episode_baseline_stability.py
필요: 04, 11, 16, 19번 .py 및 modeling_dataset_v2_filtered.csv

19번과 동일한 데이터·분할·모델 설정으로 Validation/Test 예측을 다시 만들고,
같은 재난 지역의 연속된 재난일을 하나의 Episode로 묶어 원매출 MAE/WAPE를 비교한다.
Test 결과로 모델을 다시 선택하지 않는다. 시간순 미래 예측 검증이 아니다.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / 'weather_episode_stability_20'
KEY = ['MCT_SGG_CD', 'MCT_RY_CD']


def load_module(filename, name):
    path = HERE / filename
    if not path.exists():
        raise FileNotFoundError(f'필요한 파일이 없습니다: {path}')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scores(g):
    actual = g['actual_sales'].to_numpy(dtype=float)
    pred = g['predicted_sales'].to_numpy(dtype=float)
    ae = np.abs(actual - pred)
    denom = np.abs(actual).sum()
    return pd.Series({
        'rows': len(g), 'region_days': g[['region', 'date']].drop_duplicates().shape[0],
        'sales_MAE': float(ae.mean()),
        'sales_WAPE_pct': float(100 * ae.sum() / denom) if denom > 0 else np.nan,
        'absolute_error_sum': float(ae.sum()), 'actual_sales_sum': float(denom),
    })


def main():
    print('20번: 요일 보정 기준매출의 재난 Episode별 안정성 분석')
    m11 = load_module('11_improve_original_gb_features.py', 'm11_20')
    m16 = load_module('16_tune_gradient_boosting.py', 'm16_20')
    m19 = load_module('19_compare_baseline_targets.py', 'm19_20')
    original = m11.load_original()
    df, dates, episodes = m11.prepare(original)
    OUT.mkdir(parents=True, exist_ok=True)
    episode_rows = episodes.copy()
    episode_rows['start_date'] = pd.to_datetime(episode_rows['start_date'])
    episode_rows['end_date'] = pd.to_datetime(episode_rows['end_date'])
    all_preds = []

    for hazard, months in m19.HAZARDS.items():
        target = f'Y1_{hazard}'
        baseline_old = f'baseline_{hazard}'
        new_base, _ = m19.make_weekday_baseline(df, hazard, months)
        work = df.loc[df[hazard].eq(1) & df[target].notna()].copy()
        work['baseline_new'] = new_base[work.index]
        if work['baseline_new'].isna().any() or work['baseline_new'].le(0).any():
            raise ValueError(f'{hazard}: 기준매출 누락 또는 0 이하')
        work['Y1_weekday'] = 100 * (work['TS_AT'] / work['baseline_new'] - 1)
        train = work.loc[work['split'].eq('train')]
        config = 'original_04' if hazard == 'heatwave' else 'subsample'
        variants = [('original_Y1', target, baseline_old),
                    ('weekday_shrink_Y1', 'Y1_weekday', 'baseline_new')]
        print(f'\n[{hazard}] Train={len(train):,}행')
        for variant, ycol, bcol in variants:
            model = m16.make_pipeline(m16.ALL_ADDED, m16.CONFIGS[config])
            model.fit(train[m16.ALL_ADDED + KEY], train[ycol])
            for split in ('validation', 'test'):
                part = work.loc[work['split'].eq(split)].copy()
                if part.empty:
                    continue
                predicted_y = model.predict(part[m16.ALL_ADDED + KEY])
                frame = pd.DataFrame({
                    'hazard': hazard, 'variant': variant, 'split': split,
                    'date': part['date'].to_numpy(),
                    'region': part[KEY[0]].to_numpy(),
                    'industry': part[KEY[1]].to_numpy(),
                    'actual_sales': part['TS_AT'].to_numpy(dtype=float),
                    'predicted_sales': part[bcol].to_numpy(dtype=float) * (1 + predicted_y / 100),
                })
                all_preds.append(frame)

    pred = pd.concat(all_preds, ignore_index=True)
    pred['episode_id'] = pd.NA
    for e in episode_rows.itertuples(index=False):
        mask = (pred['hazard'].eq(e.hazard) & pred['region'].eq(e.region)
                & pred['date'].between(e.start_date, e.end_date))
        pred.loc[mask, 'episode_id'] = e.episode_id
    if pred['episode_id'].isna().any():
        missing = pred.loc[pred['episode_id'].isna(), ['hazard', 'region', 'date']].drop_duplicates()
        raise ValueError('Episode 매핑 실패:\n' + missing.to_string(index=False))
    pred['abs_error'] = (pred['actual_sales'] - pred['predicted_sales']).abs()
    pred.to_csv(OUT / 'validation_test_predictions.csv', index=False, encoding='utf-8-sig')

    event = pred.groupby(['hazard', 'split', 'episode_id', 'variant'], sort=True).apply(
        scores, include_groups=False).reset_index()
    meta = episode_rows[['episode_id', 'region', 'start_date', 'end_date', 'days']]
    event = event.merge(meta, on='episode_id', how='left', validate='many_to_one')
    event.to_csv(OUT / 'episode_metrics.csv', index=False, encoding='utf-8-sig')

    wide = event.pivot(index=['hazard', 'split', 'episode_id', 'region', 'start_date', 'end_date', 'days'],
                       columns='variant', values=['sales_MAE', 'sales_WAPE_pct', 'rows']).reset_index()
    wide.columns = ['_'.join(str(v) for v in col if v) for col in wide.columns.to_flat_index()]
    wide['MAE_improvement_pct'] = 100 * (wide['sales_MAE_original_Y1'] - wide['sales_MAE_weekday_shrink_Y1']) / wide['sales_MAE_original_Y1'].replace(0, np.nan)
    wide['WAPE_improvement_pp'] = wide['sales_WAPE_pct_original_Y1'] - wide['sales_WAPE_pct_weekday_shrink_Y1']
    wide['improved'] = wide['sales_MAE_weekday_shrink_Y1'] < wide['sales_MAE_original_Y1']
    wide.to_csv(OUT / 'episode_before_after.csv', index=False, encoding='utf-8-sig')

    overview = pred.groupby(['hazard', 'split', 'variant'], sort=True).apply(
        scores, include_groups=False).reset_index()
    overview.to_csv(OUT / 'split_overview.csv', index=False, encoding='utf-8-sig')
    stability = wide.groupby(['hazard', 'split'], as_index=False).agg(
        independent_episodes=('episode_id', 'nunique'),
        improved_episodes=('improved', 'sum'),
        median_episode_MAE_improvement_pct=('MAE_improvement_pct', 'median'),
        worst_episode_MAE_improvement_pct=('MAE_improvement_pct', 'min'))
    stability['improved_episode_share'] = stability['improved_episodes'] / stability['independent_episodes']
    stability.to_csv(OUT / 'episode_stability_summary.csv', index=False, encoding='utf-8-sig')

    print('\n[전체 행 기준 성능: 같은 관측에 대한 실제 매출 비교]')
    print(overview[['hazard', 'split', 'variant', 'rows', 'sales_MAE', 'sales_WAPE_pct']].round(3).to_string(index=False))
    print('\n[독립 Episode별 개선 여부]')
    print(wide[['hazard', 'split', 'episode_id', 'region', 'start_date', 'end_date',
                'MAE_improvement_pct', 'improved']].round(2).to_string(index=False))
    print('\n[Episode 안정성 요약]')
    print(stability.round(3).to_string(index=False))
    print(f'\n완료: {OUT}')
    print('주의: Test는 모델 선택에 재사용하지 마세요. 사건 수가 적으면 개선의 일반화는 확인되지 않습니다.')
    print('주의: 같은 사건 안의 업종별 행은 독립 재난 사건이 아닙니다.')
    print('주의: 04번 분할은 미래 날짜만 평가하는 시간순 검증이 아닙니다.')
    (OUT / 'readme.json').write_text(json.dumps({
        'purpose': '19번 원매출 오차 개선을 독립 지역별 연속 재난 Episode 단위로 재확인',
        'model_selection': '19번 Validation 선택 유지; Test 재선택 금지',
        'unit': 'hazard × region × contiguous event days',
        'limitations': '작은 Episode 수, 임의 Week/Episode 분할, 인과 추정/미래 예측 아님',
    }, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
