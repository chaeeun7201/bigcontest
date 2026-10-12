"""15번: 14번의 비재난일 0행 문제 수정 및 재난/비재난 Y1 비교.

실행: python modeling/15_fix_nonhazard_controls.py
필요: 04_train_weather_impact_models.py, 11_improve_original_gb_features.py,
      modeling_dataset_v2_filtered.csv (모두 modeling 폴더)
출력: modeling/weather_hazard_control_15/

핵심: 해당 재난 플래그 == 0만 비교군으로 인정한다. 다른 계절 재난 플래그의
NaN을 0으로 변환하지 않으며, 다른 재난의 상태가 미확인임을 따로 표시한다.
비교는 관찰적 차이이며 인과 효과 또는 미래 예측 성능이 아니다.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / '11_improve_original_gb_features.py'
OUT = ROOT / 'weather_hazard_control_15'
KEY = ['MCT_SGG_CD', 'MCT_RY_CD']
HAZARDS = {'heatwave': [7, 8, 9], 'cold_wave': [11, 12]}
MIN_CONTROLS = 3


def load_11():
    if not SOURCE.exists():
        raise FileNotFoundError(f'11번 파일이 없습니다: {SOURCE}')
    spec = importlib.util.spec_from_file_location('source11_for15', SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def aggregate_controls(df, match_keys):
    return df.groupby(match_keys, as_index=False, dropna=False).agg(
        control_n=('Y1_comparable', 'size'),
        control_median_Y1=('Y1_comparable', 'median'),
        control_first_date=('date', 'min'),
        control_last_date=('date', 'max'))


def main():
    print('=' * 75)
    print('15번: 재난 플래그 결측 처리 수정 / 재난일 vs 비재난일 비교')
    print('=' * 75)
    mod = load_11()
    df, _, _ = mod.prepare(mod.load_original())
    OUT.mkdir(parents=True, exist_ok=True)
    all_rows, matched_rows, diagnostics = [], [], []

    for hazard, months in HAZARDS.items():
        other = 'cold_wave' if hazard == 'heatwave' else 'heatwave'
        work = df.loc[df['month'].isin(months) & df['split'].isin(['validation', 'test'])].copy()
        work['status'] = np.select(
            [work[hazard].eq(1), work[hazard].eq(0)],
            ['hazard', 'nonhazard'], default='unknown')
        work['other_hazard_unknown'] = work[other].isna()
        work['other_hazard_positive'] = work[other].eq(1)
        work['baseline_value'] = pd.to_numeric(work[f'baseline_{hazard}'], errors='coerce')
        work['Y1_comparable'] = np.where(
            work['baseline_value'].gt(0),
            100 * (work['TS_AT'] - work['baseline_value']) / work['baseline_value'], np.nan)
        work['hazard_type'] = hazard
        print(f'\n[{hazard}]')
        print('해당 재난 플래그:', work['status'].value_counts(dropna=False).to_dict())
        print('다른 재난 플래그 NaN:', int(work['other_hazard_unknown'].sum()),
              '/ 1:', int(work['other_hazard_positive'].sum()))
        valid = work.loc[work['status'].ne('unknown') & work['Y1_comparable'].notna()].copy()
        print('유효 재난/비재난:', valid['status'].value_counts().to_dict())
        all_rows.append(valid[['hazard_type', 'split', 'date', *KEY, 'weekday', 'status',
                               'other_hazard_unknown', 'other_hazard_positive',
                               'TS_AT', 'baseline_value', 'Y1_comparable']])
        for split in ['validation', 'test']:
            subset = valid.loc[valid['split'].eq(split)]
            ev = subset.loc[subset['status'].eq('hazard')].copy()
            co = subset.loc[subset['status'].eq('nonhazard')].copy()
            # 같은 split, 같은 지역·업종·요일. 달력 날짜가 앞선 대조일만 사용한
            # 별도 엄격한 결과도 아래에서 함께 계산한다.
            print(f' {split}: 재난 {len(ev):,}행, 비재난 {len(co):,}행')
            if ev.empty:
                continue
            match_keys = [*KEY, 'weekday']
            control_stats = aggregate_controls(co, match_keys)
            paired = ev.merge(control_stats, on=match_keys, how='left', validate='many_to_one')
            paired['extra_gap_pp'] = paired['Y1_comparable'] - paired['control_median_Y1']
            paired['matched_1plus'] = paired['control_n'].ge(1) & paired['extra_gap_pp'].notna()
            paired['matched_3plus'] = paired['control_n'].ge(MIN_CONTROLS) & paired['extra_gap_pp'].notna()
            paired['future_control_possible'] = paired['control_last_date'] > paired['date']
            # 사후 설명용: 미래 대조일 포함 가능. 별도로 사건일 이전 대조일만 사용.
            co_groups = {key: g.sort_values('date') for key, g in co.groupby(match_keys)}
            prior_n, prior_median = [], []
            for row in ev.itertuples(index=False):
                key = tuple(getattr(row, k) for k in match_keys)
                history = co_groups.get(key)
                if history is None:
                    vals = pd.Series(dtype=float)
                else:
                    vals = history.loc[history['date'] < row.date, 'Y1_comparable']
                prior_n.append(len(vals))
                prior_median.append(vals.median() if len(vals) else np.nan)
            prior = ev[[*KEY, 'weekday', 'date']].copy()
            prior['prior_control_n'] = prior_n
            prior['prior_control_median_Y1'] = prior_median
            paired = paired.merge(prior, on=[*KEY, 'weekday', 'date'], how='left', validate='one_to_one')
            paired['prior_extra_gap_pp'] = paired['Y1_comparable'] - paired['prior_control_median_Y1']
            paired['prior_matched_3plus'] = paired['prior_control_n'].ge(MIN_CONTROLS)
            matched_rows.append(paired)
            diagnostics.append({
                'hazard': hazard, 'split': split,
                'hazard_rows': len(ev), 'nonhazard_rows': len(co),
                'matched_1plus': int(paired['matched_1plus'].sum()),
                'matched_3plus': int(paired['matched_3plus'].sum()),
                'prior_matched_3plus': int(paired['prior_matched_3plus'].sum()),
                'future_control_possible': int((paired['matched_1plus'] & paired['future_control_possible']).sum()),
                'median_extra_gap_pp_3plus': paired.loc[paired['matched_3plus'], 'extra_gap_pp'].median(),
                'median_prior_extra_gap_pp_3plus': paired.loc[paired['prior_matched_3plus'], 'prior_extra_gap_pp'].median(),
            })
            print(f'  매칭 1일 이상 {int(paired["matched_1plus"].sum()):,}/{len(paired):,}, '
                  f'3일 이상 {int(paired["matched_3plus"].sum()):,}, '
                  f'사건 이전 3일 이상 {int(paired["prior_matched_3plus"].sum()):,}')
            if paired['matched_3plus'].any():
                print(f'  비교 Y1 차이 중앙값: '
                      f'{paired.loc[paired["matched_3plus"], "extra_gap_pp"].median():+.2f}%p')

    if all_rows:
        pd.concat(all_rows, ignore_index=True).to_csv(
            OUT / 'valid_hazard_nonhazard_y1.csv', index=False, encoding='utf-8-sig')
    if matched_rows:
        all_matched = pd.concat(matched_rows, ignore_index=True)
        all_matched.to_csv(OUT / 'matched_comparison.csv', index=False, encoding='utf-8-sig')
        # 업종 탐색: 반복 날짜는 독립 사건이 아닐 수 있음.
        selected = all_matched.loc[all_matched['matched_3plus']].copy()
        if not selected.empty:
            result = selected.groupby(['hazard_type', *KEY], as_index=False).agg(
                observed_days=('date', 'nunique'),
                median_extra_gap_pp=('extra_gap_pp', 'median'),
                negative_days=('extra_gap_pp', lambda s: int((s < 0).sum())),
                median_control_n=('control_n', 'median'))
            result.sort_values(['hazard_type', 'median_extra_gap_pp']).to_csv(
                OUT / 'exploratory_industry_gaps.csv', index=False, encoding='utf-8-sig')
    if diagnostics:
        summary = pd.DataFrame(diagnostics)
        summary.to_csv(OUT / 'matching_diagnostics.csv', index=False, encoding='utf-8-sig')
        print('\n[매칭 요약]')
        print(summary.to_string(index=False))
    print(f'\n완료: {OUT}')
    print('주의: 다른 재난 플래그 NaN을 0으로 대체하지 않았습니다.')
    print('주의: 같은 재난 플래그 0도 관측 범위·정의가 타당한지 확인해야 합니다.')
    print('주의: 04번 split은 시간순이 아니며, 미래 대조일이 포함될 수 있습니다.')
    print('주의: 추가 차이는 인과적 재난 피해액이 아닙니다.')


if __name__ == '__main__':
    main()
