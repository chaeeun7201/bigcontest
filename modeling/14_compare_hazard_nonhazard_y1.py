"""14번: 04/11과 동일한 Train-only 기준매출로 재난일과 비재난일의 Y1 비교.

실행: python modeling/14_compare_hazard_nonhazard_y1.py
필요: 04_train_weather_impact_models.py, 11_improve_original_gb_features.py,
      modeling_dataset_v2_filtered.csv (모두 modeling 폴더)
결과: modeling/weather_hazard_control_14/

주의: Y1은 재난일에만 정의된 기존 04번 타깃이므로, 비재난일 Y1은
      동일한 Train-only 기준매출을 사용하여 여기서 별도 계산한다.
      관찰 비교이지 인과적 재난 추가 효과의 추정치가 아니다.
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / '11_improve_original_gb_features.py'
OUT = ROOT / 'weather_hazard_control_14'
KEY = ['MCT_SGG_CD', 'MCT_RY_CD']
HAZARDS = {'heatwave': [7, 8, 9], 'cold_wave': [11, 12]}
SPLITS = ['validation', 'test']


def load_11():
    if not SOURCE.exists():
        raise FileNotFoundError(f'11번 코드가 필요합니다: {SOURCE}')
    spec = importlib.util.spec_from_file_location('weather14_source11', SOURCE)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def weighted_summary(df, by, label):
    records = []
    for keys, g in df.groupby(by, dropna=False, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        records.append(dict(zip(by, keys), comparison=label,
                            n=len(g), region_days=g[['MCT_SGG_CD', 'date']].drop_duplicates().shape[0],
                            mean_Y1=g['Y1_comparable'].mean(),
                            median_Y1=g['Y1_comparable'].median(),
                            negative_share=(g['Y1_comparable'] < 0).mean(),
                            total_sales=g['TS_AT'].sum(),
                            total_baseline=g['baseline_value'].sum()))
    return pd.DataFrame(records)


def main():
    print('=' * 72)
    print('14번: 동일 Train-only 기준매출로 재난일 vs 비재난일 매출 반응 비교')
    print('=' * 72)
    mod = load_11()
    df, dates, episodes = mod.prepare(mod.load_original())
    OUT.mkdir(parents=True, exist_ok=True)
    print(f'전체 데이터: {len(df):,}행')
    details, summaries, matched_all = [], [], []

    for hazard, months in HAZARDS.items():
        print(f'\n[{hazard}]')
        work = df[df['month'].isin(months) & df['split'].isin(SPLITS)].copy()
        # 다른 재난 플래그의 결측은 평시로 단정하지 않음.
        other = 'cold_wave' if hazard == 'heatwave' else 'heatwave'
        work['status'] = np.select(
            [work[hazard].eq(1), work[hazard].eq(0) & work[other].eq(0)],
            ['hazard', 'nonhazard'], default='unknown')
        work['baseline_value'] = pd.to_numeric(work[f'baseline_{hazard}'], errors='coerce')
        work['Y1_comparable'] = np.where(
            work['baseline_value'].gt(0),
            100 * (work['TS_AT'] - work['baseline_value']) / work['baseline_value'], np.nan)
        known = work[work['status'].isin(['hazard', 'nonhazard']) &
                     work['Y1_comparable'].notna()].copy()
        print('플래그 상태별 행 수:', work['status'].value_counts().to_dict())
        print('유효한 비교 행 수:', len(known))
        print('유효 재난/비재난:', known['status'].value_counts().to_dict())
        if known.empty:
            continue
        # 기존 04번의 재난일 Y1과 재계산 값 일치 확인.
        ev = known[known['status'].eq('hazard')]
        if len(ev):
            delta = (ev['Y1_comparable'] - ev[f'Y1_{hazard}']).abs()
            if delta.isna().any() or (delta > 1e-7 * np.maximum(1, ev[f'Y1_{hazard}'].abs())).any():
                raise ValueError(f'{hazard}: 기존 Y1과 재계산값 불일치')
        known['hazard_type'] = hazard
        details.append(known[['hazard_type', 'date', 'split', *KEY, 'month', 'weekday',
                              'status', 'TS_AT', 'baseline_value', 'Y1_comparable']])
        summ = weighted_summary(known, ['hazard_type', 'split', 'status'], 'all_valid')
        summaries.append(summ)
        print(summ[['split', 'status', 'n', 'region_days', 'median_Y1', 'negative_share']].to_string(index=False))

        # 같은 split, 지역, 업종, 요일의 비재난일을 대조로 사용.
        # 사건일이 적으므로 달력 월까지 강제하면 대조가 거의 없을 수 있음.
        controls = known[known['status'].eq('nonhazard')].copy()
        events = known[known['status'].eq('hazard')].copy()
        match_keys = ['split', *KEY, 'weekday']
        control_summary = controls.groupby(match_keys, as_index=False).agg(
            control_n=('Y1_comparable', 'size'),
            control_median_Y1=('Y1_comparable', 'median'),
            control_mean_Y1=('Y1_comparable', 'mean'),
            control_first_date=('date', 'min'), control_last_date=('date', 'max'))
        paired = events.merge(control_summary, on=match_keys, how='left', validate='many_to_one')
        paired['extra_gap_pp'] = paired['Y1_comparable'] - paired['control_median_Y1']
        paired['matched'] = paired['control_n'].ge(1) & paired['extra_gap_pp'].notna()
        paired['hazard_type'] = hazard
        paired['control_may_be_future'] = paired['control_last_date'] > paired['date']
        paired['matched_control_days_min3'] = paired['control_n'].ge(3)
        matched_all.append(paired[['hazard_type', 'split', 'date', *KEY, 'weekday',
                                   'TS_AT', 'baseline_value', 'Y1_comparable',
                                   'control_n', 'control_median_Y1', 'extra_gap_pp',
                                   'matched', 'matched_control_days_min3',
                                   'control_first_date', 'control_last_date', 'control_may_be_future']])
        ok = paired[paired['matched']]
        print(f'요일·지역·업종 매칭: {len(ok):,}/{len(paired):,}행 '
              f'(대조 3일 이상: {int(ok["matched_control_days_min3"].sum()):,}행)')
        if len(ok):
            print(f'추가 차이 중앙값: {ok["extra_gap_pp"].median():+.2f}%p / '
                  f'추가 감소 비율: {(ok["extra_gap_pp"] < 0).mean():.1%}')
            print(f'비재난 대조일에 사건일 이후 날짜 포함: '
                  f'{int(ok["control_may_be_future"].sum()):,}행')

    if not details:
        print('비교 가능한 자료가 없습니다. 재난 플래그와 split을 확인하세요.')
        return
    pd.concat(details, ignore_index=True).to_csv(
        OUT / 'comparable_hazard_nonhazard_y1.csv', index=False, encoding='utf-8-sig')
    pd.concat(summaries, ignore_index=True).to_csv(
        OUT / 'status_summary.csv', index=False, encoding='utf-8-sig')
    if matched_all:
        paired = pd.concat(matched_all, ignore_index=True)
        paired.to_csv(OUT / 'matched_weekday_comparison.csv', index=False, encoding='utf-8-sig')
        valid = paired[paired['matched']].copy()
        if len(valid):
            # 업종 순위는 최소 2개의 서로 다른 지역·날짜에 관측된 후보만 별도 출력.
            groups = valid.groupby(['hazard_type', *KEY], as_index=False).agg(
                observed_days=('date', 'nunique'),
                extra_gap_median_pp=('extra_gap_pp', 'median'),
                extra_negative_days=('extra_gap_pp', lambda s: int((s < 0).sum())),
                control_n_median=('control_n', 'median'),
                future_control_share=('control_may_be_future', 'mean'))
            groups['exploratory_repeat_candidate'] = (
                (groups['observed_days'] >= 2) & (groups['extra_negative_days'] >= 2))
            groups.sort_values(['hazard_type', 'extra_gap_median_pp']).to_csv(
                OUT / 'exploratory_industry_candidates.csv', index=False, encoding='utf-8-sig')
            print('\n[재난별 매칭 요약]')
            for hazard, g in valid.groupby('hazard_type'):
                print(f'{hazard}: {len(g):,}행 / 지역·날짜 '
                      f'{g[["MCT_SGG_CD", "date"]].drop_duplicates().shape[0]}개 / '
                      f'추가 차이 중앙값 {g["extra_gap_pp"].median():+.2f}%p')
    print(f'\n완료: {OUT}')
    print('주의 1: 비교값은 인과적 재난 추가 피해액이 아닙니다.')
    print('주의 2: 동일 split 내 비재난 대조일은 사건일보다 미래일 수 있습니다.')
    print('주의 3: 비재난일의 다른 재난 플래그가 결측이면 대조군에서 제외했습니다.')
    print('주의 4: 04번 split은 시간순 분할이 아니며, 사건 독립 표본이 적습니다.')
    print('주의 5: 14번은 새 모델 학습이나 R² 개선 실험이 아닌 비교 가능성 진단입니다.')


if __name__ == '__main__':
    main()
