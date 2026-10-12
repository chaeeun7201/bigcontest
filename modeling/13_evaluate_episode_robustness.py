"""13번: 11/12번 모델의 지역·재난일 및 업종별 안정성 진단.

실행: python modeling/13_evaluate_episode_robustness.py
입력: modeling/weather_feature_audit_12/verified_test_predictions.csv
출력: modeling/weather_episode_robustness_13/

중요: 테스트의 독립 지역-일 수가 매우 적으므로, 통계적 일반화/인과적 피해를
검증하는 도구가 아니라 취약 후보와 오차의 민감도를 기술적으로 점검하는 도구임.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / 'weather_feature_audit_12' / 'verified_test_predictions.csv'
OUT = ROOT / 'weather_episode_robustness_13'
REGION_COL = 'MCT_SGG_CD'
INDUSTRY_COL = 'MCT_RY_CD'
TARGETS = {'heatwave': 'Y1_heatwave', 'cold_wave': 'Y1_cold_wave'}
# 11번에서 Validation MAE로 선정한 모델. Test 결과로 모델을 새로 고르지 않음.
SELECTED_VARIANT = 'sales_ratios'


def load():
    if not INPUT.exists():
        raise FileNotFoundError(f'12번을 먼저 실행하세요. 파일 없음: {INPUT}')
    df = pd.read_csv(INPUT, low_memory=False)
    required = ['hazard', 'variant', 'date', REGION_COL, INDUSTRY_COL, 'predicted_Y1']
    required += list(TARGETS.values())
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f'필수 열 누락: {missing}\n현재 열: {list(df.columns)}')
    df['date'] = pd.to_datetime(df['date'], errors='raise')
    for col in [REGION_COL, INDUSTRY_COL]:
        df[col] = df[col].astype(str).str.replace(r'\.0$', '', regex=True)
    for col in ['predicted_Y1', *TARGETS.values()]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df['actual_Y1'] = np.nan
    for hazard, target in TARGETS.items():
        m = df['hazard'].eq(hazard)
        df.loc[m, 'actual_Y1'] = df.loc[m, target]
    df = df.dropna(subset=['actual_Y1', 'predicted_Y1']).copy()
    if df.duplicated(['hazard', 'variant', 'date', REGION_COL, INDUSTRY_COL]).any():
        raise ValueError('동일 재난·모델·날짜·지역·업종 중복행이 있습니다.')
    df['error_pp'] = df['predicted_Y1'] - df['actual_Y1']
    df['abs_error_pp'] = df['error_pp'].abs()
    df['actual_decrease'] = df['actual_Y1'] < 0
    df['pred_decrease'] = df['predicted_Y1'] < 0
    df['direction_correct'] = df['actual_decrease'] == df['pred_decrease']
    return df


def score(g):
    y, p = g['actual_Y1'], g['predicted_Y1']
    return {'rows': len(g), 'MAE_pp': mean_absolute_error(y, p),
            'R2': r2_score(y, p) if len(g) >= 2 and y.nunique() > 1 else np.nan,
            'direction_accuracy': g['direction_correct'].mean(),
            'always_decrease_accuracy': g['actual_decrease'].mean(),
            'mean_error_pp': g['error_pp'].mean(),
            'median_error_pp': g['error_pp'].median(),
            'actual_median_Y1': y.median(), 'predicted_median_Y1': p.median()}


def main():
    df = load()
    OUT.mkdir(exist_ok=True, parents=True)
    print('=' * 74)
    print('13번: 재난 지역·일별 안정성 및 취약 업종 후보 진단')
    print('=' * 74)
    print(f'검증 데이터: {len(df):,}행 (모델별 중복 포함)')

    group_rows = []
    for (hazard, variant, region, date), g in df.groupby(
        ['hazard', 'variant', REGION_COL, 'date'], sort=True
    ):
        group_rows.append({'hazard': hazard, 'variant': variant,
                           'region': region, 'date': date, **score(g)})
    group_scores = pd.DataFrame(group_rows)
    group_scores.to_csv(OUT / 'region_day_metrics.csv', index=False, encoding='utf-8-sig')

    # 지역·날짜 하나씩 제외한 민감도. 표본이 적어 '교차검증'이나 새 모델 학습이 아님.
    leave_rows = []
    for (hazard, variant), g in df.groupby(['hazard', 'variant']):
        unique_days = g[[REGION_COL, 'date']].drop_duplicates()
        for day in unique_days.itertuples(index=False, name=None):
            region, date = day
            retained = g[~(g[REGION_COL].eq(region) & g['date'].eq(date))]
            if retained.empty:
                continue
            leave_rows.append({'hazard': hazard, 'variant': variant,
                               'excluded_region': region, 'excluded_date': date,
                               'remaining_region_days': len(unique_days)-1,
                               **score(retained)})
    leave = pd.DataFrame(leave_rows)
    leave.to_csv(OUT / 'leave_one_region_day_out.csv', index=False, encoding='utf-8-sig')

    # 원래 목표인 '취약 업종'은 관측 Y1 감소의 반복성으로만 탐색한다.
    # 같은 지역·날짜는 여러 업종 행이더라도 독립 사건 1회임.
    selected = df[df['variant'].eq(SELECTED_VARIANT)].copy()
    industry_rows = []
    for (hazard, region, industry), g in selected.groupby(
        ['hazard', REGION_COL, INDUSTRY_COL], sort=True
    ):
        n_region_days = len(g[['date']].drop_duplicates())
        industry_rows.append({
            'hazard': hazard, 'region': region, 'industry': industry,
            'observed_region_days': n_region_days,
            'actual_negative_days': int(g['actual_decrease'].sum()),
            'predicted_negative_days': int(g['pred_decrease'].sum()),
            'correct_direction_days': int(g['direction_correct'].sum()),
            'actual_median_Y1': g['actual_Y1'].median(),
            'predicted_median_Y1': g['predicted_Y1'].median(),
            'MAE_pp': g['abs_error_pp'].mean(),
            'all_observed_days_negative': bool(g['actual_decrease'].all()),
            'exploratory_candidate': bool(n_region_days >= 2 and g['actual_decrease'].all()),
        })
    industry = pd.DataFrame(industry_rows)
    industry = industry.sort_values(['hazard', 'exploratory_candidate',
                                     'actual_median_Y1'], ascending=[True, False, True])
    industry.to_csv(OUT / 'industry_decrease_candidates.csv', index=False, encoding='utf-8-sig')

    summary = []
    for (hazard, variant), g in df.groupby(['hazard', 'variant']):
        rg = group_scores[(group_scores.hazard == hazard) & (group_scores.variant == variant)]
        summary.append({'hazard': hazard, 'variant': variant,
                        'independent_region_days': len(rg), **score(g),
                        'worst_region_day_MAE_pp': rg['MAE_pp'].max(),
                        'min_region_day_direction_accuracy': rg['direction_accuracy'].min()})
    summary = pd.DataFrame(summary)
    summary.to_csv(OUT / 'model_stability_summary.csv', index=False, encoding='utf-8-sig')

    print('\n[모델별 안정성]')
    print(summary[['hazard','variant','independent_region_days','MAE_pp','R2',
                   'direction_accuracy','always_decrease_accuracy',
                   'worst_region_day_MAE_pp']].to_string(index=False, float_format=lambda x:f'{x:.3f}'))
    print('\n[선택 모델의 지역·날짜별 결과]')
    print(group_scores[group_scores.variant.eq(SELECTED_VARIANT)][
        ['hazard','region','date','rows','MAE_pp','direction_accuracy',
         'actual_median_Y1','predicted_median_Y1']].to_string(index=False, float_format=lambda x:f'{x:.2f}'))
    print('\n[실제 Y1 감소가 관측된 탐색 후보 상위 15개]')
    candidates = industry[industry.exploratory_candidate]
    if candidates.empty:
        print('조건 충족 후보 없음')
    else:
        print(candidates[['hazard','region','industry','observed_region_days',
                          'actual_negative_days','actual_median_Y1','MAE_pp']].head(15).to_string(index=False))
    print(f'\n완료: {OUT}')
    print('주의: 테스트 재난 지역·날짜가 극히 적어 반복성·일반화 성능은 확정할 수 없습니다.')
    print('주의: 음수 Y1은 기준 대비 감소이며, 재난에 의한 추가 피해 또는 인과 효과가 아닙니다.')
    print('주의: 13번은 모델을 재학습하거나 테스트 결과로 최종 모델을 재선택하지 않습니다.')


if __name__ == '__main__':
    main()
