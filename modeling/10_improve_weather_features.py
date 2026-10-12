"""10번: 폭염/한파 매출 변화율 모델의 입력변수 단계별 개선 실험.

실행: python modeling/10_improve_weather_features.py
주의: 기존 04/53번과 완전히 동일한 Y1·분할을 보장하지 않으므로
      수치 직접 비교가 아닌 이 파일 내 동일 분할 후보 비교에 사용합니다.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'modeling_dataset_v2_filtered.csv'
OUT = ROOT / 'weather_feature_experiment_10'
HAZARDS = {'heatwave': [7, 8, 9], 'cold_wave': [11, 12]}
RANDOM_STATE = 42


def load():
    if not DATA.exists():
        raise FileNotFoundError(f'입력 데이터가 없습니다: {DATA}')
    df = pd.read_csv(DATA, low_memory=False)
    needed = ['TA_YMD', 'MCT_SGG_CD', 'MCT_RY_CD', 'TS_AT', 'heatwave', 'cold_wave']
    missing = [c for c in needed if c not in df]
    if missing:
        raise ValueError(f'필수 열 누락: {missing}')
    dates = df['TA_YMD'].astype(str).str.replace(r'\.0$', '', regex=True)
    df['date'] = pd.to_datetime(dates, format='%Y%m%d', errors='coerce')
    bad = df['date'].isna()
    if bad.any():
        df.loc[bad, 'date'] = pd.to_datetime(dates[bad], errors='coerce')
    df['sales'] = pd.to_numeric(df['TS_AT'], errors='coerce')
    for c in ['heatwave', 'cold_wave']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df['region'] = df['MCT_SGG_CD'].astype(str)
    df['industry'] = df['MCT_RY_CD'].astype(str)
    df = df.dropna(subset=['date', 'sales', 'region', 'industry'])
    df = df[df['sales'] >= 0].copy()
    if df.duplicated(['date', 'region', 'industry']).any():
        raise ValueError('지역×업종×날짜 중복 데이터가 있습니다.')
    df = df.sort_values(['region', 'industry', 'date']).reset_index(drop=True)
    df['weekday'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day_index'] = (df['date'] - df['date'].min()).dt.days
    # 모두 전일 이전 매출만 사용: 현재 날짜의 매출 누수 방지.
    group = df.groupby(['region', 'industry'], sort=False)['sales']
    df['lag1'] = group.shift(1)
    df['lag7'] = group.shift(7)
    df['rolling7'] = group.transform(lambda s: s.shift(1).rolling(7, min_periods=3).median())
    df['rolling28'] = group.transform(lambda s: s.shift(1).rolling(28, min_periods=7).median())
    df['lag7_vs_rolling28'] = 100 * (df['lag7'] - df['rolling28']) / df['rolling28'].replace(0, np.nan)
    df['rolling7_vs_rolling28'] = 100 * (df['rolling7'] - df['rolling28']) / df['rolling28'].replace(0, np.nan)
    # 날짜별 지역 기온의 과거 28일 평균 대비 편차(당일 기온 사용: 사후 설명 목적).
    for c in ['avg_temp', 'min_temp', 'max_temp']:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    if 'avg_temp' in df:
        weather = df[['region', 'date', 'avg_temp']].drop_duplicates(['region', 'date']).sort_values(['region', 'date'])
        weather['temp_prev28'] = weather.groupby('region')['avg_temp'].transform(
            lambda s: s.shift(1).rolling(28, min_periods=7).mean())
        weather['temp_anomaly'] = weather['avg_temp'] - weather['temp_prev28']
        df = df.merge(weather[['region', 'date', 'temp_anomaly']], on=['region', 'date'], how='left', validate='many_to_one')
    return df


def make_pipeline(cats, nums):
    prep = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore'), cats),
        ('num', SimpleImputer(strategy='median', add_indicator=True), nums),
    ])
    return Pipeline([
        ('prep', prep),
        ('model', GradientBoostingRegressor(n_estimators=200, learning_rate=0.04,
                                            max_depth=2, loss='huber', random_state=RANDOM_STATE))
    ])


def main():
    print('10번: Gradient Boosting 입력변수 개선 (동일 분할·동일 목표변수 내 비교)')
    df = load()
    print(f'전체 데이터: {len(df):,}행')
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    preds = []
    for hazard, months in HAZARDS.items():
        sub = df[df['month'].isin(months)].copy()
        # 재난일 지역-날짜 기준으로 시계열 분할. 같은 날짜는 여러 분할에 들어가지 않음.
        event_dates = sorted(sub.loc[sub[hazard].eq(1), 'date'].unique())
        if len(event_dates) < 6:
            print(f'[{hazard}] 재난 날짜 {len(event_dates)}개: 검증에 부족하여 건너뜀')
            continue
        n = len(event_dates)
        i = max(1, int(n * 0.65))
        j = max(i + 1, int(n * 0.85))
        if j >= n:
            j = n - 1
        train_dates, val_dates, test_dates = event_dates[:i], event_dates[i:j], event_dates[j:]
        first_val = pd.Timestamp(val_dates[0])
        # Y1의 평시 기준은 학습 시점 이전의 동일 지역×업종 비재난일로만 계산.
        baseline_rows = sub[(sub['date'] < first_val) & sub[hazard].eq(0)]
        base = baseline_rows.groupby(['region', 'industry'])['sales'].median().rename('baseline_sales')
        sub = sub.join(base, on=['region', 'industry'])
        sub = sub[sub[hazard].eq(1) & sub['baseline_sales'].gt(0)].copy()
        sub['Y1'] = 100 * (sub['sales'] - sub['baseline_sales']) / sub['baseline_sales']
        train = sub[sub['date'].isin(train_dates)]
        val = sub[sub['date'].isin(val_dates)]
        test = sub[sub['date'].isin(test_dates)]
        if min(len(train), len(val), len(test)) < 2:
            print(f'[{hazard}] 유효 데이터 부족: train={len(train)}, val={len(val)}, test={len(test)}')
            continue
        cats = ['region', 'industry']
        original = [c for c in ['avg_temp', 'min_temp', 'max_temp', 'lag1', 'lag7', 'rolling7', 'rolling28'] if c in sub]
        variants = {
            'base': original,
            'calendar': original + ['weekday', 'day_index'],
            'sales_trend': original + ['lag7_vs_rolling28', 'rolling7_vs_rolling28'],
            'calendar_and_trend': original + ['weekday', 'day_index', 'lag7_vs_rolling28', 'rolling7_vs_rolling28'],
        }
        if 'temp_anomaly' in sub:
            variants['all_features'] = variants['calendar_and_trend'] + ['temp_anomaly']
        models = {}
        print(f'\n[{hazard}] train={len(train):,}, validation={len(val):,}, test={len(test):,}')
        print(f'분할 날짜: {pd.Timestamp(train_dates[-1]).date()} / {pd.Timestamp(val_dates[-1]).date()} / {pd.Timestamp(test_dates[-1]).date()}')
        for name, nums in variants.items():
            model = make_pipeline(cats, nums)
            model.fit(train[cats + nums], train['Y1'])
            pred = model.predict(val[cats + nums])
            mae = mean_absolute_error(val['Y1'], pred)
            r2 = r2_score(val['Y1'], pred)
            results.append({'hazard': hazard, 'variant': name, 'split': 'validation',
                            'rows': len(val), 'MAE_pp': mae, 'R2': r2})
            models[name] = (model, nums, mae)
            print(f'{name:20s} Val MAE={mae:.3f}pp R²={r2:.3f}')
        winner = min(models, key=lambda k: models[k][2])
        # 공정한 비교를 위해 기존 base와 선택 모델만 Test에서 평가.
        for name in dict.fromkeys(['base', winner]):
            model, nums, _ = models[name]
            predicted = model.predict(test[cats + nums])
            mae = mean_absolute_error(test['Y1'], predicted)
            r2 = r2_score(test['Y1'], predicted)
            results.append({'hazard': hazard, 'variant': name, 'split': 'test',
                            'rows': len(test), 'MAE_pp': mae, 'R2': r2})
            temp = test[['date', 'region', 'industry', 'Y1']].copy()
            temp['hazard'] = hazard
            temp['variant'] = name
            temp['pred_Y1'] = predicted
            preds.append(temp)
            print(f'TEST {name:16s} MAE={mae:.3f}pp R²={r2:.3f}')
        print(f'Validation 선택: {winner}')
    if not results:
        print('평가 결과 없음. 재난 플래그와 기간을 확인하세요.')
        return
    pd.DataFrame(results).to_csv(OUT / 'feature_comparison.csv', index=False, encoding='utf-8-sig')
    if preds:
        pd.concat(preds, ignore_index=True).to_csv(OUT / 'test_predictions.csv', index=False, encoding='utf-8-sig')
    print(f'\n완료: {OUT}')
    print('주의: 기존 04/53번과 Y1 및 사건 분할이 다를 수 있어 점수 직접 비교 금지.')
    print('주의: 재난일 관측값을 설명하는 분석이며, 인과적 피해액 또는 사전 예측이 아닙니다.')


if __name__ == '__main__':
    main()
