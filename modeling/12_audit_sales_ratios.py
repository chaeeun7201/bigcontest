"""
12번: 11번 sales_ratios 모델의 과거 매출 변수 및 감소 방향 검증

사용법:
  python modeling/12_audit_sales_ratios.py

필요한 파일 (같은 modeling 폴더):
  04_train_weather_impact_models.py
  11_improve_original_gb_features.py
  modeling_dataset_v2_filtered.csv
  weather_feature_experiment_11/test_predictions.csv

결과: modeling/weather_feature_audit_12/
"""
from pathlib import Path
import importlib.util
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

ROOT = Path(__file__).resolve().parent
EXPERIMENT = ROOT / "weather_feature_experiment_11"
OUT = ROOT / "weather_feature_audit_12"
SOURCE_11 = ROOT / "11_improve_original_gb_features.py"
PRED_PATH = EXPERIMENT / "test_predictions.csv"
KEYS = ["MCT_SGG_CD", "MCT_RY_CD"]
LAGS = ["lag1", "lag7", "rolling7", "rolling28"]
TOL = 1e-6


def load_module():
    if not SOURCE_11.exists():
        raise FileNotFoundError(f"11번 코드가 없습니다: {SOURCE_11}")
    spec = importlib.util.spec_from_file_location("weather_11_audit", SOURCE_11)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_predictions():
    if not PRED_PATH.exists():
        raise FileNotFoundError(
            f"11번 예측 파일이 없습니다: {PRED_PATH}\n"
            "11번을 먼저 실행하세요."
        )
    p = pd.read_csv(PRED_PATH, low_memory=False)
    needed = ["date", *KEYS, "hazard", "variant", "predicted_Y1"]
    missing = [c for c in needed if c not in p.columns]
    if missing:
        raise ValueError(f"11번 예측 CSV 필수 컬럼 누락: {missing}")
    p["date"] = pd.to_datetime(p["date"], errors="raise")
    for k in KEYS:
        p[k] = p[k].astype(str).str.replace(r"\.0$", "", regex=True)
    p["predicted_Y1"] = pd.to_numeric(p["predicted_Y1"], errors="coerce")
    return p


def audit_history_features(df):
    """
    날짜 순으로 정렬한 원본 매출에서 과거 값만 사용해 기대 lag/rolling 계산.
    lag7은 '7일 전'과 '7번째 이전 관측'이 다를 수 있으므로 두 정의를 별도 비교.
    """
    base = df[["date", *KEYS, "TS_AT", *LAGS]].copy()
    base = base.sort_values(KEYS + ["date"]).reset_index(drop=True)
    if base.duplicated(KEYS + ["date"]).any():
        raise ValueError("날짜×지역×업종 중복이 있어 시계열 검증을 할 수 없습니다.")
    grp = base.groupby(KEYS, sort=False)["TS_AT"]
    base["strict_lag1_row"] = grp.shift(1)
    base["strict_lag7_row"] = grp.shift(7)
    base["strict_roll7_row"] = grp.transform(
        lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    base["strict_roll28_row"] = grp.transform(
        lambda s: s.shift(1).rolling(28, min_periods=1).mean())

    calendar = base[KEYS + ["date", "TS_AT"]].copy()
    calendar["date"] = calendar["date"] + pd.Timedelta(days=7)
    calendar = calendar.rename(columns={"TS_AT": "strict_lag7_calendar"})
    base = base.merge(calendar, on=KEYS + ["date"], how="left",
                      validate="one_to_one")

    rows = []
    checks = [
        ("lag1", "strict_lag1_row"),
        ("lag7", "strict_lag7_row"),
        ("lag7", "strict_lag7_calendar"),
        ("rolling7", "strict_roll7_row"),
        ("rolling28", "strict_roll28_row"),
    ]
    for feature, reference in checks:
        a = pd.to_numeric(base[feature], errors="coerce")
        b = pd.to_numeric(base[reference], errors="coerce")
        both = a.notna() & b.notna()
        diff = (a[both] - b[both]).abs()
        rows.append({
            "feature": feature, "reference": reference,
            "both_available": int(both.sum()),
            "matches": int((diff <= TOL * np.maximum(1, b[both].abs())).sum()),
            "match_share": float(
                (diff <= TOL * np.maximum(1, b[both].abs())).mean()
            ) if both.any() else np.nan,
            "max_abs_diff": float(diff.max()) if len(diff) else np.nan,
            "note": "불일치만으로 누수 확정 불가: 원본 04번의 정의/최소 기간 확인 필요"
        })
    return pd.DataFrame(rows), base


def audit_ratio_formulas(df):
    formulas = [
        ("lag7_ratio_28", "lag7", "rolling28"),
        ("rolling7_ratio_28", "rolling7", "rolling28"),
        ("lag1_ratio_7", "lag1", "rolling7"),
    ]
    rows = []
    for name, numerator, denominator in formulas:
        expected = (df[numerator] / df[denominator].where(
            df[denominator] > 0) - 1.0).replace(
            [np.inf, -np.inf], np.nan).clip(-5, 5)
        observed = pd.to_numeric(df[name], errors="coerce")
        both = expected.notna() & observed.notna()
        delta = (expected[both] - observed[both]).abs()
        rows.append({
            "ratio": name, "both_available": int(both.sum()),
            "max_abs_diff": float(delta.max()) if len(delta) else np.nan,
            "all_match": bool((delta < TOL).all()),
            "missing_pattern_match": bool(
                expected.isna().equals(observed.isna()))
        })
    return pd.DataFrame(rows)


def prediction_metrics(p, target):
    results = []
    for (hazard, variant), g in p.groupby(["hazard", "variant"]):
        g = g.dropna(subset=[target, "predicted_Y1"]).copy()
        if g.empty:
            continue
        y = g[target].to_numpy()
        yh = g["predicted_Y1"].to_numpy()
        actual_negative = y < 0
        predicted_negative = yh < 0
        true_positive = int((actual_negative & predicted_negative).sum())
        predicted_positive = int(predicted_negative.sum())
        actual_positive = int(actual_negative.sum())
        results.append({
            "hazard": hazard, "variant": variant, "n": len(g),
            "MAE_pp": mean_absolute_error(y, yh),
            "R2": r2_score(y, yh) if len(g) > 1 else np.nan,
            "direction_accuracy": float(
                (actual_negative == predicted_negative).mean()),
            "actual_decrease_share": float(actual_negative.mean()),
            "predicted_decrease_share": float(predicted_negative.mean()),
            "decrease_precision": (
                true_positive / predicted_positive
                if predicted_positive else np.nan),
            "decrease_recall": (
                true_positive / actual_positive
                if actual_positive else np.nan),
            "always_decrease_accuracy": float(actual_negative.mean()),
            "always_increase_accuracy": float((~actual_negative).mean()),
        })
    return pd.DataFrame(results)


def region_day_metrics(p, target):
    out = []
    for (hazard, variant, region, day), g in p.groupby(
        ["hazard", "variant", KEYS[0], "date"]
    ):
        out.append({
            "hazard": hazard, "variant": variant,
            "region": region, "date": day,
            "n_industries": len(g),
            "MAE_pp": float((g[target] - g["predicted_Y1"]).abs().mean()),
            "direction_accuracy": float(
                ((g[target] < 0) == (g["predicted_Y1"] < 0)).mean()),
            "actual_median_Y1": float(g[target].median()),
            "predicted_median_Y1": float(g["predicted_Y1"].median()),
        })
    return pd.DataFrame(out)


def temporal_split_audit(df):
    """
    04번의 week/episode 교차 여부와 별개로,
    테스트 시점보다 미래의 train 날짜가 존재하는지 확인.
    이 경우 '미래 시점 예측' 성능으로 해석하면 안 됨.
    """
    unique = df[["date", "split"]].drop_duplicates()
    train_dates = unique.loc[unique["split"].eq("train"), "date"]
    rows = []
    for split in ["validation", "test"]:
        other = unique.loc[unique["split"].eq(split), "date"]
        rows.append({
            "evaluated_split": split,
            "train_min_date": train_dates.min(),
            "train_max_date": train_dates.max(),
            "evaluation_min_date": other.min(),
            "evaluation_max_date": other.max(),
            "train_dates_after_first_evaluation_date": int(
                (train_dates > other.min()).sum()) if len(other) else 0,
            "chronological_train_before_evaluation": bool(
                train_dates.max() < other.min()) if len(other) and len(train_dates)
                else False,
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 75)
    print("12번: 11번 매출 추세 변수 누수 진단 및 감소 방향 정확도 검증")
    print("=" * 75)
    module = load_module()
    df, dates, episodes = module.prepare(module.load_original())
    p = load_predictions()

    target_map = {"heatwave": "Y1_heatwave", "cold_wave": "Y1_cold_wave"}
    frames = []
    for hazard, target in target_map.items():
        sub = p[p["hazard"].eq(hazard)].copy()
        if sub.empty:
            continue
        # 11번 예측 파일에 타깃이 있더라도 재계산된 04번 Y1과 대조한다.
        lookup = df[["date", *KEYS, target, "split"]].copy()
        sub = sub.drop(columns=[target, "split"], errors="ignore").merge(
            lookup, on=["date", *KEYS], how="left", validate="many_to_one")
        if sub[target].isna().any():
            raise ValueError(f"{hazard}: 예측 결과 중 Y1 매칭 실패가 있습니다.")
        if not sub["split"].eq("test").all():
            raise ValueError(f"{hazard}: 예측 CSV에 test 이외의 행이 있습니다.")
        frames.append(sub)
    if not frames:
        raise ValueError("검증할 예측 데이터가 없습니다.")
    pred = pd.concat(frames, ignore_index=True)

    OUT.mkdir(parents=True, exist_ok=True)
    history_summary, history_detail = audit_history_features(df)
    ratio_summary = audit_ratio_formulas(df)
    temporal = temporal_split_audit(df)
    metrics = []
    region_days = []
    for hazard, target in target_map.items():
        part = pred[pred["hazard"].eq(hazard)]
        if not part.empty:
            metrics.append(prediction_metrics(part, target))
            region_days.append(region_day_metrics(part, target))
    metrics = pd.concat(metrics, ignore_index=True)
    region_days = pd.concat(region_days, ignore_index=True)

    history_summary.to_csv(OUT / "lag_history_audit.csv",
                           index=False, encoding="utf-8-sig")
    ratio_summary.to_csv(OUT / "ratio_formula_audit.csv",
                         index=False, encoding="utf-8-sig")
    temporal.to_csv(OUT / "temporal_split_audit.csv",
                    index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT / "decrease_direction_metrics.csv",
                   index=False, encoding="utf-8-sig")
    region_days.to_csv(OUT / "region_day_direction_metrics.csv",
                       index=False, encoding="utf-8-sig")
    pred.to_csv(OUT / "verified_test_predictions.csv",
                index=False, encoding="utf-8-sig")

    print("\n[과거 매출 변수 검증]")
    print(history_summary[["feature", "reference", "both_available",
                           "match_share"]].to_string(index=False))
    print("\n[비율 변수 수식 검증]")
    print(ratio_summary.to_string(index=False))
    print("\n[시간순 분할 검증]")
    print(temporal.to_string(index=False))
    print("\n[감소 방향 성능]")
    print(metrics[["hazard", "variant", "n", "MAE_pp", "R2",
                   "direction_accuracy", "always_decrease_accuracy",
                   "always_increase_accuracy", "decrease_precision",
                   "decrease_recall"]].to_string(index=False))
    print("\n[지역·일별 방향 성능]")
    print(region_days.to_string(index=False))
    print(f"\n완료: {OUT}")
    print("주의: lag 정의 불일치는 즉시 누수 확정이 아닙니다.")
    print("주의: 04번 목표변수 Y1의 Train-only 기준매출 산정 자체는 별도 감사 대상입니다.")
    print("주의: 같은 날 실측 기온 사용 및 독립 사건 수 부족으로 사전 예측 성능을 의미하지 않습니다.")


if __name__ == "__main__":
    main()
