"""
11번: 04번 원본 Y1·Week/Episode 분할을 그대로 재사용하는 GB Feature Engineering 실험.

실행: python modeling/11_improve_original_gb_features.py
필요: 같은 modeling 폴더의 04_train_weather_impact_models.py,
      modeling_dataset_v2_filtered.csv
결과: modeling/weather_feature_experiment_11/

주의: 같은 날 실측 기온을 사용하므로 사후 매출 반응 설명용입니다.
      lag 매출은 t-1 이전 값만 사용합니다.
"""
from pathlib import Path
import importlib.util
import json
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
DATA = HERE / "modeling_dataset_v2_filtered.csv"
ORIGINAL = HERE / "04_train_weather_impact_models.py"
OUT = HERE / "weather_feature_experiment_11"
SEED = 20261003
KEY = ["MCT_SGG_CD", "MCT_RY_CD"]
HAZARDS = {"heatwave": (7, 8, 9), "cold_wave": (11, 12)}

def load_original():
    if not ORIGINAL.exists():
        raise FileNotFoundError(f"04번 원본 파일이 필요합니다: {ORIGINAL}")
    spec = importlib.util.spec_from_file_location("original_weather_04", ORIGINAL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("build_components", "assign_splits", "audit_split",
                 "add_targets", "add_calendar_features"):
        if not hasattr(mod, name):
            raise RuntimeError(f"04번 파일에서 {name} 함수를 찾지 못했습니다.")
    return mod

def prepare(mod):
    df = pd.read_csv(DATA, encoding="utf-8-sig", low_memory=False)
    required = ["TA_YMD", *KEY, "TS_AT", "heatwave", "cold_wave",
                "avg_temp", "min_temp", "max_temp"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"필수 열 누락: {missing}")
    if df.duplicated(["TA_YMD", *KEY]).any():
        raise ValueError("날짜×지역×업종 중복행이 있습니다.")
    df["date"] = pd.to_datetime(df["TA_YMD"].astype(str).str.replace(
        r"\.0$", "", regex=True), format="%Y%m%d", errors="raise")
    for c in KEY:
        df[c] = df[c].astype(str)
    for c in ["TS_AT", "avg_temp", "min_temp", "max_temp", "heatwave", "cold_wave"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[KEY + ["date", "TS_AT"]].isna().any().any():
        raise ValueError("날짜/지역/업종/매출에 결측이 있습니다.")
    print(f"전체 데이터: {len(df):,}행")
    dates, episodes = mod.build_components(df)
    dates = mod.assign_splits(dates, df, SEED)
    mod.audit_split(df, dates, episodes)
    df = df.merge(dates[["date", "week", "component", "split"]],
                  on="date", validate="many_to_one")
    df, baselines = mod.add_targets(df)
    df, lag_features = mod.add_calendar_features(df)

    # 당일 매출은 직접/간접적으로 피처에 사용하지 않음.
    # 과거 매출 lag/rolling의 비율만 파생 (결측/0분모는 NA 유지).
    for numerator, denominator, name in [
        ("lag7", "rolling28", "lag7_ratio_28"),
        ("rolling7", "rolling28", "rolling7_ratio_28"),
        ("lag1", "rolling7", "lag1_ratio_7"),
    ]:
        denom = df[denominator].where(df[denominator] > 0)
        df[name] = (df[numerator] / denom - 1.0).replace(
            [np.inf, -np.inf], np.nan).clip(-5, 5)
    df["temp_range"] = df["max_temp"] - df["min_temp"]
    df["temp_mean_min_gap"] = df["avg_temp"] - df["min_temp"]

    # 평년 편차가 아니라, 학습기간 지역별 기온 중앙값 대비 편차.
    # 해당 지역·날짜의 기온은 업종별 반복되므로 지역·날짜 단위로 계산.
    weather = df[["date", KEY[0], "avg_temp", "split"]].drop_duplicates(
        ["date", KEY[0]])
    reference = weather[weather["split"].eq("train")].groupby(KEY[0])[
        "avg_temp"].median()
    df["temp_vs_train_median"] = df["avg_temp"] - df[KEY[0]].map(reference)
    return df, dates, episodes

def make_model(numeric):
    prep = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), KEY),
    ])
    gb = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.04, max_depth=2,
        loss="huber", random_state=SEED)
    return Pipeline([("prep", prep), ("model", gb)])

def metrics(y, pred):
    return {
        "MAE_pp": mean_absolute_error(y, pred),
        "RMSE_pp": mean_squared_error(y, pred) ** 0.5,
        "R2": r2_score(y, pred) if len(y) > 1 else np.nan,
    }

def main():
    print("11번: 04번 Y1·Week/Episode 분할 재사용 / GB 입력변수 개선")
    mod = load_original()
    df, dates, episodes = prepare(mod)
    OUT.mkdir(parents=True, exist_ok=True)
    dates.to_csv(OUT / "split_by_date.csv", index=False, encoding="utf-8-sig")
    episodes.to_csv(OUT / "episodes.csv", index=False, encoding="utf-8-sig")
    base = ["avg_temp", "min_temp", "max_temp", "month", "weekday",
            "lag1", "lag7", "rolling7", "rolling28"]
    variants = {
        "original_04": base,
        "sales_ratios": base + ["lag7_ratio_28", "rolling7_ratio_28",
                                "lag1_ratio_7"],
        "temp_features": base + ["temp_range", "temp_mean_min_gap",
                                 "temp_vs_train_median"],
        "all_added": base + ["lag7_ratio_28", "rolling7_ratio_28",
                             "lag1_ratio_7", "temp_range", "temp_mean_min_gap",
                             "temp_vs_train_median"],
    }
    rows, predictions = [], []
    for hazard in HAZARDS:
        target = f"Y1_{hazard}"
        work = df[df[hazard].eq(1) & df[target].notna()].copy()
        parts = {s: work[work["split"].eq(s)] for s in
                 ("train", "validation", "test")}
        if any(p.empty for p in parts.values()):
            raise ValueError(f"{hazard}: 학습/검증/평가 중 빈 분할이 있습니다.")
        print(f"\n[{hazard}] " + ", ".join(
            f"{s}={len(p):,}" for s, p in parts.items()))
        trained = {}
        for variant, numeric in variants.items():
            model = make_model(numeric)
            model.fit(parts["train"][numeric + KEY], parts["train"][target])
            trained[variant] = model
            for split in ("validation", "test"):
                part = parts[split]
                pred = model.predict(part[numeric + KEY])
                score = metrics(part[target], pred)
                rows.append({"hazard": hazard, "variant": variant,
                             "split": split, "n": len(part), **score})
                if split == "test":
                    prediction = part[["date", *KEY, "split", target]].copy()
                    prediction["hazard"] = hazard
                    prediction["variant"] = variant
                    prediction["predicted_Y1"] = pred
                    predictions.append(prediction)
            val = [r for r in rows if r["hazard"] == hazard and
                   r["variant"] == variant and r["split"] == "validation"][0]
            print(f"  {variant:15s} Val MAE={val['MAE_pp']:.3f}pp "
                  f"R²={val['R2']:.3f}")
        val_rows = [r for r in rows if r["hazard"] == hazard and
                    r["split"] == "validation"]
        selected = min(val_rows, key=lambda x: x["MAE_pp"])["variant"]
        for variant in ("original_04", selected):
            test = next(r for r in rows if r["hazard"] == hazard and
                        r["variant"] == variant and r["split"] == "test")
            print(f"  TEST {variant:15s} MAE={test['MAE_pp']:.3f}pp "
                  f"R²={test['R2']:.3f}")
        print(f"  Validation MAE 기준 선택: {selected}")
    pd.DataFrame(rows).to_csv(OUT / "feature_comparison.csv",
                              index=False, encoding="utf-8-sig")
    pd.concat(predictions, ignore_index=True).to_csv(
        OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")
    (OUT / "experiment_config.json").write_text(json.dumps({
        "original_file": ORIGINAL.name, "seed": SEED,
        "target": "04번 add_targets의 train-only 평시 평균 기반 Y1",
        "split": "04번 build_components + assign_splits",
        "variants": variants,
        "selection": "Validation MAE only; Test not used for selection",
        "caution": "Same-day observed temperatures; retrospective explanation, not forecast"
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n완료: {OUT}")
    print("주의: 원본 04번 파일의 함수가 변경되면 재현 결과도 바뀔 수 있습니다.")
    print("주의: 재난 지역·일 표본이 적으므로 Test 수치는 불안정합니다.")

if __name__ == "__main__":
    main()
