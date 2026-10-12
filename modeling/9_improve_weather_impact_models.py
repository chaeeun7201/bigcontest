"""53: 기존 04번의 Y1/분할을 그대로 사용한 Gradient Boosting 개선 비교.

같은 입력, 같은 seed, 같은 Train-only baseline, 같은 week/episode split.
Validation에서만 모델 선택하고 Test는 최종 1회 평가한다.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE / "modeling_dataset_v2_filtered.csv"
DEFAULT_OUT = HERE / "weather_impact_improvement_53"


def load_original(path: Path):
    spec = importlib.util.spec_from_file_location("weather_original_04", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"04번 코드를 불러올 수 없습니다: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def score(y, p):
    return {
        "MAE_pp": float(mean_absolute_error(y, p)),
        "RMSE_pp": float(np.sqrt(mean_squared_error(y, p))),
        "R2": float(r2_score(y, p)) if len(y) > 1 else np.nan,
    }


def pipeline(features, numeric, categorical, params):
    prep = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ])
    return Pipeline([("prep", prep), ("model", GradientBoostingRegressor(**params))])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--original", type=Path, default=HERE / "04_train_weather_impact_models.py")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=20261003)
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"입력 데이터가 없습니다: {args.input}")
    if not args.original.is_file():
        raise FileNotFoundError(f"04번 원본 파일이 없습니다: {args.original}")
    old = load_original(args.original)
    print("53번: 기존 04번 Y1 및 분할 재사용 / Validation 기준 모델 개선", flush=True)
    df = pd.read_csv(args.input, encoding="utf-8-sig", low_memory=False)
    needed = ["TA_YMD", "MCT_SGG_CD", "MCT_RY_CD", "TS_AT", "heatwave", "cold_wave", "avg_temp", "min_temp", "max_temp"]
    if missing := sorted(set(needed) - set(df.columns)):
        raise ValueError(f"필수 열 누락: {missing}")
    df["date"] = pd.to_datetime(df["TA_YMD"].astype(str), format="%Y%m%d", errors="raise")
    df["MCT_SGG_CD"] = df["MCT_SGG_CD"].astype(str)
    df["MCT_RY_CD"] = df["MCT_RY_CD"].astype(str)
    if df.duplicated(["date", *old.KEY]).any():
        raise ValueError("지역-업종-날짜 중복이 있습니다.")
    dates, episodes = old.build_components(df)
    dates = old.assign_splits(dates, df, args.seed)
    old.audit_split(df, dates, episodes)
    df = df.merge(dates[["date", "week", "component", "split"]], on="date", validate="many_to_one")
    df, _ = old.add_targets(df)
    df, lags = old.add_calendar_features(df)

    # 과거 매출은 t-1까지의 관측값만 사용. 재난일의 실제매출은 feature로 사용하지 않음.
    base_numeric = ["avg_temp", "min_temp", "max_temp", "month", "weekday", *lags]
    enhanced_numeric = base_numeric + ["day_of_year", "month_sin", "month_cos", "temp_range", "log_lag7", "log_rolling28"]
    df["day_of_year"] = df["date"].dt.dayofyear
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["temp_range"] = df["max_temp"] - df["min_temp"]
    df["log_lag7"] = np.log1p(df["lag7"].clip(lower=0))
    df["log_rolling28"] = np.log1p(df["rolling28"].clip(lower=0))
    cats = list(old.KEY)

    # 원본 04번의 Gradient Boosting 파라미터를 첫 번째 후보로 보존.
    candidates = {
        "original_GB": ({"n_estimators": 200, "learning_rate": .04, "max_depth": 2, "loss": "huber", "random_state": args.seed}, base_numeric),
        "GB_more_trees": ({"n_estimators": 450, "learning_rate": .025, "max_depth": 2, "min_samples_leaf": 10, "loss": "huber", "random_state": args.seed}, enhanced_numeric),
        "GB_regularized": ({"n_estimators": 300, "learning_rate": .03, "max_depth": 2, "min_samples_leaf": 20, "subsample": .8, "loss": "huber", "random_state": args.seed}, enhanced_numeric),
        "GB_depth3": ({"n_estimators": 250, "learning_rate": .03, "max_depth": 3, "min_samples_leaf": 15, "subsample": .8, "loss": "huber", "random_state": args.seed}, enhanced_numeric),
        "GB_simple": ({"n_estimators": 150, "learning_rate": .04, "max_depth": 1, "min_samples_leaf": 15, "loss": "huber", "random_state": args.seed}, base_numeric),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    all_rows, all_predictions, chosen = [], [], []
    for hazard in old.HAZARDS:
        target = f"Y1_{hazard}"
        event = df.loc[df[hazard].eq(1) & df[target].notna()].copy()
        train = event[event["split"].eq("train")]
        val = event[event["split"].eq("validation")]
        test = event[event["split"].eq("test")]
        if min(len(train), len(val), len(test)) == 0:
            print(f"[{hazard}] Train/Validation/Test 표본 부족: 건너뜀")
            continue
        print(f"\n[{hazard}] train={len(train):,}, validation={len(val):,}, test={len(test):,}", flush=True)
        val_models = []
        for name, (params, nums) in candidates.items():
            features = nums + cats
            pipe = pipeline(features, nums, cats, params)
            pipe.fit(train[features], train[target])
            p = pipe.predict(val[features])
            result = {"hazard": hazard, "model": name, "split": "validation", "n": len(val), **score(val[target], p)}
            all_rows.append(result)
            val_models.append((result["MAE_pp"], name))
            print(f"  {name}: Val MAE={result['MAE_pp']:.3f}pp, R²={result['R2']:.3f}")
        # 테스트 데이터를 보고 선택하지 않음. 동일 MAE이면 후보 이름순으로 고정.
        best_name = min(val_models)[1]
        print(f"  선택: {best_name} (Validation MAE 최소)")
        chosen.append({"hazard": hazard, "selected_model": best_name, "selection_metric": "validation_MAE_pp"})
        # 공정한 테스트 비교: 원본 모델과 선택 모델만, 동일한 Test 표본.
        for name in dict.fromkeys(["original_GB", best_name]):
            params, nums = candidates[name]
            features = nums + cats
            pipe = pipeline(features, nums, cats, params)
            pipe.fit(train[features], train[target])
            p = pipe.predict(test[features])
            result = {"hazard": hazard, "model": name, "split": "test", "n": len(test), **score(test[target], p)}
            all_rows.append(result)
            print(f"  TEST {name}: MAE={result['MAE_pp']:.3f}pp, R²={result['R2']:.3f}")
            rows = test[["date", *cats, "split", target]].copy()
            rows["hazard"] = hazard
            rows["model"] = name
            rows["prediction_Y1"] = p
            rows["error_Y1"] = rows[target] - p
            all_predictions.append(rows.rename(columns={target: "actual_Y1"}))
    if not all_rows:
        print("평가할 수 있는 재난 유형이 없습니다.")
        return
    pd.DataFrame(all_rows).to_csv(args.out / "metrics.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(chosen).to_csv(args.out / "selected_models.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_predictions, ignore_index=True).to_csv(args.out / "test_predictions.csv", index=False, encoding="utf-8-sig")
    episodes.to_csv(args.out / "episodes.csv", index=False, encoding="utf-8-sig")
    dates.to_csv(args.out / "split_by_date.csv", index=False, encoding="utf-8-sig")
    (args.out / "config.json").write_text(json.dumps({"input": str(args.input), "original": str(args.original), "seed": args.seed, "selection": "validation MAE; Test evaluated once", "candidate_models": list(candidates)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n완료: {args.out}")
    print("주의: 재난일 Y1 예측력은 재난의 인과적 추가 효과에 대한 증거가 아닙니다.")
    print("주의: 기존 무작위 week/episode 분할은 미래 사건 시간순 예측 검증이 아닙니다.")


if __name__ == "__main__":
    main()
