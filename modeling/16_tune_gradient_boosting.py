"""16번: 기존 04번 Y1·Week/Episode 분할을 유지한 Gradient Boosting 튜닝.

실행: python modeling/16_tune_gradient_boosting.py
필요 파일 (같은 modeling 폴더):
  04_train_weather_impact_models.py
  11_improve_original_gb_features.py
  modeling_dataset_v2_filtered.csv

모델 선택은 Validation MAE로만 수행하고, Test는 선택 완료 후 평가한다.
관측된 당일 기온을 사용하므로 재난 발생 후 매출 반응 설명용이다.
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
SOURCE_11 = HERE / "11_improve_original_gb_features.py"
OUT = HERE / "weather_gb_tuning_16"
SEED = 20261003
KEY = ["MCT_SGG_CD", "MCT_RY_CD"]
HAZARDS = ("heatwave", "cold_wave")

BASE = ["avg_temp", "min_temp", "max_temp", "month", "weekday",
        "lag1", "lag7", "rolling7", "rolling28"]
ALL_ADDED = BASE + ["lag7_ratio_28", "rolling7_ratio_28", "lag1_ratio_7",
                    "temp_range", "temp_mean_min_gap", "temp_vs_train_median"]

# 후보 수를 제한하여 검증 데이터에 대한 과도한 탐색을 줄인다.
CONFIGS = {
    "original_04": dict(n_estimators=200, learning_rate=0.04, max_depth=2,
                        min_samples_leaf=1, subsample=1.0, loss="huber"),
    "more_trees": dict(n_estimators=350, learning_rate=0.025, max_depth=2,
                       min_samples_leaf=3, subsample=1.0, loss="huber"),
    "leaf_5": dict(n_estimators=200, learning_rate=0.04, max_depth=2,
                   min_samples_leaf=5, subsample=1.0, loss="huber"),
    "leaf_10": dict(n_estimators=200, learning_rate=0.04, max_depth=2,
                    min_samples_leaf=10, subsample=1.0, loss="huber"),
    "shallow": dict(n_estimators=300, learning_rate=0.03, max_depth=1,
                    min_samples_leaf=5, subsample=1.0, loss="huber"),
    "depth_3": dict(n_estimators=200, learning_rate=0.03, max_depth=3,
                    min_samples_leaf=5, subsample=1.0, loss="huber"),
    "subsample": dict(n_estimators=300, learning_rate=0.03, max_depth=2,
                      min_samples_leaf=5, subsample=0.8, loss="huber"),
    "squared_error": dict(n_estimators=200, learning_rate=0.04, max_depth=2,
                          min_samples_leaf=5, subsample=1.0, loss="squared_error"),
}


def load_11():
    if not SOURCE_11.exists():
        raise FileNotFoundError(f"11번 파일을 찾지 못했습니다: {SOURCE_11}")
    spec = importlib.util.spec_from_file_location("model11", SOURCE_11)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_pipeline(numeric, params):
    prep = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), KEY),
    ])
    reg = GradientBoostingRegressor(**params, random_state=SEED)
    return Pipeline([("prep", prep), ("model", reg)])


def scores(y, pred):
    y = np.asarray(y, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return {
        "MAE_pp": float(mean_absolute_error(y, pred)),
        "RMSE_pp": float(mean_squared_error(y, pred) ** 0.5),
        "R2": float(r2_score(y, pred)) if len(y) > 1 else np.nan,
        "direction_accuracy": float(np.mean((y < 0) == (pred < 0))),
    }


def main():
    print("16번: Gradient Boosting 하이퍼파라미터 개선")
    print("04번 Y1·Week/Episode 분할 재사용, Validation으로만 모델 선택")
    module11 = load_11()
    original = module11.load_original()
    df, dates, episodes = module11.prepare(original)
    OUT.mkdir(parents=True, exist_ok=True)
    dates.to_csv(OUT / "split_by_date.csv", index=False, encoding="utf-8-sig")
    episodes.to_csv(OUT / "episodes.csv", index=False, encoding="utf-8-sig")

    rows, test_predictions, selections = [], [], []
    for hazard in HAZARDS:
        target = f"Y1_{hazard}"
        work = df[df[hazard].eq(1) & df[target].notna()].copy()
        parts = {s: work.loc[work["split"].eq(s)].copy()
                 for s in ("train", "validation", "test")}
        if any(p.empty for p in parts.values()):
            raise ValueError(f"{hazard}: train/validation/test 중 빈 분할이 있습니다.")
        print(f"\n[{hazard}] " + " / ".join(f"{s}={len(p):,}행" for s, p in parts.items()))
        print("Validation 후보 비교 (MAE 단위: 퍼센트포인트)")

        # 04번 원본 피처+파라미터와 11번 all_added 피처+동일 파라미터를 비교 기준으로 둔다.
        candidates = [("original_04", BASE, CONFIGS["original_04"])]
        candidates += [(f"all_added__{name}", ALL_ADDED, params)
                       for name, params in CONFIGS.items()]
        fitted = {}
        for name, numeric, params in candidates:
            model = make_pipeline(numeric, params)
            model.fit(parts["train"][numeric + KEY], parts["train"][target])
            pred = model.predict(parts["validation"][numeric + KEY])
            result = scores(parts["validation"][target], pred)
            rows.append({"hazard": hazard, "candidate": name,
                         "split": "validation", "n": len(parts["validation"]), **result})
            fitted[name] = (model, numeric)
            print(f"  {name:28s} MAE={result['MAE_pp']:7.3f}pp "
                  f"R²={result['R2']:7.3f} 방향정확도={result['direction_accuracy']:.3f}")

        hazard_val = [r for r in rows if r["hazard"] == hazard and r["split"] == "validation"]
        winner = min(hazard_val, key=lambda r: r["MAE_pp"])["candidate"]
        print(f"Validation 선택: {winner}")
        selections.append({"hazard": hazard, "selected_candidate": winner,
                           "selection_metric": "validation_MAE_pp"})

        # Test는 모델 선택이 끝난 뒤 원본과 최종 선택 모델만 확인한다.
        for name in dict.fromkeys(["original_04", winner]):
            model, numeric = fitted[name]
            pred = model.predict(parts["test"][numeric + KEY])
            result = scores(parts["test"][target], pred)
            rows.append({"hazard": hazard, "candidate": name,
                         "split": "test", "n": len(parts["test"]), **result})
            print(f"TEST {name:28s} MAE={result['MAE_pp']:7.3f}pp "
                  f"R²={result['R2']:7.3f} 방향정확도={result['direction_accuracy']:.3f}")
            frame = parts["test"][["date", *KEY, target]].copy()
            frame["hazard"] = hazard
            frame["candidate"] = name
            frame["predicted_Y1"] = pred
            test_predictions.append(frame)

    pd.DataFrame(rows).to_csv(OUT / "model_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(selections).to_csv(OUT / "selected_models.csv", index=False, encoding="utf-8-sig")
    pd.concat(test_predictions, ignore_index=True).to_csv(
        OUT / "test_predictions.csv", index=False, encoding="utf-8-sig")
    (OUT / "experiment_config.json").write_text(json.dumps({
        "seed": SEED,
        "target": "04번 Train-only 평시 기준 Y1",
        "split": "04번 Week+Episode (11번 prepare 함수 재사용)",
        "feature_sets": {"original_04": BASE, "all_added": ALL_ADDED},
        "configs": CONFIGS,
        "selection": "Validation MAE only; Test not used for selection",
        "limitation": "Only two independent test region-days per hazard in prior experiments; retrospective observed-weather response, not causal effect or future forecasting"
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n완료: {OUT}")
    print("주의: 재난별 Test 지역·날짜가 매우 적어 결과의 불확실성이 큽니다.")
    print("주의: 이 모델은 재난의 인과적 매출 피해액을 산출하지 않습니다.")


if __name__ == "__main__":
    main()
