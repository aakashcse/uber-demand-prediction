"""Step 3 - Model training and evaluation.

* Time-based split: the last month is the test set, everything before it trains
  the model (random splits would leak future information into training).
* Compares two simple baselines, Linear Regression and a Gradient Boosting model.
* Saves the best model plus metrics, feature importance and test predictions
  that the app displays.

Run:  python -m src.train
"""
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from src import config
from src.features import FEATURES, TARGET, build_features
from src.model import predict, prepare_X


def regression_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
        # Weighted Absolute Percentage Error: total error / total demand.
        # Used instead of plain MAPE, which explodes when a zone has 0-1 pickups.
        "WAPE": round(float(np.abs(y_true - y_pred).sum() / y_true.sum()), 4),
    }


def time_split(df):
    test_month = df["timestamp"].dt.to_period("M").max()
    is_test = df["timestamp"].dt.to_period("M") == test_month
    return df[~is_test], df[is_test]


def train_linear(train):
    # Teacher's approach: one-hot encode categories, linear model on top
    encoder = ColumnTransformer(
        [("onehot", OneHotEncoder(handle_unknown="ignore"), ["zone_id", "hour", "day_of_week"])],
        remainder="passthrough")
    model = make_pipeline(encoder, LinearRegression())
    model.fit(train[FEATURES], train[TARGET])
    return model


def train_gradient_boosting(train, zone_categories):
    model = HistGradientBoostingRegressor(
        loss="poisson",              # pickups are counts (>= 0), Poisson loss suits count data
        learning_rate=0.08,
        max_iter=1000,               # upper limit - early stopping picks the actual number
        max_leaf_nodes=63,
        min_samples_leaf=40,
        l2_regularization=1.0,
        categorical_features="from_dtype",
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=30,
        random_state=config.RANDOM_STATE,
    )
    model.fit(prepare_X(train, zone_categories), train[TARGET])
    return model


def main():
    demand = pd.read_parquet(config.DEMAND_FILE)
    data = build_features(demand)
    train, test = time_split(data)
    zone_categories = sorted(data["zone_id"].unique().tolist())
    print(f"Train: {len(train):,} rows ({train['timestamp'].min():%d %b %Y} - {train['timestamp'].max():%d %b %Y})")
    print(f"Test : {len(test):,} rows ({test['timestamp'].min():%d %b %Y} - {test['timestamp'].max():%d %b %Y})")

    y_test = test[TARGET]
    results = {
        "Baseline: last 15 min": regression_metrics(y_test, test["lag_1"]),
        "Baseline: same time last week": regression_metrics(y_test, test["lag_1w"]),
    }

    print("Training Linear Regression ...")
    linear = train_linear(train)
    results["Linear Regression"] = regression_metrics(y_test, np.clip(linear.predict(test[FEATURES]), 0, None))

    print("Training Gradient Boosting ...")
    gbr = train_gradient_boosting(train, zone_categories)
    bundle = {"model": gbr, "zone_categories": zone_categories, "features": FEATURES}
    test_pred = predict(bundle, test)
    results["Gradient Boosting (HistGBR)"] = regression_metrics(y_test, test_pred)

    for name, m in results.items():
        print(f"  {name:32s} MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.3f}  WAPE={m['WAPE']:.1%}")

    # Feature importance: how much the test error grows when a feature is shuffled
    sample = test.sample(min(40_000, len(test)), random_state=config.RANDOM_STATE)
    imp = permutation_importance(gbr, prepare_X(sample, zone_categories), sample[TARGET],
                                 scoring="neg_mean_absolute_error", n_repeats=3,
                                 random_state=config.RANDOM_STATE)
    importance = (pd.DataFrame({"feature": FEATURES, "importance": imp.importances_mean})
                  .sort_values("importance", ascending=False))

    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    bundle["trained_at"] = datetime.now().isoformat(timespec="seconds")
    joblib.dump(bundle, config.MODEL_FILE, compress=3)
    importance.to_csv(config.IMPORTANCE_FILE, index=False)
    (test[["zone_id", "timestamp", TARGET]]
     .assign(predicted=test_pred.round(2).astype("float32"))
     .to_parquet(config.PREDICTIONS_FILE, index=False))

    metrics = {
        "chosen_model": "Gradient Boosting (HistGBR)",
        "results": results,
        "train_period": [str(train["timestamp"].min()), str(train["timestamp"].max())],
        "test_period": [str(test["timestamp"].min()), str(test["timestamp"].max())],
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "n_iterations": int(gbr.n_iter_),
        "params": {k: v for k, v in gbr.get_params().items()
                   if k in ("loss", "learning_rate", "max_iter", "max_leaf_nodes", "min_samples_leaf", "l2_regularization")},
    }
    config.METRICS_FILE.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved model -> {config.MODEL_FILE.relative_to(config.ROOT)}")


if __name__ == "__main__":
    main()
