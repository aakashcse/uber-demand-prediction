"""Step 2 - Feature engineering.

Turns the demand time series into a table the model can learn from.
Every row = one zone at one 15-minute interval; the target is `pickups`.
All features only use information available *before* that interval
(no data leakage), so the same code works for training and for live prediction.
"""
import pandas as pd

from src import config

# Recent demand in the same zone
LAG_FEATURES = ["lag_1", "lag_2", "lag_3", "lag_4"]        # previous 4 intervals (last hour)
SEASONAL_FEATURES = ["lag_1d", "lag_1w"]                   # same time yesterday / last week
TREND_FEATURES = ["ewma", "rolling_mean_1h"]
TIME_FEATURES = ["hour", "quarter", "day_of_week", "is_weekend"]
ZONE_FEATURE = ["zone_id"]

FEATURES = ZONE_FEATURE + TIME_FEATURES + LAG_FEATURES + SEASONAL_FEATURES + TREND_FEATURES
TARGET = "pickups"

FEATURE_DESCRIPTIONS = {
    "zone_id": "Taxi zone (categorical) - every zone has its own demand level",
    "hour": "Hour of the day (0-23)",
    "quarter": "Quarter of the hour (0 = :00, 1 = :15, 2 = :30, 3 = :45)",
    "day_of_week": "Day of week (0 = Monday ... 6 = Sunday)",
    "is_weekend": "1 on Saturday/Sunday",
    "lag_1": "Pickups in the previous 15 minutes",
    "lag_2": "Pickups 30 minutes ago",
    "lag_3": "Pickups 45 minutes ago",
    "lag_4": "Pickups 60 minutes ago",
    "lag_1d": "Pickups at the same time yesterday",
    "lag_1w": "Pickups at the same time last week",
    "ewma": "Exponentially weighted average of past demand (recent intervals count more)",
    "rolling_mean_1h": "Average pickups over the last hour",
}

EWMA_ALPHA = 0.4


def add_time_features(df):
    ts = df["timestamp"]
    df["hour"] = ts.dt.hour.astype("int8")
    df["quarter"] = (ts.dt.minute // 15).astype("int8")
    df["day_of_week"] = ts.dt.dayofweek.astype("int8")
    df["is_weekend"] = (df["day_of_week"] >= 5).astype("int8")
    return df


def build_features(demand):
    """demand: columns zone_id, timestamp, pickups (complete 15-min grid per zone)."""
    df = demand.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)
    by_zone = df.groupby("zone_id")["pickups"]

    for k in range(1, 5):
        df[f"lag_{k}"] = by_zone.shift(k)
    df["lag_1d"] = by_zone.shift(config.INTERVALS_PER_DAY)
    df["lag_1w"] = by_zone.shift(config.INTERVALS_PER_WEEK)
    df["rolling_mean_1h"] = df[LAG_FEATURES].mean(axis=1)
    # EWMA of *past* values only (shift(1) so the current interval is never used)
    df["ewma"] = by_zone.transform(lambda s: s.shift(1).ewm(alpha=EWMA_ALPHA, adjust=False).mean())

    df = add_time_features(df)
    # The first week of each zone has no "last week" value -> drop those rows
    return df.dropna(subset=LAG_FEATURES + SEASONAL_FEATURES).reset_index(drop=True)


def manual_feature_row(zone_id, timestamp, last_4, same_time_yesterday, same_time_last_week):
    """Feature row built from values typed in by the user (what-if prediction).
    last_4 = [15 min ago, 30 min ago, 45 min ago, 60 min ago]."""
    row = pd.DataFrame({"zone_id": [zone_id], "timestamp": [pd.Timestamp(timestamp)]})
    row = add_time_features(row)
    for k, value in enumerate(last_4, start=1):
        row[f"lag_{k}"] = float(value)
    row["lag_1d"] = float(same_time_yesterday)
    row["lag_1w"] = float(same_time_last_week)
    row["rolling_mean_1h"] = sum(last_4) / 4
    # EWMA approximated from the last hour (oldest -> newest)
    ewma = float(last_4[3])
    for value in reversed(last_4[:3]):
        ewma = EWMA_ALPHA * value + (1 - EWMA_ALPHA) * ewma
    row["ewma"] = ewma
    return row[FEATURES]
