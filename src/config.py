"""Central place for paths and settings used by the pipeline and the app."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

RAW_DIR = ROOT / "data" / "raw"              # downloaded NYC TLC files (not committed)
PROCESSED_DIR = ROOT / "data" / "processed"  # small files the app needs (committed)
MODELS_DIR = ROOT / "models"

# Raw inputs
TRIP_FILES_GLOB = "yellow_tripdata_*.parquet"
ZONE_LOOKUP_FILE = RAW_DIR / "taxi_zone_lookup.csv"


def find_zone_boundaries():
    """Zone boundary file from the TLC site: new GeoParquet or the older zipped shapefile."""
    for pattern in ("taxi_zone*.parquet", "taxi_zone*.zip"):
        found = sorted(RAW_DIR.glob(pattern))
        if found:
            return found[0]
    return None

# Processed outputs
DEMAND_FILE = PROCESSED_DIR / "demand_15min.parquet"   # pickups per zone per 15 minutes
ZONES_FILE = PROCESSED_DIR / "zones.csv"               # zone id, name, borough, centroid
ZONES_GEOJSON = PROCESSED_DIR / "zones.geojson"        # simplified zone boundaries for the map
SUMMARY_FILE = PROCESSED_DIR / "data_summary.json"     # cleaning stats shown in the app

# Model outputs
MODEL_FILE = MODELS_DIR / "model.joblib"
METRICS_FILE = MODELS_DIR / "metrics.json"
IMPORTANCE_FILE = MODELS_DIR / "feature_importance.csv"
PREDICTIONS_FILE = MODELS_DIR / "test_predictions.parquet"

# Time resolution of the demand forecast
INTERVAL = "15min"
INTERVALS_PER_DAY = 96          # 24 h * 4
INTERVALS_PER_WEEK = 96 * 7

# Zones that together cover this share of all pickups are modelled
# (the rest have almost no taxi pickups and would only add noise).
ZONE_COVERAGE = 0.97

# Cleaning rules (removes GPS/meter errors and extreme outliers)
MIN_DISTANCE_MILES = 0.1
MAX_DISTANCE_MILES = 50
MIN_FARE = 1.0
MAX_FARE = 250
MIN_DURATION_MIN = 1
MAX_DURATION_MIN = 180
UNKNOWN_ZONE_IDS = [264, 265]   # "Unknown" / "Outside of NYC" in the TLC lookup table

RANDOM_STATE = 42
