"""Step 1 - Data processing.

Reads the raw NYC TLC yellow-taxi trip files, removes bad records and turns
millions of individual trips into a demand time series:

    number of pickups  per taxi zone  per 15-minute interval

Run:  python -m src.data_processing
"""
import json
import re

import numpy as np
import pandas as pd

from src import config
from src.zones import build_zone_geometry

TRIP_COLUMNS = ["tpep_pickup_datetime", "tpep_dropoff_datetime",
                "PULocationID", "trip_distance", "fare_amount"]


def load_month(path):
    """Load one monthly file and keep only pickups that fall inside that month.
    (TLC files contain a few trips with wrong timestamps, e.g. from 2008 or 2090.)"""
    year, month = map(int, re.search(r"(\d{4})-(\d{2})", path.name).groups())
    df = pd.read_parquet(path, columns=TRIP_COLUMNS)
    start = pd.Timestamp(year=year, month=month, day=1)
    end = start + pd.offsets.MonthBegin(1)
    in_month = df["tpep_pickup_datetime"].between(start, end, inclusive="left")
    return df.loc[in_month]


def clean_trips(df):
    """Remove records that are clearly errors. Returns (clean_df, stats)."""
    stats = {"raw_trips": int(len(df))}
    duration_min = (df["tpep_dropoff_datetime"] - df["tpep_pickup_datetime"]).dt.total_seconds() / 60

    rules = {
        "unknown_zone": ~df["PULocationID"].isin(config.UNKNOWN_ZONE_IDS),
        "distance": df["trip_distance"].between(config.MIN_DISTANCE_MILES, config.MAX_DISTANCE_MILES),
        "fare": df["fare_amount"].between(config.MIN_FARE, config.MAX_FARE),
        "duration": duration_min.between(config.MIN_DURATION_MIN, config.MAX_DURATION_MIN),
    }
    keep = np.ones(len(df), dtype=bool)
    for name, rule in rules.items():
        rule = rule.to_numpy()
        stats[f"removed_{name}"] = int((keep & ~rule).sum())   # removed by this rule (not earlier ones)
        keep &= rule

    clean = df.loc[keep, ["tpep_pickup_datetime", "PULocationID"]]
    stats["clean_trips"] = int(len(clean))
    return clean, stats


def select_zones(zone_counts, coverage):
    """Busiest zones that together account for `coverage` of all pickups."""
    ordered = zone_counts.sort_values(ascending=False)
    share = ordered.cumsum() / ordered.sum()
    n_zones = int((share < coverage).sum()) + 1
    return ordered.index[:n_zones].tolist()


def build_demand(trips, zone_ids):
    """Count pickups per zone per 15 minutes, including intervals with 0 pickups."""
    trips = trips[trips["PULocationID"].isin(zone_ids)]
    interval = trips["tpep_pickup_datetime"].dt.floor(config.INTERVAL)
    counts = trips.groupby([trips["PULocationID"], interval]).size()

    full_index = pd.MultiIndex.from_product(
        [sorted(zone_ids),
         pd.date_range(interval.min(), interval.max(), freq=config.INTERVAL)],
        names=["zone_id", "timestamp"])
    counts.index.names = ["zone_id", "timestamp"]
    demand = counts.reindex(full_index, fill_value=0).rename("pickups").reset_index()
    demand["zone_id"] = demand["zone_id"].astype("int16")
    demand["pickups"] = demand["pickups"].astype("int32")
    return demand


def main():
    files = sorted(config.RAW_DIR.glob(config.TRIP_FILES_GLOB))
    if not files:
        raise SystemExit(f"No trip files found in {config.RAW_DIR}. See README for the download links.")
    print("Reading:", ", ".join(f.name for f in files))

    trips, stats = [], {}
    for path in files:
        month_df, month_stats = clean_trips(load_month(path))
        trips.append(month_df)
        for key, value in month_stats.items():
            stats[key] = stats.get(key, 0) + value
        print(f"  {path.name}: {month_stats['raw_trips']:,} trips -> {month_stats['clean_trips']:,} after cleaning")
    trips = pd.concat(trips, ignore_index=True)

    # Zone information (names, boroughs) + total pickups for every zone
    zones = pd.read_csv(config.ZONE_LOOKUP_FILE).rename(columns={
        "LocationID": "zone_id", "Borough": "borough", "Zone": "zone", "service_zone": "service_zone"})
    zone_counts = trips["PULocationID"].value_counts()
    zones["total_pickups"] = zones["zone_id"].map(zone_counts).fillna(0).astype(int)

    modelled = select_zones(zone_counts, config.ZONE_COVERAGE)
    zones["modelled"] = zones["zone_id"].isin(modelled)

    # Zone centroids + simplified boundaries for the map (needs the shapefile)
    boundaries = config.find_zone_boundaries()
    if boundaries is not None:
        print(f"Building zone map from {boundaries.name}")
        centroids = build_zone_geometry(boundaries, config.ZONES_GEOJSON)
        zones = zones.merge(centroids, on="zone_id", how="left")
    else:
        print("Zone boundary file not found - the map page will be disabled.")
    zones = zones[~zones["zone_id"].isin(config.UNKNOWN_ZONE_IDS)]

    demand = build_demand(trips, modelled)

    stats.update({
        "months": [re.search(r"(\d{4}-\d{2})", f.name).group(1) for f in files],
        "start": str(demand["timestamp"].min()),
        "end": str(demand["timestamp"].max()),
        "zones_total": int((zones["total_pickups"] > 0).sum()),
        "zones_modelled": len(modelled),
        "modelled_share": round(float(zone_counts.loc[modelled].sum() / zone_counts.sum()), 4),
    })

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    demand.to_parquet(config.DEMAND_FILE, index=False)
    zones.to_csv(config.ZONES_FILE, index=False)
    config.SUMMARY_FILE.write_text(json.dumps(stats, indent=2))

    print(f"\nClean trips: {stats['clean_trips']:,} of {stats['raw_trips']:,}")
    print(f"Modelled zones: {len(modelled)} (cover {stats['modelled_share']:.1%} of pickups)")
    print(f"Demand table: {len(demand):,} rows -> {config.DEMAND_FILE.name}")


if __name__ == "__main__":
    main()
