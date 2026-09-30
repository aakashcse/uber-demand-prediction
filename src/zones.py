"""Converts the official TLC taxi-zone boundaries into:
  * one centre point (lat/lon) per zone, and
  * a small GeoJSON file with simplified zone boundaries for the demand map.

Supports both formats the TLC has published: the new GeoParquet file and the
older zipped shapefile. Boundaries are stored in the New York State Plane
coordinate system (EPSG:2263, feet) and are converted to latitude/longitude.
Only needed when (re)building the data - the app just reads the output files.
"""
import json
import struct
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- readers
def _parse_wkb(buf):
    """Minimal WKB reader for Polygon / MultiPolygon -> list of polygons, each a list of rings."""
    def read_polygon(offset):
        order = "<" if buf[offset] == 1 else ">"
        gtype = struct.unpack_from(order + "I", buf, offset + 1)[0]
        offset += 5
        if gtype & 0x20000000:          # EWKB with embedded SRID
            offset += 4
        gtype &= 0xFFFF
        if gtype % 1000 == 6:           # MultiPolygon
            n = struct.unpack_from(order + "I", buf, offset)[0]
            offset += 4
            polys = []
            for _ in range(n):
                p, offset = read_polygon(offset)
                polys.extend(p)
            return polys, offset
        dims = 3 if gtype // 1000 in (1, 2) else 4 if gtype // 1000 == 3 else 2
        n_rings = struct.unpack_from(order + "I", buf, offset)[0]
        offset += 4
        rings = []
        for _ in range(n_rings):
            n_pts = struct.unpack_from(order + "I", buf, offset)[0]
            offset += 4
            pts = np.frombuffer(buf, dtype=order + "f8", count=n_pts * dims, offset=offset).reshape(n_pts, dims)
            rings.append(pts[:, :2].copy())
            offset += n_pts * dims * 8
        return [rings], offset
    return read_polygon(0)[0]


def _read_geoparquet(path):
    import pyarrow.parquet as pq
    table = pq.read_table(path)
    meta = json.loads((table.schema.metadata or {}).get(b"geo", b"{}") or b"{}")
    geom_col = meta.get("primary_column", "geometry")
    crs = meta.get("columns", {}).get(geom_col, {}).get("crs") or {}
    epsg = (crs.get("id") or {}).get("code") if isinstance(crs, dict) else None
    df = table.to_pandas()
    id_col = next(c for c in df.columns if c.lower() == "locationid")
    for zone_id, wkb in zip(df[id_col], df[geom_col]):
        yield int(zone_id), _parse_wkb(bytes(wkb)), epsg


def _read_shapefile_zip(path):
    import shapefile  # pyshp
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(path) as zf:
            zf.extractall(tmp)
        reader = shapefile.Reader(str(next(Path(tmp).rglob("*.shp"))))
        fields = [f[0].lower() for f in reader.fields[1:]]
        for record, shape in zip(reader.records(), reader.shapes()):
            pts = np.asarray(shape.points, dtype=float)
            parts = list(shape.parts) + [len(pts)]
            rings = [pts[a:b] for a, b in zip(parts[:-1], parts[1:])]
            yield int(dict(zip(fields, record))["locationid"]), [[r] for r in rings], 2263


# ---------------------------------------------------------------- geometry helpers
def _simplify(points, tolerance):
    """Ramer-Douglas-Peucker line simplification (keeps the shape, drops extra points)."""
    if len(points) < 3:
        return points
    start, end = points[0], points[-1]
    line = end - start
    norm = np.hypot(*line)
    dists = np.hypot(*(points - start).T) if norm == 0 else np.abs(np.cross(line, points - start)) / norm
    idx = int(np.argmax(dists))
    if dists[idx] > tolerance:
        left = _simplify(points[: idx + 1], tolerance)
        right = _simplify(points[idx:], tolerance)
        return np.vstack([left[:-1], right])
    return np.vstack([start, end])


def _area_centroid(ring):
    x, y = ring[:, 0], ring[:, 1]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    area = cross.sum() / 2
    if area == 0:
        return 0.0, x.mean(), y.mean()
    return abs(area), ((x[:-1] + x[1:]) * cross).sum() / (6 * area), ((y[:-1] + y[1:]) * cross).sum() / (6 * area)


# ---------------------------------------------------------------- main entry
def build_zone_geometry(source: Path, geojson_out: Path):
    from pyproj import Transformer

    reader = _read_geoparquet(source) if source.suffix == ".parquet" else _read_shapefile_zip(source)
    features, centroids, transformers = [], [], {}

    for zone_id, polygons, epsg in reader:
        sample = polygons[0][0]
        projected = np.abs(sample).max() > 360          # feet / metres, not degrees
        epsg = int(epsg or 2263) if projected else 4326
        if epsg not in transformers and epsg != 4326:
            transformers[epsg] = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
        tolerance = 120 if projected else 0.0004         # ~120 ft

        def to_lonlat(pts):
            if epsg == 4326:
                return pts[:, 0], pts[:, 1]
            return transformers[epsg].transform(pts[:, 0], pts[:, 1])

        out_polys, rings_info = [], []
        for rings in polygons:
            outer = rings[0]
            rings_info.append(_area_centroid(outer))
            simple = _simplify(outer, tolerance)
            if len(simple) < 4:
                continue
            lon, lat = to_lonlat(simple)
            out_polys.append([[[round(float(a), 5), round(float(b), 5)] for a, b in zip(lon, lat)]])

        _, cx, cy = max(rings_info, key=lambda t: t[0])   # centre of the largest part
        lon_c, lat_c = to_lonlat(np.array([[cx, cy]]))
        centroids.append({"zone_id": zone_id, "lat": round(float(np.ravel(lat_c)[0]), 5),
                          "lon": round(float(np.ravel(lon_c)[0]), 5)})
        if out_polys:
            features.append({"type": "Feature", "properties": {"zone_id": zone_id},
                             "geometry": {"type": "MultiPolygon", "coordinates": out_polys}})

    geojson_out.parent.mkdir(parents=True, exist_ok=True)
    geojson_out.write_text(json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":")))
    return pd.DataFrame(centroids).drop_duplicates("zone_id")
