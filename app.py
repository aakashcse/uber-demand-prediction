"""Uber Demand Prediction - NYC
Streamlit app: dashboard, demand prediction, demand map and model insights.

Run:  streamlit run app.py
"""
import copy
import datetime as dt
import json

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pydeck as pdk
import streamlit as st

from src import charts, config
from src.features import FEATURE_DESCRIPTIONS, FEATURES, build_features, manual_feature_row
from src.model import predict

st.set_page_config(page_title="Uber Demand Prediction", page_icon="🚕", layout="wide")

st.markdown("""
<style>
  .block-container {padding-top: 2rem; max-width: 1300px;}
  .hero h1 {font-size: 2.1rem; margin: 0 0 .2rem 0;}
  .hero p {color: #c3c2b7; margin: 0 0 1.2rem 0;}
  .kpi {background: #1a1a19; border: 1px solid rgba(255,255,255,0.10); border-radius: 12px;
        padding: 14px 16px; min-height: 112px;}
  .kpi .label {color: #898781; font-size: .75rem; text-transform: uppercase; letter-spacing: .06em;}
  .kpi .value {color: #ffffff; font-size: 1.6rem; font-weight: 650; line-height: 1.3; margin-top: 2px;
               white-space: nowrap; overflow: hidden; text-overflow: ellipsis;}
  .kpi .sub {color: #c3c2b7; font-size: .82rem;}
  .result {background: #1a1a19; border: 1px solid rgba(255,255,255,0.10); border-left: 4px solid #3987e5;
           border-radius: 12px; padding: 18px 20px;}
  .result .big {font-size: 2.6rem; font-weight: 700; color: #ffffff; line-height: 1.1;}
  .pill {display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: .8rem; font-weight: 600;
         border: 1px solid rgba(255,255,255,0.15); color: #ffffff;}
  .muted {color: #898781; font-size: .85rem;}
  .legend-bar {height: 10px; border-radius: 4px; margin: 4px 0;
               background: linear-gradient(90deg, #104281, #256abf, #3987e5, #6da7ec, #9ec5f4, #cde2fb);}
  h4 {margin-top: .6rem !important;}
</style>
""", unsafe_allow_html=True)

DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


# ------------------------------------------------------------------ data loading
@st.cache_data(show_spinner=False)
def load_data():
    demand = pd.read_parquet(config.DEMAND_FILE)
    zones = pd.read_csv(config.ZONES_FILE)
    zones["label"] = zones["zone"] + " (" + zones["borough"] + ")"
    summary = json.loads(config.SUMMARY_FILE.read_text())
    metrics = json.loads(config.METRICS_FILE.read_text())
    importance = pd.read_csv(config.IMPORTANCE_FILE)
    test_pred = pd.read_parquet(config.PREDICTIONS_FILE)
    geojson = json.loads(config.ZONES_GEOJSON.read_text()) if config.ZONES_GEOJSON.exists() else None
    return demand, zones, summary, metrics, importance, test_pred, geojson


@st.cache_resource(show_spinner=False)
def load_model():
    return joblib.load(config.MODEL_FILE)


@st.cache_data(show_spinner=False)
def load_features():
    return build_features(pd.read_parquet(config.DEMAND_FILE)).set_index(["zone_id", "timestamp"]).sort_index()


with st.spinner("Loading data and model ..."):
    demand, zones, summary, metrics, importance, test_pred, geojson = load_data()
    bundle = load_model()
    features = load_features()

zone_info = zones.set_index("zone_id")
modelled_zones = zones[zones["modelled"]].sort_values("total_pickups", ascending=False)
test_start = pd.Timestamp(metrics["test_period"][0])
test_end = pd.Timestamp(metrics["test_period"][1])


# ------------------------------------------------------------------ helpers
def kpi(col, label, value, sub=""):
    col.markdown(f'<div class="kpi"><div class="label">{label}</div>'
                 f'<div class="value" title="{value}">{value}</div><div class="sub">{sub}</div></div>',
                 unsafe_allow_html=True)


def fmt_hour(h):
    return dt.time(int(h)).strftime("%I %p").lstrip("0")


def interval_label(ts):
    return f"{ts:%a %d %b %Y}, {ts:%H:%M}–{ts + pd.Timedelta(minutes=15):%H:%M}"


def demand_level(zone_id, value):
    """Compare a value with the zone's usual 15-min demand (training period)."""
    history = demand.loc[(demand["zone_id"] == zone_id) & (demand["timestamp"] < test_start), "pickups"]
    pct = (history < value).mean() * 100
    if pct >= 90:
        return "Very high", "#cde2fb", pct
    if pct >= 65:
        return "High", "#86b6ef", pct
    if pct >= 30:
        return "Normal", "#3987e5", pct
    return "Low", "#184f95", pct


def zone_picker(key, default_zone=None):
    boroughs = ["All boroughs"] + sorted(modelled_zones["borough"].unique())
    c1, c2 = st.columns([1, 2])
    borough = c1.selectbox("Borough", boroughs, key=f"{key}_borough")
    options = modelled_zones if borough == "All boroughs" else modelled_zones[modelled_zones["borough"] == borough]
    labels = options["label"].tolist()
    index = 0
    if default_zone is not None and default_zone in options["zone_id"].values:
        index = int(np.where(options["zone_id"].values == default_zone)[0][0])
    label = c2.selectbox("Pickup zone", labels, index=index, key=f"{key}_zone",
                         help="Zones are ordered from busiest to quietest.")
    return int(options.loc[options["label"] == label, "zone_id"].iloc[0])


def time_picker(key):
    c1, c2 = st.columns(2)
    date = c1.date_input("Date", value=test_start.date() + dt.timedelta(days=7),
                         min_value=test_start.date(), max_value=test_end.date(), key=f"{key}_date",
                         help="Limited to the test period, which the model never saw during training.")
    slots = [dt.time(h, m) for h in range(24) for m in (0, 15, 30, 45)]
    time = c2.selectbox("Time (15-min slot)", slots, index=slots.index(dt.time(18, 0)), key=f"{key}_time",
                        format_func=lambda t: f"{t:%H:%M} – {(dt.datetime.combine(dt.date.today(), t) + dt.timedelta(minutes=15)):%H:%M}")
    return pd.Timestamp.combine(date, time)


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown("## 🚕 Uber Demand\n**Prediction · NYC**")
    page = st.radio("Navigate", ["📊 Dashboard", "🔮 Predict Demand", "🗺️ Demand Map", "🧠 Model Insights"],
                    label_visibility="collapsed")
    st.divider()
    st.markdown(
        f"**Data:** NYC TLC Yellow Taxi trips  \n"
        f"**Period:** {pd.Timestamp(summary['start']):%d %b %Y} – {pd.Timestamp(summary['end']):%d %b %Y}  \n"
        f"**Trips:** {summary['clean_trips']:,} (after cleaning)  \n"
        f"**Zones modelled:** {summary['zones_modelled']} ({summary['modelled_share']:.0%} of pickups)  \n"
        f"**Model:** Gradient Boosting (Poisson)")
    st.caption("Forecast unit: pickups per taxi zone per 15 minutes.")

st.markdown('<div class="hero"><h1>🚕 Uber Demand Prediction</h1>'
            '<p>Forecasting ride demand across New York City taxi zones, 15 minutes ahead, '
            'using machine learning on real NYC trip records.</p></div>', unsafe_allow_html=True)


# ================================================================== DASHBOARD
if page == "📊 Dashboard":
    months = sorted(demand["timestamp"].dt.to_period("M").astype(str).unique())
    f1, f2, _ = st.columns([1, 1, 2])
    borough = f1.selectbox("Borough", ["All boroughs"] + sorted(modelled_zones["borough"].unique()))
    month = f2.selectbox("Month", ["All months"] + months,
                         format_func=lambda m: m if m == "All months" else pd.Period(m).strftime("%B %Y"))

    d = demand.merge(zones[["zone_id", "borough", "zone"]], on="zone_id")
    if borough != "All boroughs":
        d = d[d["borough"] == borough]
    if month != "All months":
        d = d[d["timestamp"].dt.to_period("M").astype(str) == month]

    per_interval = d.groupby("timestamp")["pickups"].sum()
    by_hour_total = per_interval.groupby(per_interval.index.hour).mean()
    daily = per_interval.resample("D").sum()
    by_day = daily.groupby(daily.index.dayofweek).mean()
    by_zone = d.groupby(["zone_id", "zone", "borough"])["pickups"].sum().sort_values(ascending=False).reset_index()

    k = st.columns(5)
    kpi(k[0], "Total trips", f"{int(d['pickups'].sum()):,}", f"{len(daily)} days · modelled zones")
    kpi(k[1], "Avg demand / 15 min", f"{per_interval.mean():,.0f}", "pickups, selected zones")
    peak = int(by_hour_total.idxmax())
    kpi(k[2], "Peak hour", f"{fmt_hour(peak)}–{fmt_hour((peak + 1) % 24)}", f"{by_hour_total.max():,.0f} pickups / 15 min")
    kpi(k[3], "Busiest day", dt.date(2024, 1, 1 + int(by_day.idxmax())).strftime("%A"),
        f"{by_day.max():,.0f} trips on average")
    kpi(k[4], "Top zone", by_zone.iloc[0]["zone"], f"{int(by_zone.iloc[0]['pickups']):,} trips")

    st.markdown("#### Daily trips")
    fig = go.Figure(go.Scatter(x=daily.index, y=daily.values, mode="lines", line=dict(color=charts.ACTUAL, width=2),
                               hovertemplate="%{x|%a %d %b}: %{y:,.0f} trips<extra></extra>"))
    st.plotly_chart(charts.style(fig, height=260, legend=False, y_title="Trips per day"), width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Demand by hour of day")
        weekend = per_interval.index.dayofweek >= 5
        wk = per_interval[~weekend].groupby(per_interval[~weekend].index.hour).mean()
        we = per_interval[weekend].groupby(per_interval[weekend].index.hour).mean()
        fig = go.Figure()
        for series, name, color, dash in [(wk, "Weekdays", charts.ACTUAL, "solid"),
                                          (we, "Weekends", charts.PREDICTED, "dash")]:
            fig.add_trace(go.Scatter(x=series.index, y=series.values, name=name, mode="lines",
                                     line=dict(color=color, width=2, dash=dash),
                                     hovertemplate=name + ": %{y:,.0f}<extra></extra>"))
        fig.update_layout(hovermode="x unified")
        fig.update_xaxes(tickvals=list(range(0, 24, 3)), ticktext=[fmt_hour(h) for h in range(0, 24, 3)])
        st.plotly_chart(charts.style(fig, y_title="Avg pickups / 15 min"), width="stretch")
    with c2:
        st.markdown("#### Demand by day of week")
        fig = charts.bar([DAY_NAMES[i] for i in by_day.index], by_day.values, hover="%{x}: %{y:,.0f} trips",
                         y_title="Avg trips per day")
        st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Peak-demand heatmap (hour × day)")
        heat = (per_interval.groupby([per_interval.index.dayofweek.rename("day"),
                                      per_interval.index.hour.rename("hour")]).mean().unstack())
        fig = go.Figure(go.Heatmap(
            z=heat.values, x=[fmt_hour(h) for h in heat.columns], y=[DAY_NAMES[i] for i in heat.index],
            colorscale=[[i / (len(charts.SEQUENTIAL) - 1), c] for i, c in enumerate(charts.SEQUENTIAL)],
            xgap=2, ygap=2, colorbar=dict(title=dict(text="Pickups<br>/15 min", font=dict(color=charts.MUTED)),
                                          tickfont=dict(color=charts.MUTED), thickness=10),
            hovertemplate="%{y} %{x}: %{z:,.0f} pickups / 15 min<extra></extra>"))
        fig.update_yaxes(autorange="reversed", showgrid=False)
        st.plotly_chart(charts.style(fig, legend=False), width="stretch")
    with c2:
        st.markdown("#### Top 10 pickup zones")
        top = by_zone.head(10).iloc[::-1]
        fig = charts.bar(top["pickups"], top["zone"], orientation="h", hover="%{y}: %{x:,.0f} trips")
        fig.update_yaxes(showgrid=False)
        fig.update_xaxes(showgrid=True, gridcolor=charts.GRID)
        st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Busiest time slots")
        slots = heat.stack().sort_values(ascending=False).head(8).reset_index()
        slots.columns = ["day", "hour", "avg"]
        st.dataframe(pd.DataFrame({
            "Day": [DAY_NAMES[i] for i in slots["day"]],
            "Time": [f"{fmt_hour(h)}–{fmt_hour((h + 1) % 24)}" for h in slots["hour"]],
            "Avg pickups / 15 min": slots["avg"].round(0).astype(int)}),
            hide_index=True, width="stretch")
    with c2:
        st.markdown("#### Trips by borough")
        b = d.groupby("borough")["pickups"].sum().sort_values()
        fig = charts.bar(b.values, b.index, orientation="h", height=300, hover="%{y}: %{x:,.0f} trips")
        fig.update_yaxes(showgrid=False)
        fig.update_xaxes(showgrid=True, gridcolor=charts.GRID)
        st.plotly_chart(fig, width="stretch")


# ================================================================== PREDICT
elif page == "🔮 Predict Demand":
    tab1, tab2 = st.tabs(["📅 Predict from real history", "🧪 What-if scenario"])

    with tab1:
        st.caption("Pick a zone and a time. The model uses the real demand just before that time "
                   "(last hour, yesterday, last week) to forecast the next 15 minutes, "
                   "then we compare it with what actually happened.")
        zone_id = zone_picker("hist")
        ts = time_picker("hist")
        if st.button("Predict demand", type="primary", key="hist_btn"):
            st.session_state.hist_on = True

        if st.session_state.get("hist_on"):
            try:
                row = features.loc[[(zone_id, ts)]].reset_index()
            except KeyError:
                st.error("No data is available for this zone and time. Please choose another time.")
                st.stop()

            pred = float(predict(bundle, row)[0])
            actual = int(row["pickups"].iloc[0])
            level, color, pct = demand_level(zone_id, pred)
            name = zone_info.loc[zone_id, "label"]

            c1, c2 = st.columns([1.2, 1])
            with c1:
                st.markdown(f"""
                <div class="result">
                  <div class="muted">{name} · {interval_label(ts)}</div>
                  <div class="big">{pred:,.0f} pickups</div>
                  <div style="margin-top:6px">predicted demand &nbsp;
                    <span class="pill" style="background:{color}33">{level} demand</span></div>
                  <div class="muted" style="margin-top:10px">Busier than {pct:.0f}% of this zone's
                    15-minute intervals during training.</div>
                </div>""", unsafe_allow_html=True)
            with c2:
                k = st.columns(2)
                kpi(k[0], "Actual pickups", f"{actual:,}", "what really happened")
                err = pred - actual
                kpi(k[1], "Prediction error", f"{err:+.1f}", f"{abs(err) / max(actual, 1):.0%} of actual")

            st.markdown("#### Inputs the model used")
            used = row[FEATURES].T.reset_index()
            used.columns = ["Feature", "Value"]
            used["Meaning"] = used["Feature"].map(FEATURE_DESCRIPTIONS)
            used["Value"] = used["Value"].map(lambda v: f"{float(v):,.0f}" if float(v).is_integer() else f"{float(v):,.2f}")
            st.dataframe(used, hide_index=True, width="stretch", height=280)

            st.markdown(f"#### Actual vs predicted — {name}, {ts:%A %d %b}")
            day = features.loc[zone_id].loc[ts.normalize(): ts.normalize() + pd.Timedelta(hours=23, minutes=45)].reset_index()
            day["zone_id"] = zone_id
            day_pred = predict(bundle, day)
            fig = charts.actual_vs_predicted(day["timestamp"], day["pickups"], day_pred, marker_x=ts,
                                             y_title="Pickups / 15 min")
            st.plotly_chart(fig, width="stretch")
            day_wape = np.abs(day["pickups"] - day_pred).sum() / max(day["pickups"].sum(), 1)
            st.caption(f"Dotted line = selected time. Over this whole day the model's total error was "
                       f"{day_wape:.0%} of actual demand (WAPE).")

    with tab2:
        st.caption("Enter your own recent demand values to see how the model reacts. "
                   "Useful for testing scenarios such as a sudden spike or an unusually quiet hour.")
        zone_id = zone_picker("what", default_zone=int(modelled_zones.iloc[0]["zone_id"]))
        ts = time_picker("what")
        hist = demand[(demand["zone_id"] == zone_id) & (demand["timestamp"] < test_start)]
        typical = hist[hist["timestamp"].dt.hour == ts.hour]["pickups"].mean()
        typical = int(round(typical)) if pd.notna(typical) else 10
        st.markdown(f"<span class='muted'>Defaults are this zone's typical demand at "
                    f"{fmt_hour(ts.hour)} ({typical} pickups / 15 min).</span>", unsafe_allow_html=True)
        c = st.columns(6)
        lags = [c[i].number_input(lbl, min_value=0, max_value=2000, value=typical, step=1)
                for i, lbl in enumerate(["15 min ago", "30 min ago", "45 min ago", "60 min ago"])]
        yesterday = c[4].number_input("Same time yesterday", min_value=0, max_value=2000, value=typical)
        last_week = c[5].number_input("Same time last week", min_value=0, max_value=2000, value=typical)
        if st.button("Predict demand", type="primary", key="what_btn"):
            st.session_state.what_on = True

        if st.session_state.get("what_on"):
            row = manual_feature_row(zone_id, ts, lags, yesterday, last_week)
            pred = float(predict(bundle, row)[0])
            level, color, pct = demand_level(zone_id, pred)
            change = pred - typical
            st.markdown(f"""
            <div class="result">
              <div class="muted">{zone_info.loc[zone_id, 'label']} · {interval_label(ts)}</div>
              <div class="big">{pred:,.0f} pickups</div>
              <div style="margin-top:6px">predicted for the next 15 minutes &nbsp;
                <span class="pill" style="background:{color}33">{level} demand</span></div>
              <div class="muted" style="margin-top:10px">{change:+.0f} vs this zone's typical
                {fmt_hour(ts.hour)} demand ({typical}).</div>
            </div>""", unsafe_allow_html=True)


# ================================================================== MAP
elif page == "🗺️ Demand Map":
    st.caption("Predicted demand for every modelled zone at the selected time (test period). "
               "Hover a zone to compare prediction and actual pickups.")
    c1, c2 = st.columns([1, 2])
    with c1:
        ts = time_picker("map")
    try:
        snap = features.xs(ts, level="timestamp").reset_index()
    except KeyError:
        snap = None
    if snap is None or snap.empty:
        st.error("No data for this time. Please choose another time.")
        st.stop()
    snap["predicted"] = predict(bundle, snap)
    snap = snap.merge(zones[["zone_id", "zone", "borough", "lat", "lon"]]
                      if "lat" in zones else zones[["zone_id", "zone", "borough"]], on="zone_id")

    k = st.columns(4)
    kpi(k[0], "Predicted pickups", f"{snap['predicted'].sum():,.0f}", "all modelled zones, 15 min")
    kpi(k[1], "Actual pickups", f"{int(snap['pickups'].sum()):,}", interval_label(ts).split(", ")[1])
    busiest = snap.sort_values("predicted", ascending=False).iloc[0]
    kpi(k[2], "Hottest zone", busiest["zone"], f"{busiest['predicted']:.0f} predicted pickups")
    kpi(k[3], "Zones above 50", f"{int((snap['predicted'] >= 50).sum())}", "predicted pickups / 15 min")

    m1, m2 = st.columns([2, 1])
    with m1:
        if geojson is not None:
            # colour by quantile bins of predicted demand (log-like spread for skewed data)
            values = snap.set_index("zone_id")
            bins = np.unique(np.quantile(snap["predicted"], np.linspace(0, 1, 8)))
            ramp = charts.SEQUENTIAL[::2] + [charts.SEQUENTIAL[-1]]
            gj = copy.deepcopy(geojson)
            for feat in gj["features"]:
                zid = feat["properties"]["zone_id"]
                info = zone_info.loc[zid] if zid in zone_info.index else None
                feat["properties"]["name"] = f"{info['zone']} ({info['borough']})" if info is not None else "Unknown"
                if zid in values.index:
                    p = float(values.loc[zid, "predicted"])
                    idx = int(np.clip(np.searchsorted(bins, p, side="right") - 1, 0, len(ramp) - 1))
                    rgb = [int(ramp[idx][i:i + 2], 16) for i in (1, 3, 5)]
                    feat["properties"].update(fill=rgb + [210], predicted=f"{p:.0f}",
                                              actual=f"{int(values.loc[zid, 'pickups'])}")
                else:
                    feat["properties"].update(fill=[60, 60, 58, 90], predicted="not modelled", actual="–")
            layer = pdk.Layer("GeoJsonLayer", gj, pickable=True, stroked=True, filled=True,
                              get_fill_color="properties.fill", get_line_color=[26, 26, 25, 255],
                              line_width_min_pixels=1)
            deck = pdk.Deck(layers=[layer], map_provider="carto", map_style=pdk.map_styles.CARTO_DARK,
                            initial_view_state=pdk.ViewState(latitude=40.73, longitude=-73.94, zoom=10.2),
                            tooltip={"html": "<b>{name}</b><br/>Predicted: {predicted}<br/>Actual: {actual}",
                                     "style": {"backgroundColor": "#262624", "color": "#ffffff"}})
            st.pydeck_chart(deck, height=520)
            st.markdown(f'<div class="legend-bar"></div><div class="muted" style="display:flex;justify-content:space-between">'
                        f'<span>Low ({bins[0]:.0f})</span><span>Predicted pickups / 15 min</span>'
                        f'<span>High ({bins[-1]:.0f})</span></div>', unsafe_allow_html=True)
        else:
            st.info("Zone boundaries not found (data/processed/zones.geojson). Showing the table only.")
    with m2:
        st.markdown("#### Top 10 zones right now")
        top = snap.sort_values("predicted", ascending=False).head(10)
        st.dataframe(pd.DataFrame({"Zone": top["zone"], "Predicted": top["predicted"].round(0).astype(int),
                                   "Actual": top["pickups"].astype(int)}),
                     hide_index=True, width="stretch", height=420)


# ================================================================== MODEL INSIGHTS
elif page == "🧠 Model Insights":
    res = pd.DataFrame(metrics["results"]).T
    best = res.loc[metrics["chosen_model"]]
    base = res.loc["Baseline: last 15 min"]

    k = st.columns(4)
    kpi(k[0], "Algorithm", "HistGBR", "Gradient Boosting · Poisson loss")
    kpi(k[1], "MAE (test)", f"{best['MAE']:.2f}", "pickups per zone per 15 min")
    kpi(k[2], "R² (test)", f"{best['R2']:.3f}", "share of variation explained")
    kpi(k[3], "Error vs baseline", f"{(best['MAE'] / base['MAE'] - 1):+.0%}", "MAE vs 'repeat last 15 min'")

    st.markdown("#### Why this model")
    st.markdown(
        "- **Gradient boosting** builds many small decision trees, each correcting the previous ones. "
        "It captures non-linear patterns such as *rush hour at Midtown ≠ rush hour at JFK* without manual feature crosses.\n"
        "- **Poisson loss** – the target is a count of pickups (never negative, spread grows with the mean), "
        "which is exactly what the Poisson distribution models.\n"
        "- **Native categorical support** for the zone, fast training on hundreds of thousands of rows, "
        "and no extra libraries beyond scikit-learn.\n"
        f"- It was compared with two baselines and **Linear Regression** (the classic approach) on a "
        f"**time-based split**: trained on data before {test_start:%d %b %Y}, tested on "
        f"{test_start:%d %b} – {test_end:%d %b %Y}, a month the model never saw.")

    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown("#### Model comparison (test month)")
        table = res.copy()
        table["WAPE"] = (table["WAPE"] * 100).round(1).astype(str) + "%"
        table.index.name = "Model"
        st.dataframe(table, width="stretch")
        st.caption("MAE/RMSE in pickups per zone per 15 min (lower is better). "
                   "WAPE = total absolute error ÷ total demand. R²: higher is better.")
    with c2:
        st.markdown("#### MAE by model")
        r = res["MAE"].sort_values(ascending=False)
        fig = charts.bar(r.values, r.index, orientation="h", height=260, hover="%{y}: MAE %{x:.2f}")
        fig.update_yaxes(showgrid=False)
        fig.update_xaxes(showgrid=True, gridcolor=charts.GRID)
        st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Feature importance")
        imp = importance.sort_values("importance")
        fig = charts.bar(imp["importance"], imp["feature"], orientation="h", height=420,
                         hover="%{y}: +%{x:.3f} MAE when shuffled")
        fig.update_yaxes(showgrid=False)
        fig.update_xaxes(showgrid=True, gridcolor=charts.GRID, title=dict(text="Increase in MAE when shuffled",
                                                                           font=dict(color=charts.MUTED)))
        st.plotly_chart(fig, width="stretch")
    with c2:
        st.markdown("#### Features used")
        st.dataframe(pd.DataFrame({"Feature": FEATURES, "Description": [FEATURE_DESCRIPTIONS[f] for f in FEATURES]}),
                     hide_index=True, width="stretch", height=420)

    st.markdown("#### Historical vs predicted demand (test month, all modelled zones)")
    zone_opts = ["All modelled zones"] + modelled_zones["label"].tolist()
    choice = st.selectbox("Zone", zone_opts, key="insight_zone")
    tp = test_pred if choice == zone_opts[0] else \
        test_pred[test_pred["zone_id"] == int(modelled_zones.loc[modelled_zones["label"] == choice, "zone_id"].iloc[0])]
    hourly = tp.groupby("timestamp")[["pickups", "predicted"]].sum().resample("h").sum()
    fig = charts.actual_vs_predicted(hourly.index, hourly["pickups"], hourly["predicted"], height=320,
                                     y_title="Pickups per hour")
    fig.update_xaxes(rangeslider=dict(visible=True, bgcolor="#262624", thickness=0.06))
    st.plotly_chart(fig, width="stretch")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Where is the model least accurate? (MAE by hour)")
        err = test_pred.assign(abs_err=(test_pred["pickups"] - test_pred["predicted"]).abs(),
                               hour=test_pred["timestamp"].dt.hour).groupby("hour")["abs_err"].mean()
        fig = charts.bar([fmt_hour(h) for h in err.index], err.values, height=300,
                         hover="%{x}: MAE %{y:.2f}", y_title="MAE (pickups / 15 min)")
        st.plotly_chart(fig, width="stretch")
        st.caption("Errors are larger in busy hours simply because there are more pickups to predict.")
    with c2:
        st.markdown("#### Data cleaning summary")
        s = summary
        steps = pd.DataFrame({
            "Step": ["Raw trips (in-month)", "Unknown pickup zone", "Invalid distance",
                     "Invalid fare", "Invalid duration", "Clean trips"],
            "Trips": [s["raw_trips"], -s["removed_unknown_zone"], -s["removed_distance"],
                      -s["removed_fare"], -s["removed_duration"], s["clean_trips"]]})
        steps["Trips"] = steps["Trips"].map(lambda v: f"{v:+,}" if v < 0 else f"{v:,}")
        st.dataframe(steps, hide_index=True, width="stretch")
        st.caption(f"{s['zones_modelled']} busiest zones modelled ({s['modelled_share']:.0%} of all pickups); "
                   f"the remaining zones have too few pickups to forecast reliably.")

st.markdown("<br><div class='muted' style='text-align:center'>Built by Aakash Kumar Singh · "
            "Data: NYC Taxi & Limousine Commission · "
            "<a href='https://github.com/aakashcse/uber-demand-prediction' style='color:#86b6ef'>Source on GitHub</a>"
            "</div>", unsafe_allow_html=True)
