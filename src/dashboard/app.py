"""AirEquity NSW — polished Streamlit dashboard for Tasks 1–4.

Run from repository root:
    streamlit run src/dashboard/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

THRESHOLD = 25.0
OBS_PATH = Path("data/processed/observations")
FEATURES_PATH = Path("data/processed/features")
SITES_PATH = Path("data/raw/sites.json")
DEFAULT_STATION_START = pd.Timestamp("2023-09-01")
DEFAULT_STATION_END = pd.Timestamp("2023-09-30")
DEFAULT_EVENT_START = pd.Timestamp("2023-09-10")
DEFAULT_EVENT_END = pd.Timestamp("2023-09-14")

st.set_page_config(
    page_title="AirEquity NSW",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------------------
# Visual system — force a clean light dashboard even when the browser/Streamlit
# preference is dark. This also fixes low-contrast widget labels.
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
      :root {
        --ink:#10233f;
        --muted:#64748b;
        --line:#dce7f2;
        --card:#ffffff;
        --teal:#0f9aa8;
        --blue:#2563eb;
        --navy:#0b3b63;
        --red:#ef4444;
        --orange:#f97316;
        --gold:#f4b400;
      }

      html, body, [class*="css"], .stApp {
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }

      .stApp {
        background:
          radial-gradient(circle at 90% 0%, rgba(37,99,235,.09), transparent 25%),
          radial-gradient(circle at 6% 20%, rgba(15,154,168,.07), transparent 22%),
          #f5f9fd;
        color: var(--ink);
      }

      [data-testid="stHeader"] {background: rgba(245,249,253,.88);}
      [data-testid="stToolbar"] {right: 1rem;}
      .block-container {max-width: 1480px; padding-top: 1.0rem; padding-bottom: 2.2rem;}
      footer {visibility:hidden;}

      /* Keep all Streamlit labels readable in light mode. */
      label, [data-testid="stWidgetLabel"], [data-testid="stMarkdownContainer"] p {
        color: var(--ink) !important;
      }
      div[data-baseweb="select"] > div,
      div[data-testid="stDateInput"] input {
        background:#ffffff !important;
        color:var(--ink) !important;
        border-color:#d6e1ec !important;
      }

      .hero {
        position:relative;
        overflow:hidden;
        border-radius:24px;
        padding:1.35rem 1.55rem 1.25rem;
        background:linear-gradient(120deg,#087f8c 0%,#075985 46%,#2457c6 100%);
        color:#fff;
        box-shadow:0 18px 42px rgba(7,89,133,.18);
        margin:.2rem 0 1rem;
      }
      .hero:after {
        content:""; position:absolute; width:280px; height:280px; border-radius:50%;
        right:-85px; top:-160px; background:rgba(255,255,255,.10);
      }
      .hero-row {display:flex; align-items:center; justify-content:space-between; gap:1rem;}
      .brand {font-size:2rem; font-weight:850; letter-spacing:-.03em; line-height:1.1;}
      .tagline {font-size:.98rem; opacity:.92; margin-top:.35rem;}
      .hero-badges {display:flex; gap:.5rem; flex-wrap:wrap; justify-content:flex-end;}
      .badge {background:rgba(255,255,255,.15); border:1px solid rgba(255,255,255,.22); padding:.42rem .72rem; border-radius:999px; font-size:.78rem; font-weight:650; white-space:nowrap;}

      .control-card {
        background:#fff; border:1px solid var(--line); border-radius:18px;
        padding:.75rem .95rem .25rem; box-shadow:0 8px 24px rgba(15,23,42,.045);
        margin-bottom:.85rem;
      }
      .eyebrow {font-size:.73rem; text-transform:uppercase; letter-spacing:.08em; color:#0f7890; font-weight:800; margin-bottom:.2rem;}
      .section-title {font-size:1.18rem; color:var(--ink); font-weight:800; margin:.15rem 0 .12rem; letter-spacing:-.015em;}
      .section-sub {font-size:.88rem; color:var(--muted); margin-bottom:.55rem;}

      .kpi {
        min-height:128px; border-radius:19px; padding:1rem 1.05rem;
        background:#fff; border:1px solid var(--line); box-shadow:0 10px 25px rgba(15,23,42,.055);
        position:relative; overflow:hidden;
      }
      .kpi:after {content:""; position:absolute; width:92px; height:92px; border-radius:50%; right:-28px; top:-28px; background:rgba(37,99,235,.05);}
      .kpi-icon {font-size:1.05rem; margin-bottom:.35rem;}
      .kpi-label {font-size:.76rem; text-transform:uppercase; letter-spacing:.06em; font-weight:800; color:#64748b;}
      .kpi-value {font-size:1.72rem; font-weight:850; color:var(--ink); margin-top:.15rem; letter-spacing:-.025em;}
      .kpi-note {font-size:.79rem; color:#64748b; margin-top:.26rem; line-height:1.35;}
      .accent-teal {border-top:4px solid #0f9aa8;}
      .accent-red {border-top:4px solid #ef4444;}
      .accent-orange {border-top:4px solid #f59e0b;}
      .accent-blue {border-top:4px solid #2563eb;}

      .chart-card {
        background:#fff; border:1px solid var(--line); border-radius:20px;
        padding:.9rem 1rem .55rem; box-shadow:0 10px 28px rgba(15,23,42,.05);
        height:100%;
      }
      .insight {
        background:linear-gradient(90deg,#eff6ff,#f5fbff);
        border:1px solid #cfe1f5; color:#234a76; border-radius:14px;
        padding:.72rem .85rem; font-size:.84rem; margin-top:.35rem;
      }
      .event-banner {
        background:linear-gradient(90deg,#fff7ed,#fffaf4);
        border:1px solid #fed7aa; border-radius:16px; padding:.75rem .9rem;
        color:#9a3412; font-size:.86rem; margin:.2rem 0 .8rem;
      }
      .footnote {color:#7b8ba1; font-size:.78rem; text-align:center; padding-top:.8rem;}

      div[data-testid="stPlotlyChart"] {border-radius:14px; overflow:hidden;}
      hr {border-color:#e7eef5 !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading PM2.5 observations…")
def load_observations() -> pd.DataFrame:
    df = pd.read_parquet(OBS_PATH)
    required = {"site_id", "site_name", "region", "timestamp", "parameter", "value", "frequency"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Observations missing columns: {sorted(missing)}")

    df = df[(df["parameter"] == "PM2.5") & (df["frequency"] == "Hourly average")].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["site_id", "site_name", "timestamp", "value"])


@st.cache_data(show_spinner="Loading crossing labels…")
def load_features() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH)
    required = {"site_id", "site_name", "region", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Features missing columns: {sorted(missing)}")
    df = df.copy()
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    return df.dropna(subset=["site_id", "site_name", "region", "label"])


@st.cache_data(show_spinner=False)
def load_sites() -> pd.DataFrame:
    if SITES_PATH.exists():
        with SITES_PATH.open(encoding="utf-8") as f:
            raw = json.load(f)
        sites = pd.DataFrame(
            [
                {
                    "site_id": s.get("Site_Id", s.get("site_id")),
                    "site_name": s.get("SiteName", s.get("site_name")),
                    "region": s.get("Region", s.get("region")),
                    "latitude": s.get("Latitude", s.get("latitude")),
                    "longitude": s.get("Longitude", s.get("longitude")),
                }
                for s in raw
            ]
        )
    else:
        cols = ["site_id", "site_name", "region", "latitude", "longitude"]
        sites = pd.read_parquet(OBS_PATH, columns=cols).drop_duplicates("site_id")

    sites["latitude"] = pd.to_numeric(sites["latitude"], errors="coerce")
    sites["longitude"] = pd.to_numeric(sites["longitude"], errors="coerce")
    return sites.dropna(subset=["site_id", "latitude", "longitude"]).drop_duplicates("site_id")


missing_paths = [str(p) for p in (OBS_PATH, FEATURES_PATH) if not p.exists()]
if missing_paths:
    st.error("Dashboard data is missing: " + ", ".join(missing_paths))
    st.stop()

try:
    obs = load_observations()
    feat = load_features()
    sites = load_sites()
except Exception as exc:
    st.error(f"Could not load dashboard data: {exc}")
    st.stop()

operational_ids = set(feat["site_id"].unique())
obs = obs[obs["site_id"].isin(operational_ids)].copy()
sites = sites[sites["site_id"].isin(operational_ids)].copy()

if obs.empty or feat.empty:
    st.warning("The PM2.5 dashboard dataset is empty.")
    st.stop()

# -----------------------------------------------------------------------------
# Derived metrics
# -----------------------------------------------------------------------------
station_rates = (
    feat.groupby(["site_id", "site_name", "region"], as_index=False)["label"]
    .mean()
    .rename(columns={"label": "crossing_rate"})
)
station_rates["crossing_pct"] = station_rates["crossing_rate"] * 100

regional = (
    feat.groupby("region", as_index=False)["label"]
    .mean()
    .rename(columns={"label": "crossing_rate"})
)
regional["crossing_pct"] = regional["crossing_rate"] * 100
regional = regional.sort_values("crossing_pct", ascending=False)

network_crossing = float(feat["label"].mean() * 100)
min_date = obs["timestamp"].min().date()
max_date = obs["timestamp"].max().date()
station_names = sorted(obs["site_name"].dropna().unique())

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero">
      <div class="hero-row">
        <div>
          <div class="brand">🌫️ AirEquity NSW</div>
          <div class="tagline">PM2.5 exposure intelligence for Greater Sydney</div>
        </div>
        <div class="hero-badges">
          <div class="badge">18 operational stations</div>
          <div class="badge">2023–2024</div>
          <div class="badge">Health threshold · 25 µg/m³</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Controls — all visible at the top, no large sidebar.
# -----------------------------------------------------------------------------
st.markdown('<div class="control-card">', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">Explore the network</div>', unsafe_allow_html=True)
fc1, fc2, fc3 = st.columns([1.15, 1.35, 1.35], gap="large")

with fc1:
    station = st.selectbox("Station", station_names, index=0)
with fc2:
    default_s0 = max(min_date, DEFAULT_STATION_START.date())
    default_s1 = min(max_date, DEFAULT_STATION_END.date())
    station_range = st.date_input(
        "Station trend window",
        value=(default_s0, default_s1),
        min_value=min_date,
        max_value=max_date,
        key="station_range",
    )
with fc3:
    default_e0 = max(min_date, DEFAULT_EVENT_START.date())
    default_e1 = min(max_date, DEFAULT_EVENT_END.date())
    event_range = st.date_input(
        "Smoke-event window",
        value=(default_e0, default_e1),
        min_value=min_date,
        max_value=max_date,
        key="event_range",
    )
st.markdown('</div>', unsafe_allow_html=True)

if isinstance(station_range, (tuple, list)) and len(station_range) == 2:
    station_start, station_end = station_range
else:
    station_start = station_end = station_range

if isinstance(event_range, (tuple, list)) and len(event_range) == 2:
    event_start, event_end = event_range
else:
    event_start = event_end = event_range

s0 = pd.Timestamp(station_start)
s1 = pd.Timestamp(station_end) + pd.Timedelta(days=1)
station_df = obs[
    (obs["site_name"] == station)
    & (obs["timestamp"] >= s0)
    & (obs["timestamp"] < s1)
].sort_values("timestamp")

es = pd.Timestamp(event_start)
ee = pd.Timestamp(event_end) + pd.Timedelta(days=1)
event_df = obs[(obs["timestamp"] >= es) & (obs["timestamp"] < ee)].copy()

if event_df.empty:
    peak_value = 0.0
    peak_station = "—"
    stations_over = 0
else:
    peak_row = event_df.loc[event_df["value"].idxmax()]
    peak_value = float(peak_row["value"])
    peak_station = str(peak_row["site_name"]).title()
    stations_over = int(event_df.loc[event_df["value"] > THRESHOLD, "site_id"].nunique())

station_rate_row = station_rates[station_rates["site_name"] == station]
selected_station_rate = float(station_rate_row["crossing_pct"].iloc[0]) if not station_rate_row.empty else np.nan
selected_region = str(station_rate_row["region"].iloc[0]) if not station_rate_row.empty else "—"

# -----------------------------------------------------------------------------
# KPI row
# -----------------------------------------------------------------------------
k1, k2, k3, k4 = st.columns(4, gap="medium")
with k1:
    st.markdown(
        f'<div class="kpi accent-teal"><div class="kpi-icon">📍</div><div class="kpi-label">Selected station</div>'
        f'<div class="kpi-value">{station.title()}</div><div class="kpi-note">{selected_region} · crossing rate {selected_station_rate:.2f}%</div></div>',
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f'<div class="kpi accent-blue"><div class="kpi-icon">🌐</div><div class="kpi-label">Network crossing rate</div>'
        f'<div class="kpi-value">{network_crossing:.2f}%</div><div class="kpi-note">24-hour future PM2.5 threshold labels across all feature rows</div></div>',
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f'<div class="kpi accent-red"><div class="kpi-icon">🔥</div><div class="kpi-label">Event peak PM2.5</div>'
        f'<div class="kpi-value">{peak_value:,.0f} µg/m³</div><div class="kpi-note">Peak at {peak_station} in the selected smoke-event window</div></div>',
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f'<div class="kpi accent-orange"><div class="kpi-icon">⚠️</div><div class="kpi-label">Stations over threshold</div>'
        f'<div class="kpi-value">{stations_over} / {len(operational_ids)}</div><div class="kpi-note">Stations with at least one hourly reading above 25 µg/m³ in event window</div></div>',
        unsafe_allow_html=True,
    )

st.write("")

# -----------------------------------------------------------------------------
# Station trend — full width, uncluttered.
# -----------------------------------------------------------------------------
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Station PM2.5 trend</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Hourly concentration for the selected station. The dashed red line is the 25 µg/m³ health threshold.</div>', unsafe_allow_html=True)

trend = go.Figure()
trend.add_trace(
    go.Scatter(
        x=station_df["timestamp"],
        y=station_df["value"],
        mode="lines",
        line=dict(color="#0f9aa8", width=2.1),
        fill="tozeroy",
        fillcolor="rgba(15,154,168,.08)",
        name=station.title(),
        hovertemplate="%{x|%d %b %Y %H:%M}<br><b>%{y:.1f} µg/m³</b><extra></extra>",
    )
)
trend.add_hline(
    y=THRESHOLD,
    line_dash="dash",
    line_color="#ef4444",
    line_width=1.8,
    annotation_text="25 µg/m³ threshold",
    annotation_position="top right",
    annotation_font_color="#dc2626",
)
trend.update_layout(
    height=360,
    template="plotly_white",
    margin=dict(l=18, r=18, t=10, b=15),
    paper_bgcolor="#ffffff",
    plot_bgcolor="#ffffff",
    font=dict(color="#243b53"),
    xaxis=dict(title="", gridcolor="#edf2f7", zeroline=False),
    yaxis=dict(title="PM2.5 (µg/m³)", gridcolor="#edf2f7", zeroline=False),
    hovermode="x unified",
    showlegend=False,
)
st.plotly_chart(trend, use_container_width=True, theme=None, config={"displayModeBar": False, "displaylogo": False})
st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# -----------------------------------------------------------------------------
# Map + regional comparison
# -----------------------------------------------------------------------------
left, right = st.columns([1.15, .85], gap="large")

with left:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Sydney station map</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Location, colour and marker size encode each station’s 24-hour threshold-crossing rate.</div>', unsafe_allow_html=True)

    mapped = station_rates.merge(sites[["site_id", "latitude", "longitude"]], on="site_id", how="inner")
    mapped = mapped.dropna(subset=["latitude", "longitude"])

    # OpenStreetMap is token-free and more reliable than the previous Carto style.
    map_fig = go.Figure(
        go.Scattermap(
            lat=mapped["latitude"],
            lon=mapped["longitude"],
            mode="markers",
            text=mapped["site_name"].str.title(),
            customdata=np.stack([mapped["region"], mapped["crossing_pct"]], axis=-1),
            marker=dict(
                size=12 + mapped["crossing_pct"].clip(lower=.2) * 5.2,
                color=mapped["crossing_pct"],
                colorscale=[[0, "#facc15"], [.45, "#fb923c"], [.75, "#ef4444"], [1, "#991b1b"]],
                cmin=max(0.0, float(mapped["crossing_pct"].min())),
                cmax=float(mapped["crossing_pct"].max()),
                colorbar=dict(title="Crossing<br>rate (%)", thickness=12, len=.68),
                opacity=.92,
            ),
            hovertemplate="<b>%{text}</b><br>%{customdata[0]}<br>Crossing rate %{customdata[1]:.2f}%<extra></extra>",
        )
    )
    map_fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#ffffff",
        font=dict(color="#243b53"),
        map=dict(
            style="open-street-map",
            center=dict(lat=float(mapped["latitude"].mean()), lon=float(mapped["longitude"].mean())),
            zoom=8.25,
        ),
    )
    st.plotly_chart(map_fig, use_container_width=True, theme=None, config={"displayModeBar": False, "displaylogo": False})
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Regional exposure comparison</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Average 24-hour threshold-crossing rate by Greater Sydney region.</div>', unsafe_allow_html=True)

    palette = {
        "Sydney North-west": "#ef4444",
        "Sydney South-west": "#f97316",
        "Sydney East": "#f4b400",
    }
    reg_plot = regional.sort_values("crossing_pct", ascending=True)
    bar = go.Figure(
        go.Bar(
            y=reg_plot["region"],
            x=reg_plot["crossing_pct"],
            orientation="h",
            marker=dict(color=[palette.get(x, "#2563eb") for x in reg_plot["region"]], line=dict(width=0)),
            text=[f"{x:.2f}%" for x in reg_plot["crossing_pct"]],
            textposition="outside",
            cliponaxis=False,
            hovertemplate="%{y}<br><b>%{x:.2f}%</b><extra></extra>",
        )
    )
    bar.update_layout(
        height=310,
        template="plotly_white",
        margin=dict(l=10, r=65, t=5, b=10),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        showlegend=False,
        xaxis=dict(title="Crossing rate (%)", gridcolor="#edf2f7", rangemode="tozero"),
        yaxis=dict(title="", showgrid=False),
        font=dict(color="#243b53"),
    )
    st.plotly_chart(bar, use_container_width=True, theme=None, config={"displayModeBar": False, "displaylogo": False})

    nw = regional[regional["region"] == "Sydney North-west"]
    east = regional[regional["region"] == "Sydney East"]
    if not nw.empty and not east.empty and east.iloc[0]["crossing_pct"] > 0:
        ratio = float(nw.iloc[0]["crossing_pct"] / east.iloc[0]["crossing_pct"])
        st.markdown(
            f'<div class="insight"><b>Key finding:</b> North-west is <b>{nw.iloc[0]["crossing_pct"]:.2f}%</b> versus '
            f'<b>{east.iloc[0]["crossing_pct"]:.2f}%</b> in the East — about <b>{ratio:.1f}× higher</b>.</div>',
            unsafe_allow_html=True,
        )
    st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# -----------------------------------------------------------------------------
# Smoke event — curated top lines + heatmap for readability.
# -----------------------------------------------------------------------------
st.markdown('<div class="chart-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Smoke event explorer</div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="section-sub">Network view from {pd.Timestamp(event_start).strftime("%d %b %Y")} to {pd.Timestamp(event_end).strftime("%d %b %Y")}. '
    'Instead of plotting all 18 lines at once, the chart defaults to the six stations with the highest event peaks.</div>',
    unsafe_allow_html=True,
)

if event_df.empty:
    st.info("No observations are available in the selected event window.")
else:
    peaks = event_df.groupby("site_name")["value"].max().sort_values(ascending=False)
    default_event_stations = peaks.head(min(6, len(peaks))).index.tolist()
    selected_event_stations = st.multiselect(
        "Stations shown in event chart",
        options=peaks.index.tolist(),
        default=default_event_stations,
        max_selections=8,
    )

    show_event = event_df[event_df["site_name"].isin(selected_event_stations)].sort_values("timestamp")
    colors = ["#ef4444", "#f97316", "#f4b400", "#0f9aa8", "#2563eb", "#7c3aed", "#db2777", "#16a34a"]

    event_fig = go.Figure()
    for i, name in enumerate(selected_event_stations):
        g = show_event[show_event["site_name"] == name]
        event_fig.add_trace(
            go.Scatter(
                x=g["timestamp"],
                y=g["value"],
                mode="lines",
                name=name.title(),
                line=dict(color=colors[i % len(colors)], width=2.0),
                hovertemplate=f"<b>{name.title()}</b><br>%{{x|%d %b %H:%M}}<br>%{{y:.1f}} µg/m³<extra></extra>",
            )
        )
    event_fig.add_hline(
        y=THRESHOLD,
        line_dash="dash",
        line_color="#dc2626",
        line_width=1.6,
        annotation_text="25 µg/m³ threshold",
        annotation_position="top right",
        annotation_font_color="#dc2626",
    )
    event_fig.update_layout(
        height=390,
        template="plotly_white",
        margin=dict(l=20, r=20, t=15, b=20),
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#243b53"),
        xaxis=dict(title="", gridcolor="#edf2f7"),
        yaxis=dict(title="PM2.5 (µg/m³)", gridcolor="#edf2f7", rangemode="tozero"),
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=10)),
        hovermode="x unified",
    )
    st.plotly_chart(event_fig, use_container_width=True, theme=None, config={"displayModeBar": False, "displaylogo": False})

    if peak_value > 400:
        st.markdown(
            f'<div class="event-banner">🔥 <b>10–14 September smoke episode:</b> the selected window peaks at '
            f'<b>{peak_value:,.0f} µg/m³</b> at <b>{peak_station}</b>, with <b>{stations_over}</b> operational stations exceeding '
            f'{THRESHOLD:.0f} µg/m³ at least once.</div>',
            unsafe_allow_html=True,
        )

st.markdown('</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="footnote">AirEquity NSW · academic decision-support prototype · data from NSW Air Quality observations · not a replacement for official health advisories</div>',
    unsafe_allow_html=True,
)
