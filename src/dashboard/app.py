"""AirEquity NSW — 24-hour PM2.5 advisory dashboard.

Run from the repository root:
    streamlit run src/dashboard/app.py

Tabs
    Forecast          the next 24 hours for every station
    Station           one station: recent readings and its forecast
    Network history   2023-2024 exposure patterns and the smoke event explorer
    Method            how the forecast is made, and its limitations
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# Paths and constants
# -----------------------------------------------------------------------------
THRESHOLD = 25.0
OBS_PATH = Path("data/processed/observations")
FEATURES_PATH = Path("data/processed/features")
SITES_PATH = Path("data/raw/sites.json")
FORECAST_PATH = Path("data/processed/latest_forecast.parquet")
FORECAST_META_PATH = Path("data/processed/latest_forecast_meta.json")
RECENT_OBS_PATH = Path("data/processed/latest_observations.parquet")

DEFAULT_STATION_START = pd.Timestamp("2023-09-01")
DEFAULT_STATION_END = pd.Timestamp("2023-09-30")
DEFAULT_EVENT_START = pd.Timestamp("2023-09-10")
DEFAULT_EVENT_END = pd.Timestamp("2023-09-14")

DATA_RELEASE_URL = (
    "https://github.com/ahriyad1/airequity-nsw/releases/"
    "download/v0.1-data/observations.zip"
)
SUPPORTED_YEARS = {"year=2023", "year=2024"}

# Leave-one-station-out validation results (Assessment 2)
PRECISION_PENALTY = 0.143
PRECISION_CI = (0.050, 0.238)

# -----------------------------------------------------------------------------
# Visual tokens
# -----------------------------------------------------------------------------
CANVAS = "#0B1120"
PANEL = "#121A2B"
RAISED = "#172033"
INK = "#E8ECF4"
MUTED = "#94A3B8"
LINE = "rgba(255,255,255,0.08)"
SOFT = "rgba(255,255,255,0.06)"
ACCENT = "#4F8CFF"
VIOLET = "#A78BFA"
AMBER = "#FBBF24"
TEAL = "#2DD4BF"
GREEN = "#34D399"
ORANGE = "#FB923C"
DANGER = "#F43F5E"
FONT = "Geist, -apple-system, BlinkMacSystemFont, 'SF Pro Text', system-ui, sans-serif"

# Chance of exceeding 25 ug/m3. Fixed scale, so a clean day stays green.
PROB_MAX = 0.30
RISK_SCALE = [
    [0.000, "#15303A"],
    [0.060, "#17564B"],
    [0.130, "#1F9D72"],
    [0.217, "#E7C94A"],   # ~6.5%, advisory level with own sensor
    [0.300, "#F59E0B"],   # ~9%, advisory level without
    [0.550, "#F4623A"],
    [1.000, "#E11D48"],
]
SERIES = ["#4F8CFF", "#FBBF24", "#34D399", "#A78BFA", "#F472B6",
          "#22D3EE", "#FB923C", "#E8ECF4"]
PLOT_CONFIG = {"displayModeBar": False, "displaylogo": False}


def risk_colour(p: float) -> str:
    if p < 0.03:
        return GREEN
    if p < 0.065:
        return "#FACC15"
    if p < 0.09:
        return "#F59E0B"
    if p < 0.165:
        return "#F97316"
    return DANGER


st.set_page_config(page_title="AirEquity NSW", page_icon="🌬️",
                   layout="wide", initial_sidebar_state="collapsed")

st.markdown(
    f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700&display=swap');

      html, body, .stApp, [class*="css"], button, input, select, textarea {{
        font-family: {FONT}; -webkit-font-smoothing: antialiased;
      }}
      .stApp {{
        background:
          radial-gradient(1100px 520px at 12% -10%, rgba(79,140,255,.14), transparent 60%),
          radial-gradient(900px 480px at 100% 0%, rgba(167,139,250,.11), transparent 60%),
          {CANVAS};
        color: {INK};
      }}
      header[data-testid="stHeader"] {{ background: transparent; }}
      footer {{ visibility: hidden; }}
      .block-container {{ max-width: 1200px; padding-top: 1.1rem; padding-bottom: 3rem; }}
      label, [data-testid="stWidgetLabel"] p {{ color: {INK} !important; font-weight: 500; }}

      /* ---- top bar ---- */
      .topbar {{ display:flex; align-items:center; justify-content:space-between; padding:.3rem 0 1.1rem; }}
      .brand {{ display:flex; align-items:center; gap:.65rem; font-size:1.15rem;
                font-weight:650; letter-spacing:-.015em; color:{INK}; }}
      .brand-mark {{ width:32px; height:32px; border-radius:9px;
                     background:linear-gradient(140deg,#8B5CF6 0%,#3B82F6 55%,#22D3EE 100%);
                     display:grid; place-items:center; box-shadow:0 4px 16px rgba(79,140,255,.45); }}
      .brand-sub {{ color:{MUTED}; font-weight:450; font-size:.95rem; margin-left:.2rem; }}
      .live {{ display:flex; align-items:center; gap:.5rem; color:{MUTED}; font-size:.88rem; }}
      .live-dot {{ width:8px; height:8px; border-radius:50%; background:{GREEN};
                   box-shadow:0 0 0 4px rgba(52,211,153,.18), 0 0 12px rgba(52,211,153,.6); }}

      /* ---- segmented tabs ---- */
      .stTabs [data-baseweb="tab-list"] {{
        background:rgba(255,255,255,.05); border:1px solid rgba(255,255,255,.06);
        padding:4px; border-radius:12px; gap:2px; width:fit-content;
      }}
      .stTabs [data-baseweb="tab"] {{
        height:34px; padding:0 18px; border-radius:9px; background:transparent;
        color:{MUTED}; font-weight:500; font-size:.93rem;
      }}
      .stTabs [aria-selected="true"] {{
        background:linear-gradient(180deg,#26324A,#1C2638); color:#FFFFFF; font-weight:600;
        box-shadow:0 1px 0 rgba(255,255,255,.08) inset, 0 4px 12px rgba(0,0,0,.35);
      }}
      .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display:none; }}

      /* ---- panels ---- */
      div[data-testid="stVerticalBlockBorderWrapper"] {{
        background:{PANEL}; border-radius:18px !important;
        border:1px solid {LINE} !important;
        box-shadow:0 1px 0 rgba(255,255,255,.04) inset, 0 12px 32px rgba(0,0,0,.35);
      }}
      .panel-title {{ font-size:1.04rem; font-weight:650; letter-spacing:-.01em;
                      margin:.25rem 0 .1rem; color:{INK}; }}
      .panel-sub {{ color:{MUTED}; font-size:.87rem; margin-bottom:.35rem; }}

      /* ---- hero ---- */
      .hero {{
        background:
          radial-gradient(520px 240px at 0% 0%, rgba(79,140,255,.20), transparent 70%),
          radial-gradient(460px 220px at 100% 0%, rgba(167,139,250,.16), transparent 70%),
          {PANEL};
        border:1px solid {LINE}; border-radius:22px;
        padding:1.6rem 1.8rem 1.55rem; margin:1rem 0 1.1rem;
        box-shadow:0 1px 0 rgba(255,255,255,.05) inset, 0 18px 44px rgba(0,0,0,.4);
      }}
      .pill {{ display:inline-flex; align-items:center; gap:.45rem; padding:.3rem .75rem;
               border-radius:999px; font-size:.82rem; font-weight:600; }}
      .pill .dot {{ width:7px; height:7px; border-radius:50%; background:currentColor;
                    box-shadow:0 0 10px currentColor; }}
      .pill.clear {{ background:rgba(52,211,153,.14); color:{GREEN}; }}
      .pill.alert {{ background:rgba(251,146,60,.16); color:{ORANGE}; }}
      .pill.neutral {{ background:rgba(148,163,184,.14); color:#CBD5E1; }}
      .hero-title {{ font-size:clamp(1.8rem,3.1vw,2.65rem); font-weight:700; color:#FFFFFF;
                     letter-spacing:-.03em; line-height:1.08; margin:.85rem 0 .4rem; }}
      .hero-sub {{ color:{MUTED}; font-size:1.02rem; margin:0 0 1.35rem; }}
      .stats {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:.75rem; }}
      .stat {{ border-radius:14px; padding:.9rem 1rem .85rem;
               border:1px solid rgba(255,255,255,.06); }}
      .stat-label {{ color:#CBD5E1; font-size:.8rem; font-weight:500;
                     display:flex; align-items:center; gap:.45rem; }}
      .chip {{ width:8px; height:8px; border-radius:50%; flex:none; }}
      .stat-value {{ font-size:1.75rem; font-weight:650; letter-spacing:-.025em; color:#FFFFFF;
                     margin-top:.3rem; font-variant-numeric:tabular-nums; line-height:1.1; }}
      .stat-note {{ color:{MUTED}; font-size:.8rem; margin-top:.25rem; }}
      @media (max-width: 820px) {{ .stats {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}

      /* ---- station ranking ---- */
      .rank {{ margin-top:.3rem; max-height:392px; overflow-y:auto; padding-right:.35rem; }}
      .rank-row {{ display:grid; grid-template-columns:minmax(0,1fr) 96px 52px;
                   align-items:center; gap:.8rem; padding:.58rem 0;
                   border-bottom:1px solid {SOFT}; }}
      .rank-row:last-child {{ border-bottom:none; }}
      .rank-name {{ font-weight:550; font-size:.93rem; color:{INK}; }}
      .rank-region {{ color:{MUTED}; font-size:.77rem; }}
      .bar {{ height:6px; border-radius:99px; background:rgba(255,255,255,.08); overflow:hidden; }}
      .bar span {{ display:block; height:100%; border-radius:99px; }}
      .rank-val {{ text-align:right; font-variant-numeric:tabular-nums; font-weight:600;
                   font-size:.92rem; color:#FFFFFF; }}

      /* ---- big figures ---- */
      .figure {{ margin:.4rem 0 1rem; }}
      .figure-value {{ font-size:2.4rem; font-weight:700; letter-spacing:-.035em;
                       line-height:1; font-variant-numeric:tabular-nums; }}
      .figure-label {{ color:{INK}; font-weight:550; font-size:.95rem; margin-top:.35rem; }}
      .figure-note {{ color:{MUTED}; font-size:.84rem; margin-top:.15rem; }}

      /* ---- method ---- */
      .steps {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:.75rem;
                margin:.2rem 0 1.2rem; }}
      .step {{ background:{PANEL}; border:1px solid {LINE}; border-radius:16px; padding:1rem;
               box-shadow:0 10px 26px rgba(0,0,0,.3); }}
      .step-n {{ width:28px; height:28px; border-radius:8px; font-weight:650; font-size:.85rem;
                 display:grid; place-items:center; }}
      .step-t {{ font-weight:600; font-size:.95rem; margin:.65rem 0 .25rem; color:#FFFFFF; }}
      .step-d {{ color:{MUTED}; font-size:.85rem; line-height:1.45; }}
      @media (max-width: 900px) {{ .steps {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} }}
      table.results {{ width:100%; border-collapse:collapse; font-size:.93rem; border:none !important; }}
      table.results th, table.results td {{ border:none !important; background:transparent !important; }}
      table.results th {{ text-align:left; color:{MUTED}; font-weight:500; font-size:.82rem;
                          padding:.5rem .6rem; border-bottom:1px solid {LINE} !important; }}
      table.results td {{ padding:.65rem .6rem; border-bottom:1px solid {SOFT} !important;
                          font-variant-numeric:tabular-nums; color:{INK}; }}
      table.results tr:last-child td {{ border-bottom:none !important; }}
      table.results td.num {{ text-align:right; font-weight:600; color:#FFFFFF; }}
      table.results th.num {{ text-align:right; }}
      .limits {{ color:{INK}; font-size:.93rem; line-height:1.55; padding-left:1.1rem; margin:.3rem 0; }}
      .limits li {{ margin-bottom:.35rem; }}
      .limits li::marker {{ color:{ACCENT}; }}

      .empty {{ background:{PANEL}; border:1px dashed rgba(255,255,255,.18); border-radius:18px;
                padding:2rem; margin-top:1rem; color:{MUTED}; }}
      .empty b {{ color:#FFFFFF; }}
      .foot {{ color:{MUTED}; font-size:.8rem; margin-top:2.4rem; text-align:center; }}
      code {{ background:rgba(255,255,255,.08); color:{INK}; padding:.1rem .35rem; border-radius:5px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# First-run data bootstrap for Streamlit Community Cloud
# -----------------------------------------------------------------------------
def ensure_dashboard_data() -> None:
    if OBS_PATH.exists() and FEATURES_PATH.exists():
        return
    processed_root = Path("data") / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)

    if not OBS_PATH.exists():
        with st.spinner("Preparing 2023–2024 data for the first run…"):
            archive = Path("data") / "observations-release.zip"
            urllib.request.urlretrieve(DATA_RELEASE_URL, archive)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(processed_root)
            archive.unlink(missing_ok=True)
            if OBS_PATH.exists():
                for child in OBS_PATH.iterdir():
                    if (child.is_dir() and child.name.startswith("year=")
                            and child.name not in SUPPORTED_YEARS):
                        shutil.rmtree(child)

    if not FEATURES_PATH.exists():
        with st.spinner("Building threshold labels for the first run…"):
            result = subprocess.run(
                [sys.executable, "-m", "src.features.build_features"],
                check=False, capture_output=True, text=True)
            if result.returncode != 0:
                details = (result.stderr or result.stdout or "unknown error").strip()
                raise RuntimeError("Feature generation failed: " + details[-1600:])


ensure_dashboard_data()


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading PM2.5 observations…")
def load_observations() -> pd.DataFrame:
    df = pd.read_parquet(OBS_PATH)
    df = df[(df["parameter"] == "PM2.5") & (df["frequency"] == "Hourly average")].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce").clip(lower=0)
    return df.dropna(subset=["site_id", "site_name", "timestamp", "value"])


@st.cache_data(show_spinner="Loading threshold labels…")
def load_features() -> pd.DataFrame:
    df = pd.read_parquet(FEATURES_PATH, columns=["site_id", "site_name", "region", "label"])
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    return df.dropna(subset=["site_id", "site_name", "region", "label"])


@st.cache_data(show_spinner=False)
def load_sites() -> pd.DataFrame:
    if SITES_PATH.exists():
        raw = json.loads(SITES_PATH.read_text(encoding="utf-8"))
        sites = pd.DataFrame([{
            "site_id": s.get("Site_Id", s.get("site_id")),
            "latitude": s.get("Latitude", s.get("latitude")),
            "longitude": s.get("Longitude", s.get("longitude")),
        } for s in raw])
    else:
        sites = pd.read_parquet(OBS_PATH, columns=["site_id", "latitude", "longitude"])
    sites["latitude"] = pd.to_numeric(sites["latitude"], errors="coerce")
    sites["longitude"] = pd.to_numeric(sites["longitude"], errors="coerce")
    return sites.dropna().drop_duplicates("site_id")


@st.cache_data(ttl=600, show_spinner=False)
def load_forecast():
    if not FORECAST_PATH.exists():
        return None, {}
    fc = pd.read_parquet(FORECAST_PATH)
    fc["timestamp"] = pd.to_datetime(fc["timestamp"])
    meta = {}
    if FORECAST_META_PATH.exists():
        meta = json.loads(FORECAST_META_PATH.read_text(encoding="utf-8"))
    return fc, meta


@st.cache_data(ttl=600, show_spinner=False)
def load_recent_readings():
    if not RECENT_OBS_PATH.exists():
        return None
    df = pd.read_parquet(RECENT_OBS_PATH).rename(columns={"PM2.5": "value"})
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce").clip(lower=0)
    return df[["site_name", "timestamp", "value"]].dropna()


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def nice(name) -> str:
    return str(name).title().replace(" And ", " and ")


def hour_label(ts) -> str:
    ts = pd.Timestamp(ts)
    return f"{int(ts.strftime('%I'))} {ts.strftime('%p').lower()}"


def day_hour(ts) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.day} {ts.strftime('%b')}, {hour_label(ts)}"


def pct(p, digits=1) -> str:
    return f"{p * 100:.{digits}f}%"


def style_fig(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=4, r=4, t=4, b=4),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, color=INK, size=13),
        hoverlabel=dict(font_family=FONT, bgcolor=RAISED, bordercolor="#334155",
                        font_color=INK),
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=12, color=MUTED)),
    )
    fig.update_xaxes(gridcolor=SOFT, zeroline=False, linecolor=LINE,
                     tickfont=dict(color=MUTED), automargin=True)
    fig.update_yaxes(gridcolor=SOFT, zeroline=False, linecolor=LINE,
                     tickfont=dict(color=MUTED), automargin=True)
    return fig


def show(fig: go.Figure) -> None:
    st.plotly_chart(fig, theme=None, config=PLOT_CONFIG)


def panel_head(title: str, sub: str | None = None) -> None:
    st.markdown(f'<div class="panel-title">{title}</div>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="panel-sub">{sub}</div>', unsafe_allow_html=True)


def stat(label: str, value: str, note: str = "", colour: str = ACCENT) -> str:
    return (f'<div class="stat" style="background:linear-gradient(180deg,{colour}24 0%,'
            f'rgba(255,255,255,.025) 75%)"><div class="stat-label">'
            f'<span class="chip" style="background:{colour};box-shadow:0 0 10px {colour}"></span>'
            f'{label}</div>'
            f'<div class="stat-value">{value}</div>'
            f'<div class="stat-note">{note}</div></div>')


def threshold_line(fig: go.Figure, y: float, text: str, colour: str, pos="top left"):
    fig.add_hline(y=y, line_dash="dash", line_color=colour, line_width=1.3,
                  annotation_text=text, annotation_position=pos,
                  annotation_font_color=colour, annotation_font_size=11)


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
missing = [str(p) for p in (OBS_PATH, FEATURES_PATH) if not p.exists()]
if missing:
    st.error("Historical data is missing: " + ", ".join(missing)
             + ". Run the ingestion and feature scripts, then reload.")
    st.stop()

try:
    obs = load_observations()
    feat = load_features()
    sites = load_sites()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not load historical data: {exc}")
    st.stop()

fc, meta = load_forecast()

operational_ids = set(feat["site_id"].unique())
obs = obs[obs["site_id"].isin(operational_ids)]
sites = sites[sites["site_id"].isin(operational_ids)]

station_rates = (feat.groupby(["site_id", "site_name", "region"], as_index=False)["label"]
                     .mean().rename(columns={"label": "rate"}))
station_rates["rate_pct"] = station_rates["rate"] * 100
regional = (feat.groupby("region", as_index=False)["label"].mean()
                .rename(columns={"label": "rate"}))
regional["rate_pct"] = regional["rate"] * 100
network_rate_pct = float(feat["label"].mean() * 100)

has_fc = fc is not None and not fc.empty
if has_fc:
    issued = pd.Timestamp(meta.get("issued_at", fc["issued_at"].iloc[0]))
    obs_to = pd.Timestamp(meta.get("observations_up_to", fc["observations_up_to"].iloc[0]))
    thr_mon = float(fc["threshold_monitored"].iloc[0])
    thr_unmon = float(fc["threshold_unmonitored"].iloc[0])

# -----------------------------------------------------------------------------
# Top bar
# -----------------------------------------------------------------------------
WIND_ICON = ('<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="white" '
             'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">'
             '<path d="M12.8 19.6A2 2 0 1 0 14 16H2"/><path d="M17.5 8a2.5 2.5 0 1 1 2 4H2"/>'
             '<path d="M9.8 4.4A2 2 0 1 1 11 8H2"/></svg>')
updated = f"Updated {day_hour(issued)}" if has_fc else "No forecast yet"
st.markdown(
    f'<div class="topbar"><div class="brand"><div class="brand-mark">{WIND_ICON}</div>'
    f'AirEquity<span class="brand-sub">Sydney PM2.5 forecast</span></div>'
    f'<div class="live"><span class="live-dot"></span>{updated}</div></div>',
    unsafe_allow_html=True)

tab_fc, tab_station, tab_history, tab_method = st.tabs(
    ["Forecast", "Station", "Network history", "Method"])

EMPTY_FC = ('<div class="empty"><b>No forecast has been generated yet.</b><br>'
            'Run <code>python3 src/models/predict_today.py</code> from the repository '
            'root, then reload this page.</div>')


# =============================================================================
# FORECAST
# =============================================================================
with tab_fc:
    if not has_fc:
        st.markdown(EMPTY_FC, unsafe_allow_html=True)
    else:
        alert_hours = fc.groupby("site_name")["alert_monitored"].sum()
        alerted = alert_hours[alert_hours > 0].sort_values(ascending=False).index.tolist()
        top = fc.loc[fc["prob_monitored"].idxmax()]
        n_st = fc["site_id"].nunique()
        age_h = (issued - obs_to).total_seconds() / 3600

        if alerted:
            pill = '<span class="pill alert"><span class="dot"></span>Advisory</span>'
            title = (f"Advisory for {nice(alerted[0])}" if len(alerted) == 1
                     else f"Advisory for {len(alerted)} stations")
            sub = f"Chance of PM2.5 exceeding 25 µg/m³ passes {pct(thr_mon)} in the next 24 hours."
        else:
            pill = '<span class="pill clear"><span class="dot"></span>All clear</span>'
            title = "No advisories in the next 24 hours"
            sub = f"Every station stays below the {pct(thr_mon)} advisory level."

        st.markdown(
            f'<div class="hero">{pill}<div class="hero-title">{title}</div>'
            f'<div class="hero-sub">{sub}</div><div class="stats">'
            + stat("Highest chance", pct(top["prob_monitored"]),
                   f'{nice(top["site_name"])}, {hour_label(top["timestamp"])}', ORANGE)
            + stat("Stations on advisory", f"{len(alerted)} of {n_st}",
                   "with their own sensors", VIOLET)
            + stat("Advisory level", pct(thr_mon), "lowest cost when tested", AMBER)
            + stat("Readings age", f"{age_h:.0f} h", f"up to {day_hour(obs_to)}", TEAL)
            + '</div></div>',
            unsafe_allow_html=True)

        # ---- 24-hour grid ---------------------------------------------------
        with st.container(border=True):
            head_l, head_r = st.columns([1.3, 1])
            with head_l:
                panel_head("Next 24 hours",
                           "Chance of exceeding 25 µg/m³, by station and hour.")
            with head_r:
                view = st.segmented_control(
                    "View", ["With own sensor", "Without sensor"],
                    default="With own sensor", label_visibility="collapsed")
            use_mon = view != "Without sensor"
            pcol = "prob_monitored" if use_mon else "prob_unmonitored"
            acol = "alert_monitored" if use_mon else "alert_unmonitored"

            order = fc.groupby("site_name")[pcol].max().sort_values().index.tolist()
            grid = fc.pivot_table(index="site_name", columns="timestamp",
                                  values=pcol, aggfunc="first").reindex(order)
            heat = go.Figure(go.Heatmap(
                z=grid.values, x=grid.columns, y=[nice(s) for s in grid.index],
                colorscale=RISK_SCALE, zmin=0, zmax=PROB_MAX, xgap=3, ygap=3,
                colorbar=dict(tickvals=[0, .05, .10, .20, .30],
                              ticktext=["0%", "5%", "10%", "20%", "30%+"],
                              thickness=8, len=.8, outlinewidth=0,
                              tickfont=dict(color=MUTED, size=11)),
                hovertemplate="<b>%{y}</b><br>%{x|%-I %p, %-d %b}"
                              "<br>Chance: %{z:.1%}<extra></extra>"))
            hits = fc[fc[acol] == 1]
            if not hits.empty:
                heat.add_trace(go.Scatter(
                    x=hits["timestamp"], y=[nice(s) for s in hits["site_name"]],
                    mode="markers", marker=dict(size=6, color=CANVAS),
                    hoverinfo="skip", showlegend=False))
            style_fig(heat, height=80 + 30 * len(order))
            heat.update_xaxes(tickformat="%-I %p", dtick=3 * 3600 * 1000, showgrid=False,
                              showline=False)
            heat.update_yaxes(showgrid=False, showline=False, ticks="", tickfont=dict(color=INK))
            show(heat)

        st.write("")

        # ---- map + ranking ----------------------------------------------------
        m_col, r_col = st.columns([1.45, 1], gap="medium")
        peaks = (fc.groupby(["site_id", "site_name", "region", "latitude", "longitude"],
                            as_index=False)[pcol].max()
                   .sort_values(pcol, ascending=False))

        with m_col:
            with st.container(border=True):
                panel_head("Map", "Highest chance at each station over the next 24 hours.")
                fmap = go.Figure(go.Scattermap(
                    lat=peaks["latitude"], lon=peaks["longitude"], mode="markers",
                    text=[nice(s) for s in peaks["site_name"]],
                    customdata=np.stack([peaks["region"], peaks[pcol]], axis=-1),
                    marker=dict(size=18, color=peaks[pcol], colorscale=RISK_SCALE,
                                cmin=0, cmax=PROB_MAX, opacity=.95),
                    hovertemplate="<b>%{text}</b><br>%{customdata[0]}"
                                  "<br>Highest chance %{customdata[1]:.1%}<extra></extra>"))
                style_fig(fmap, height=430)
                fmap.update_layout(map=dict(
                    style="carto-darkmatter", zoom=8.4,
                    center=dict(lat=float(peaks["latitude"].mean()),
                                lon=float(peaks["longitude"].mean()))))
                show(fmap)

        with r_col:
            with st.container(border=True):
                panel_head("Stations", "Ranked by highest chance.")
                rows = []
                for _, r in peaks.iterrows():
                    p = float(r[pcol])
                    w = min(p / PROB_MAX, 1) * 100
                    rows.append(
                        f'<div class="rank-row"><div><div class="rank-name">{nice(r["site_name"])}</div>'
                        f'<div class="rank-region">{r["region"]}</div></div>'
                        f'<div class="bar"><span style="width:{max(w, 2):.1f}%;'
                        f'background:{risk_colour(p)}"></span></div>'
                        f'<div class="rank-val">{pct(p)}</div></div>')
                st.markdown(f'<div class="rank">{"".join(rows)}</div>',
                            unsafe_allow_html=True)

        st.write("")

        # ---- sensor comparison ---------------------------------------------
        with st.container(border=True):
            panel_head("What a missing sensor changes",
                       "From our original validation, which trained on neighbouring stations over the "
                       "same days it tested. Being re-measured with a stricter test; see Method.")
            f_col, c_col = st.columns([1, 1.8], gap="large")
            with f_col:
                st.markdown(
                    f'<div class="figure"><div class="figure-value" style="color:{ORANGE}">'
                    f'−{pct(PRECISION_PENALTY)}</div>'
                    f'<div class="figure-label">Precision without a sensor, original test</div>'
                    f'<div class="figure-note">95% CI {pct(PRECISION_CI[0])} to '
                    f'{pct(PRECISION_CI[1])}. More false alarms.</div></div>'
                    f'<div class="figure"><div class="figure-value" style="color:{GREEN}">≈ 0</div>'
                    f'<div class="figure-label">Change in events caught, original test</div>'
                    f'<div class="figure-note">Recall 38% with a sensor, 37% without.</div></div>',
                    unsafe_allow_html=True)
            with c_col:
                cmp = (fc.groupby("site_name", as_index=False)
                         .agg(mon=("prob_monitored", "max"), unmon=("prob_unmonitored", "max"))
                         .sort_values("mon"))
                names = [nice(s) for s in cmp["site_name"]]
                xs, ys = [], []
                for n, a, b in zip(names, cmp["unmon"], cmp["mon"]):
                    xs += [a, b, None]
                    ys += [n, n, None]
                dumb = go.Figure()
                dumb.add_trace(go.Scatter(x=xs, y=ys, mode="lines",
                                          line=dict(color="rgba(255,255,255,0.12)", width=4),
                                          hoverinfo="skip", showlegend=False))
                dumb.add_trace(go.Scatter(
                    x=cmp["unmon"], y=names, mode="markers", name="Without sensor",
                    marker=dict(size=11, color=PANEL, line=dict(color=MUTED, width=2)),
                    hovertemplate="%{y}<br>Without sensor: %{x:.1%}<extra></extra>"))
                dumb.add_trace(go.Scatter(
                    x=cmp["mon"], y=names, mode="markers", name="With own sensor",
                    marker=dict(size=11, color=ACCENT),
                    hovertemplate="%{y}<br>With sensor: %{x:.1%}<extra></extra>"))
                style_fig(dumb, height=60 + 26 * len(names))
                dumb.update_xaxes(tickformat=".0%", rangemode="tozero")
                dumb.update_yaxes(showgrid=False, showline=False, ticks="", tickfont=dict(color=INK))
                show(dumb)


# =============================================================================
# STATION
# =============================================================================
with tab_station:
    if not has_fc:
        st.markdown(EMPTY_FC, unsafe_allow_html=True)
    else:
        ranked = (fc.groupby("site_name")["prob_monitored"].max()
                    .sort_values(ascending=False).index.tolist())
        st.write("")
        sel = st.selectbox("Station", ranked, format_func=nice,
                           help="Ordered from highest to lowest forecast chance.")
        sfc = fc[fc["site_name"] == sel].sort_values("timestamp")
        region = str(sfc["region"].iloc[0])
        peak = sfc.loc[sfc["prob_monitored"].idxmax()]
        rate_row = station_rates[station_rates["site_name"] == sel]
        rate_txt = f'{rate_row["rate_pct"].iloc[0]:.2f}%' if not rate_row.empty else "—"
        on_alert = int(sfc["alert_monitored"].sum())

        pill = ('<span class="pill alert"><span class="dot"></span>Advisory</span>'
                if on_alert else
                '<span class="pill clear"><span class="dot"></span>All clear</span>')
        st.markdown(
            f'<div class="hero">{pill} <span class="pill neutral">{region}</span>'
            f'<div class="hero-title">{nice(sel)}</div>'
            f'<div class="hero-sub">Forecast for the next 24 hours, with recent readings.</div>'
            f'<div class="stats">'
            + stat("Highest chance", pct(peak["prob_monitored"]), f'around {hour_label(peak["timestamp"])}', ORANGE)
            + stat("Without its sensor", pct(sfc["prob_unmonitored"].max()), "highest chance", VIOLET)
            + stat("Hours on advisory", str(on_alert), "in the next 24", AMBER)
            + stat("Exceeded 2023–24", rate_txt, f"network {network_rate_pct:.2f}%", TEAL)
            + '</div></div>',
            unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            with st.container(border=True):
                panel_head("Last ten days", "Hourly PM2.5 readings.")
                recent = load_recent_readings()
                source = recent if recent is not None else obs[["site_name", "timestamp", "value"]]
                win = source[(source["site_name"] == sel)
                             & (source["timestamp"] > issued - pd.Timedelta(days=10))
                             & (source["timestamp"] <= issued)].sort_values("timestamp")
                if win.empty:
                    st.markdown(
                        '<div class="panel-sub">Recent readings aren\'t in this deployment. '
                        'Commit <code>data/processed/latest_observations.parquet</code> '
                        'to show them.</div>', unsafe_allow_html=True)
                else:
                    rfig = go.Figure(go.Scatter(
                        x=win["timestamp"], y=win["value"], mode="lines",
                        line=dict(color=ACCENT, width=2), fill="tozeroy",
                        fillcolor="rgba(79,140,255,.16)",
                        hovertemplate="%{x|%-d %b, %-I %p}<br>%{y:.1f} µg/m³<extra></extra>"))
                    threshold_line(rfig, THRESHOLD, "25 µg/m³", DANGER)
                    style_fig(rfig, height=320)
                    rfig.update_yaxes(title="µg/m³", rangemode="tozero")
                    rfig.update_layout(showlegend=False)
                    show(rfig)

        with c2:
            with st.container(border=True):
                panel_head("Next 24 hours", "Chance of exceeding 25 µg/m³.")
                ffig = go.Figure()
                ffig.add_trace(go.Scatter(
                    x=sfc["timestamp"], y=sfc["prob_unmonitored"], mode="lines",
                    name="Without sensor", line=dict(color=VIOLET, width=2, dash="dot"),
                    hovertemplate="%{x|%-I %p}<br>Without sensor: %{y:.1%}<extra></extra>"))
                ffig.add_trace(go.Scatter(
                    x=sfc["timestamp"], y=sfc["prob_monitored"], mode="lines",
                    name="With own sensor", line=dict(color=ACCENT, width=2.6),
                    hovertemplate="%{x|%-I %p}<br>With sensor: %{y:.1%}<extra></extra>"))
                threshold_line(ffig, thr_mon, f"Advisory {pct(thr_mon)}", AMBER)
                style_fig(ffig, height=320)
                ymax = max(float(sfc[["prob_monitored", "prob_unmonitored"]].max().max()),
                           thr_unmon) * 1.3
                ffig.update_yaxes(tickformat=".0%", range=[0, ymax])
                ffig.update_xaxes(tickformat="%-I %p")
                show(ffig)


# =============================================================================
# NETWORK HISTORY
# =============================================================================
with tab_history:
    hist_obs = obs[obs["timestamp"] < pd.Timestamp("2025-01-01")]
    min_date = hist_obs["timestamp"].min().date()
    max_date = hist_obs["timestamp"].max().date()

    nw = regional.loc[regional["region"] == "Sydney North-west", "rate_pct"]
    east = regional.loc[regional["region"] == "Sydney East", "rate_pct"]
    nw_v = float(nw.iloc[0]) if not nw.empty else np.nan
    east_v = float(east.iloc[0]) if not east.empty else np.nan
    ratio = nw_v / east_v if east_v and not np.isnan(east_v) else np.nan

    st.markdown(
        '<div class="hero"><span class="pill neutral">2023–2024</span>'
        '<div class="hero-title">Network history</div>'
        '<div class="hero-sub">How often PM2.5 exceeded 25 µg/m³, and where.</div>'
        '<div class="stats">'
        + stat("Network", f"{network_rate_pct:.2f}%", "of station-hours", ACCENT)
        + stat("North-west", f"{nw_v:.2f}%", "highest region", ORANGE)
        + stat("East", f"{east_v:.2f}%", "lowest region", TEAL)
        + stat("North-west vs east", f"{ratio:.1f}×", "with fewer stations", VIOLET)
        + '</div></div>',
        unsafe_allow_html=True)

    with st.container(border=True):
        panel_head("Station readings")
        hc1, hc2 = st.columns([1, 1.4], gap="medium")
        with hc1:
            h_station = st.selectbox("Station", sorted(hist_obs["site_name"].dropna().unique()),
                                     format_func=nice, key="h_station")
        with hc2:
            rng = st.date_input(
                "Dates", value=(max(min_date, DEFAULT_STATION_START.date()),
                                min(max_date, DEFAULT_STATION_END.date())),
                min_value=min_date, max_value=max_date, key="h_range")
        s0, s1 = (rng if isinstance(rng, (tuple, list)) and len(rng) == 2 else (rng, rng))
        sdf = hist_obs[(hist_obs["site_name"] == h_station)
                       & (hist_obs["timestamp"] >= pd.Timestamp(s0))
                       & (hist_obs["timestamp"] < pd.Timestamp(s1) + pd.Timedelta(days=1))
                       ].sort_values("timestamp")
        tfig = go.Figure(go.Scatter(
            x=sdf["timestamp"], y=sdf["value"], mode="lines",
            line=dict(color=ACCENT, width=1.8), fill="tozeroy",
            fillcolor="rgba(79,140,255,.14)",
            hovertemplate="%{x|%-d %b %Y, %-I %p}<br>%{y:.1f} µg/m³<extra></extra>"))
        threshold_line(tfig, THRESHOLD, "25 µg/m³", DANGER)
        style_fig(tfig, height=320)
        tfig.update_yaxes(title="µg/m³", rangemode="tozero")
        tfig.update_layout(showlegend=False)
        show(tfig)

    st.write("")
    m1, m2 = st.columns([1.35, 1], gap="medium")
    rate_max = float(station_rates["rate_pct"].max())
    with m1:
        with st.container(border=True):
            panel_head("By station", "Share of hours above 25 µg/m³.")
            mapped = station_rates.merge(sites, on="site_id", how="inner")
            hmap = go.Figure(go.Scattermap(
                lat=mapped["latitude"], lon=mapped["longitude"], mode="markers",
                text=[nice(s) for s in mapped["site_name"]],
                customdata=np.stack([mapped["region"], mapped["rate_pct"]], axis=-1),
                marker=dict(size=10 + mapped["rate_pct"] * 5, color=mapped["rate_pct"],
                            colorscale=RISK_SCALE[1:], cmin=0, cmax=rate_max, opacity=.95),
                hovertemplate="<b>%{text}</b><br>%{customdata[0]}"
                              "<br>%{customdata[1]:.2f}% of hours<extra></extra>"))
            style_fig(hmap, height=400)
            hmap.update_layout(map=dict(
                style="carto-darkmatter", zoom=8.2,
                center=dict(lat=float(mapped["latitude"].mean()),
                            lon=float(mapped["longitude"].mean()))))
            show(hmap)

    with m2:
        with st.container(border=True):
            panel_head("By region", "Share of hours above 25 µg/m³.")
            reg = regional.sort_values("rate_pct")
            bar = go.Figure(go.Bar(
                y=reg["region"], x=reg["rate_pct"], orientation="h",
                marker=dict(color=[risk_colour(v / 100 * 5) for v in reg["rate_pct"]],
                            line=dict(width=0), cornerradius=6),
                text=[f"{v:.2f}%" for v in reg["rate_pct"]], textposition="outside",
                cliponaxis=False,
                hovertemplate="%{y}<br>%{x:.2f}% of hours<extra></extra>"))
            style_fig(bar, height=400)
            bar.update_xaxes(rangemode="tozero", showgrid=False, showticklabels=False)
            bar.update_yaxes(showgrid=False, showline=False, ticks="", tickfont=dict(color=INK))
            bar.update_layout(showlegend=False, bargap=.45, margin=dict(l=4, r=60, t=4, b=4))
            show(bar)

    st.write("")
    with st.container(border=True):
        panel_head("Smoke event explorer",
                   "Default window: hazard reduction burn smoke, 10–14 September 2023.")
        e_a, e_b = st.columns([1, 1.6], gap="medium")
        with e_a:
            ev = st.date_input(
                "Window", value=(max(min_date, DEFAULT_EVENT_START.date()),
                                 min(max_date, DEFAULT_EVENT_END.date())),
                min_value=min_date, max_value=max_date, key="ev_range")
        e0, e1 = (ev if isinstance(ev, (tuple, list)) and len(ev) == 2 else (ev, ev))
        edf = hist_obs[(hist_obs["timestamp"] >= pd.Timestamp(e0))
                       & (hist_obs["timestamp"] < pd.Timestamp(e1) + pd.Timedelta(days=1))]
        if edf.empty:
            st.markdown('<div class="panel-sub">No readings in this window. Choose dates '
                        'between January 2023 and December 2024.</div>',
                        unsafe_allow_html=True)
        else:
            peaks_ev = edf.groupby("site_name")["value"].max().sort_values(ascending=False)
            with e_b:
                chosen = st.multiselect(
                    "Stations", options=peaks_ev.index.tolist(),
                    default=peaks_ev.head(min(6, len(peaks_ev))).index.tolist(),
                    format_func=nice, max_selections=8)
            pk = edf.loc[edf["value"].idxmax()]
            n_over = int(edf.loc[edf["value"] > THRESHOLD, "site_id"].nunique())
            st.markdown(
                '<div class="stats" style="margin:.4rem 0 .6rem">'
                + stat("Peak", f'{pk["value"]:,.0f}', f'µg/m³ at {nice(pk["site_name"])}', DANGER)
                + stat("Stations over 25", f"{n_over} of {len(operational_ids)}", "at least once", ORANGE)
                + stat("Window", f"{(pd.Timestamp(e1) - pd.Timestamp(e0)).days + 1} days",
                       f"{pd.Timestamp(e0):%-d %b} to {pd.Timestamp(e1):%-d %b %Y}", ACCENT)
                + stat("Peak time", hour_label(pk["timestamp"]), f'{pd.Timestamp(pk["timestamp"]):%-d %B}', VIOLET)
                + '</div>', unsafe_allow_html=True)
            efig = go.Figure()
            for i, name in enumerate(chosen):
                g = edf[edf["site_name"] == name].sort_values("timestamp")
                efig.add_trace(go.Scatter(
                    x=g["timestamp"], y=g["value"], mode="lines", name=nice(name),
                    line=dict(color=SERIES[i % len(SERIES)], width=2),
                    hovertemplate=f"<b>{nice(name)}</b><br>%{{x|%-d %b, %-I %p}}"
                                  f"<br>%{{y:.1f}} µg/m³<extra></extra>"))
            threshold_line(efig, THRESHOLD, "25 µg/m³", DANGER)
            style_fig(efig, height=380)
            efig.update_yaxes(title="µg/m³", rangemode="tozero")
            efig.update_layout(hovermode="x unified")
            show(efig)


# =============================================================================
# METHOD
# =============================================================================
with tab_method:
    st.markdown(
        '<div class="hero"><span class="pill neutral">How it works</span>'
        '<div class="hero-title">From readings to an advisory</div>'
        '<div class="hero-sub">Forecasts for 18 Sydney stations, tested two ways. '
        'The stricter test, on days the model has never seen, is the one to trust.</div></div>',
        unsafe_allow_html=True)

    steps = [
        ("Fetch", "Hourly readings from the NSW Air Quality API, recent weather from Open-Meteo."),
        ("Describe neighbours", "What nearby stations read now, weighted by distance and wind."),
        ("Estimate chance", "Two models trained on 2023–2024: with and without the station's own sensor."),
        ("Decide", "Advisory when the chance passes the level with the lowest cost in testing."),
        ("Publish", "Computed in a batch job and written to a small file this page reads."),
    ]
    st.markdown(
        '<div class="steps">' + "".join(
            f'<div class="step"><div class="step-n" style="background:{c}26;color:{c}">{i}</div>'
            f'<div class="step-t">{t}</div>'
            f'<div class="step-d">{d}</div></div>' for i, ((t, d), c) in enumerate(zip(steps, [ACCENT, VIOLET, TEAL, AMBER, GREEN]), 1))
        + '</div>', unsafe_allow_html=True)

    a, b = st.columns([1.35, 1], gap="medium")
    with a:
        with st.container(border=True):
            panel_head("How well it works",
                       "Averages across 18 stations, each held out in turn. Original test: trained "
                       "on the other stations over the same days. Stricter test: trained on the "
                       "other stations in 2023, tested on 2024.")
            st.markdown(
                '<table class="results">'
                '<tr><th></th><th colspan="2" style="text-align:center">Original test</th>'
                '<th colspan="2" style="text-align:center">Stricter test</th></tr>'
                '<tr><th>Forecast</th><th class="num">Caught</th><th class="num">Right</th>'
                '<th class="num">Caught</th><th class="num">Right</th></tr>'
                '<tr><td>With own sensor</td><td class="num">38%</td><td class="num">36%</td>'
                '<td class="num">3%</td><td class="num">5%</td></tr>'
                '<tr><td>Without own sensor</td><td class="num">37%</td><td class="num">31%</td>'
                '<td class="num">3%</td><td class="num">3%</td></tr>'
                '<tr><td>"Tomorrow like today"</td><td class="num">10%</td><td class="num">10%</td>'
                '<td class="num">10%</td><td class="num">10%</td></tr>'
                '</table>'
                '<div class="panel-sub" style="margin-top:.8rem">Smoke events hit the whole basin '
                'at once, so the original test could learn each day\'s outcome from the neighbouring '
                'stations, which a real forecast cannot. On unseen days the model does not yet beat '
                '"tomorrow like today". Improving it is the main work before the final report.</div>',
                unsafe_allow_html=True)
    with b:
        with st.container(border=True):
            panel_head("Limitations")
            st.markdown(
                '<ul class="limits">'
                '<li>On days it has not seen, the model does not yet beat a forecast of '
                '"tomorrow like today". The original validation figures were optimistic.</li>'
                '<li>Each forecast uses conditions 24 hours before the hour it predicts, '
                'as in training.</li>'
                '<li>Station readings arrive about 40 hours late, so neighbour and station '
                'inputs are older than in testing, which lowers accuracy further.</li>'
                '<li>Most warnings would be false alarms. A prototype, not an alert service.</li>'
                '</ul>', unsafe_allow_html=True)

st.markdown(
    '<div class="foot">AirEquity NSW is an academic prototype. It does not replace '
    'official NSW air quality advisories.</div>', unsafe_allow_html=True)