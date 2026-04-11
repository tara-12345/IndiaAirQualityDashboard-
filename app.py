"""
Air Quality Dashboard
Data from OpenAQ, stored in a local DuckDB database.
"""

from __future__ import annotations
import os
from pathlib import Path

import duckdb
import pandas as pd
import plotly.express as px
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
from sklearn.metrics import r2_score, mean_absolute_error

# Paths

REPO_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.environ.get("AIR_QUALITY_DB", str(Path(__file__).resolve().parent / "air_quality_full.db")))

# Pink theme for aestetics. 
st.set_page_config(layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
      .stApp                          { background-color: #fff6f9; color: #2b2b2b; }
      section[data-testid="stSidebar"]{ background-color: #fde7ef; border-right: 1px solid #f3c6d6; }
      h1, h2, h3                      { color: #2b2b2b; }
      .stCaption                      { color: #6b5b66; }
      div[data-testid="metric-container"] {
        background-color: #ffffff;
        border-left: 5px solid #f3a6c1;
        border-radius: 10px;
        padding: 10px;
      }
      button[data-baseweb="tab"]                        { font-weight: 600; color: #6b5b66; }
      button[data-baseweb="tab"][aria-selected="true"]  { color: #2b2b2b; border-bottom: 2px solid #f3a6c1; }
      .stDownloadButton button, button[kind="primary"]  {
        background-color: #f3a6c1; color: white;
        border-radius: 8px; border: none; font-weight: 600;
      }
      .stDownloadButton button:hover, button[kind="primary"]:hover { background-color: #ec8fb0; }
      hr { border-top: 1px solid #f3c6d6; }
    </style>
""", unsafe_allow_html=True)



# Database connections

def connect_ro() -> duckdb.DuckDBPyConnection:
    #Open a read-only DuckDB connection.
    temp_dir = REPO_ROOT / ".duckdb_temp"
    temp_dir.mkdir(exist_ok=True)

    con = duckdb.connect(str(DB_PATH), read_only=True, config={"access_mode": "READ_ONLY"})
    con.execute(f"PRAGMA temp_directory='{temp_dir.as_posix()}';")
    con.execute("PRAGMA max_temp_directory_size='50GiB';")
    con.execute("PRAGMA memory_limit='4GiB';")
    return con


# Chart and user interface 
def style_fig(fig, height=420):
    #Apply consistent layout (transparent background, tight margins)
    fig.update_layout(
        height=height,
        margin=dict(l=30, r=20, t=60, b=35),
        title_x=0.02,
        hovermode="x unified",
        legend_title_text="",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#2b2b2b"),
    )
    # Don't draw lines in genuine data gaps
    for trace in fig.data:
        if getattr(trace, "type", None) in {"scatter", "scattergl"}:
            trace.update(connectgaps=False)
    return fig


def download_csv_button(df, label, filename, key):
    # a Streamlit download button for a DataFrame as CSV
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv_bytes, file_name=filename, mime="text/csv", key=key)


def tab_intro(title, body): #shows a small explainer heading + divider at the bottom of a tab.
    st.markdown(f"### {title}")
    st.markdown(body)
    st.divider()


# Time-series preparation 
def prep_daily_series(df, start_ts, end_ts, smooth, value_col="value"):
    #Reindex to a full daily calendar (so gaps show as NaN, not connected lines), then optionally apply a rolling mean.
    out = df.copy()
    out["day"]      = pd.to_datetime(out["day"],      errors="coerce")
    out[value_col]  = pd.to_numeric(out[value_col],   errors="coerce")
    out             = out.sort_values("day")

    # Build a complete daily date spine
    full_days = pd.date_range(start=start_ts.normalize(), end=end_ts.normalize(), freq="D")
    out = (
        out.set_index("day")[[value_col]]
           .reindex(full_days)
           .rename_axis("day")
           .reset_index()
    )

    # Smooth if requested
    if smooth == "7-day rolling mean":
        out["value_plot"] = out[value_col].rolling(window=7,  min_periods=3).mean()
        suffix = " (7-day mean)"
    elif smooth == "30-day rolling mean":
        out["value_plot"] = out[value_col].rolling(window=30, min_periods=7).mean()
        suffix = " (30-day mean)"
    else:
        out["value_plot"] = out[value_col]
        suffix = ""

    out.attrs["suffix"] = suffix
    return out


# WHO guideline banding

# 2021 WHO annual guideline values (µg/m³, except CO in mg/m³)
WHO_GUIDELINES = {
    "pm25": {"aqg": 15.0,  "unit": "µg/m³"},
    "pm10": {"aqg": 45.0,  "unit": "µg/m³"},
    "no2":  {"aqg": 25.0,  "unit": "µg/m³"},
    "o3":   {"aqg": 100.0, "unit": "µg/m³"},
    "so2":  {"aqg": 40.0,  "unit": "µg/m³"},
    "co":   {"aqg": 4.0,   "unit": "mg/m³"},
}

def add_who_band(df, parameter):
    
    # Classify each row into a WHO exceedance band (1 = within guideline → 5 = extremely high).

    out = df.copy()
    out["value_num"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["value_num"])

    p = str(parameter).lower().strip()

    if p not in WHO_GUIDELINES:
        out["who_band"] = "WHO bands unavailable"
        return out

    aqg  = WHO_GUIDELINES[p]["aqg"]
    unit = WHO_GUIDELINES[p]["unit"]

    # CO is sometimes stored in µg/m³; convert to mg/m³ before comparing
    if p == "co":
        units_str = str(out["units"].dropna().iloc[0]).lower() if out["units"].notna().any() else ""
        if "µg" in units_str or "ug" in units_str:
            out["value_for_who"] = out["value_num"] / 1000.0
        else:
            out["value_for_who"] = out["value_num"]
    else:
        out["value_for_who"] = out["value_num"]

    # Band thresholds at 1×, 2×, 4×, 8× guideline
    t1, t2, t3, t4 = aqg, 2*aqg, 4*aqg, 8*aqg

    bins   = [-float("inf"), t1, t2, t3, t4, float("inf")]
    labels = [
        f"1 Within guideline (≤{t1:g} {unit})",
        f"2 Above guideline ({t1:g}–{t2:g} {unit})",
        f"3 High ({t2:g}–{t3:g} {unit})",
        f"4 Very high ({t3:g}–{t4:g} {unit})",
        f"5 Extremely high (>{t4:g} {unit})",
    ]

    out["who_band"] = pd.cut(out["value_for_who"], bins=bins, labels=labels, include_lowest=True)
    out["who_band"] = out["who_band"].astype(str)   # Plotly needs plain strings
    return out


# WHO colour palette keyed on band number prefix
WHO_PALETTE = {
    "1": "#2E8B57",   # green
    "2": "#BFD833",   # yellow-green
    "3": "#FDB863",   # amber
    "4": "#F46D43",   # orange-red
    "5": "#D73027",   # red
    "WHO bands unavailable": "#999999",
}


# Correlation helper text 

def correlation_interpretation(x, y):
    a, b = sorted([str(x).lower(), str(y).lower()])

    explanations = {
        ("no2", "o3"):
            "O₃ and NO₂ can show an inverse relationship because NO can titrate ozone (NO + O₃ → NO₂ + O₂), "
            "especially near traffic. In sunny conditions, photochemistry can increase O₃ downwind.",
        ("pm10", "pm25"):
            "PM₂.₅ and PM₁₀ may correlate because they share sources (dust, combustion, resuspension). "
            "Differences can indicate shifts between coarse dust and fine combustion-related particles.",
        ("no2", "pm25"):
            "PM₂.₅ and NO₂ may correlate due to shared combustion sources (traffic, industry) and stagnant weather.",
        ("o3", "temperature"):
            "Higher temperatures often support ozone formation via faster photochemistry and stronger sunlight.",
        ("pm25", "relativehumidity"):
            "Higher humidity can raise PM₂.₅ via aerosol water uptake and enhanced secondary particle formation, "
            "especially during haze events.",
        ("no2", "so2"):
            "NO₂ and SO₂ may vary together near industrial sources such as power plants that emit both.",
        ("co", "no2"):
            "CO and NO₂ are both products of incomplete combustion, so they often peak together near traffic.",
        ("co", "pm25"):
            "CO and PM₂.₅ share combustion sources; both tend to peak during cold, stagnant conditions.",
        ("o3", "so2"):
            "SO₂ and O₃ may show a negative relationship as SO₂ can be oxidised by O₃, consuming both.",
    }

    return explanations.get(
        (a, b),
        "No specific chemical interpretation is available for this pair. "
        "A positive r may suggest shared sources or common meteorological drivers; "
        "a negative r may indicate chemical interactions or different diurnal peak times."
    )

# ── Wind rose chart ────────────────────────────────────────────────────────────

def wind_rose(df, title): #Build a polar bar chart showing wind frequency by direction and speed.
    d = df.dropna(subset=["wind_speed", "wind_direction"]).copy()
    d["wind_direction"] = pd.to_numeric(d["wind_direction"], errors="coerce") % 360
    d["wind_speed"]     = pd.to_numeric(d["wind_speed"],     errors="coerce")
    d = d.dropna()

    if d.empty:
        return None

    # 16 directional sectors of 22.5 degrees each
    edges      = list(range(0, 361, 22))
    dir_labels = [f"{a}–{a+22}°" for a in edges[:-1]]
    d["sector"] = pd.cut(d["wind_direction"], bins=edges, labels=dir_labels,
                         include_lowest=True, right=False)

    # Speed in m/s
    speed_bins = [0, 1, 3, 5, 8, 12, 20, 1e9]
    d["speed_bin"] = pd.cut(d["wind_speed"], bins=speed_bins, include_lowest=True)

    agg = d.groupby(["sector", "speed_bin"]).size().reset_index(name="count")

    fig = px.bar_polar(agg, r="count", theta="sector", color="speed_bin", title=title)
    fig.update_layout(legend_title_text="Wind speed")
    return fig


#Database queries (all cached)

@st.cache_data(show_spinner=False)
def get_parameters(): #All distinct pollutant parameters in the database.
    with connect_ro() as con:
        rows = con.execute(
            "SELECT DISTINCT parameter FROM presentation.daily_air_quality_stats ORDER BY parameter"
        ).fetchall()
    return [r[0] for r in rows]


@st.cache_data(show_spinner=False)
def get_date_bounds(parameter): #get earliest and lastest dates for parameters
    with connect_ro() as con:
        mn, mx = con.execute("""
            SELECT MIN(measurement_date), MAX(measurement_date)
            FROM presentation.daily_air_quality_stats
            WHERE parameter = ?
        """, [parameter]).fetchone()
    return pd.to_datetime(mn), pd.to_datetime(mx)


@st.cache_data(show_spinner=False)
def get_station_options(parameter, start, end):  # keep all stations with parameters in this date range. 
    with connect_ro() as con:
        return con.execute("""
            SELECT DISTINCT location_id, location
            FROM presentation.daily_air_quality_stats
            WHERE parameter = ?
              AND measurement_date >= ?
              AND measurement_date <= ?
            ORDER BY location, location_id
        """, [parameter, start.date(), end.date()]).df()


@st.cache_data(show_spinner=False)
def load_station_timeseries(parameter, location_id, start, end):#daily average concentration. 
    with connect_ro() as con:
        return con.execute("""
            SELECT
                measurement_date AS day,
                AVG(average_value) AS value,
                ANY_VALUE(units)   AS units
            FROM presentation.daily_air_quality_stats
            WHERE parameter   = ?
              AND location_id = ?
              AND measurement_date >= ?
              AND measurement_date <= ?
            GROUP BY measurement_date
            ORDER BY measurement_date
        """, [parameter, location_id, start.date(), end.date()]).df()


@st.cache_data(show_spinner=False)
def load_distribution_sample(parameter, start, end, limit=20_000): #Sample of daily averages for the distribution chart.
    with connect_ro() as con:
        return con.execute("""
            SELECT average_value AS value, units
            FROM presentation.daily_air_quality_stats
            WHERE parameter = ?
              AND measurement_date >= ?
              AND measurement_date <= ?
            ORDER BY measurement_date DESC
            LIMIT ?
        """, [parameter, start.date(), end.date(), limit]).df()


@st.cache_data(show_spinner=False)
def load_coverage_stats(parameter, start, end): #Number of days with data per city (for the coverage bar chart).
    with connect_ro() as con:
        return con.execute("""
            SELECT
                location,
                COUNT(DISTINCT measurement_date) AS days_available
            FROM presentation.daily_air_quality_stats
            WHERE parameter = ?
              AND measurement_date >= ?
              AND measurement_date <= ?
            GROUP BY location
            ORDER BY days_available DESC
        """, [parameter, start.date(), end.date()]).df()


@st.cache_data(show_spinner=False)
def load_city_wide_daily(city, start, end, limit=8_000): #Multi-pollutant + met daily averages for a city (used in correlations & ML).
    with connect_ro() as con:
        return con.execute("""
            SELECT
                measurement_date AS day,
                pm25, pm10, no2, o3, co, so2,
                temperature, relativehumidity
            FROM presentation.daily_city_wide_core
            WHERE location = ?
              AND measurement_date >= ?
              AND measurement_date <= ?
            ORDER BY measurement_date
            LIMIT ?
        """, [city, start.date(), end.date(), limit]).df()


@st.cache_data(show_spinner=False)
def load_city_wind_daily(city, start, end, limit=8_000): #Daily wind speed and direction for a city.
    with connect_ro() as con:
        return con.execute("""
            SELECT day, wind_speed, wind_direction
            FROM presentation.daily_city_wind
            WHERE location = ?
              AND day >= ?
              AND day <= ?
            ORDER BY day
            LIMIT ?
        """, [city, start.date(), end.date(), limit]).df()


@st.cache_data(show_spinner=False)
def load_download_daily_multi(parameters, location_ids, start, end): #multi-parameter, multi-station daily data for the download tab.
    with connect_ro() as con:
        return con.execute("""
            SELECT
                s.measurement_date AS day,
                s.location, s.location_id,
                l.lat, l.lon,
                s.parameter, s.average_value, s.units
            FROM presentation.daily_air_quality_stats AS s
            LEFT JOIN presentation.dim_locations      AS l
              ON s.location_id = l.location_id
            WHERE s.parameter   IN (SELECT * FROM UNNEST(?))
              AND s.location_id IN (SELECT * FROM UNNEST(?))
              AND s.measurement_date >= ?
              AND s.measurement_date <= ?
            ORDER BY s.measurement_date, s.location, s.parameter
        """, [parameters, location_ids, start.date(), end.date()]).df()


@st.cache_data(show_spinner=False)
def load_latest_points_presentation(parameter, limit): #Most recent measurement per station (for the map).
    with connect_ro() as con:
        return con.execute("""
            SELECT location, location_id, datetime, lat, lon, parameter, value, units
            FROM presentation.latest_param_values_per_location
            WHERE parameter = ?
              AND lat   IS NOT NULL
              AND lon   IS NOT NULL
              AND value IS NOT NULL
            ORDER BY datetime DESC
            LIMIT ?
        """, [parameter, limit]).df()


# These three are cheap existence checks — used to decide which tabs to show
@st.cache_data(show_spinner=False)
def has_station_timeseries(parameter, location_id, start, end):
    with connect_ro() as con:
        return con.execute("""
            SELECT 1 FROM presentation.daily_air_quality_stats
            WHERE parameter=? AND location_id=? AND measurement_date>=? AND measurement_date<=?
            LIMIT 1
        """, [parameter, location_id, start.date(), end.date()]).fetchone() is not None

@st.cache_data(show_spinner=False)
def has_city_wide(city, start, end):
    with connect_ro() as con:
        return con.execute("""
            SELECT 1 FROM presentation.daily_city_wide_core
            WHERE location=? AND measurement_date>=? AND measurement_date<=?
            LIMIT 1
        """, [city, start.date(), end.date()]).fetchone() is not None

@st.cache_data(show_spinner=False)
def has_wind(city, start, end):
    with connect_ro() as con:
        return con.execute("""
            SELECT 1 FROM presentation.daily_city_wind
            WHERE location=? AND day>=? AND day<=?
              AND wind_speed IS NOT NULL AND wind_direction IS NOT NULL
            LIMIT 1
        """, [city, start.date(), end.date()]).fetchone() is not None


# Sidebar global filters
params = get_parameters()
default_param = "pm25" if "pm25" in params else params[0]

if st.button(" Refresh Data"):
    st.cache_data.clear()
    st.rerun()

with st.sidebar:
    st.header("Filters")

    parameter = st.selectbox(
        "Parameter", params,
        index=params.index(default_param),
        key="sb_param",
        format_func=lambda p: str(p).upper(),
    )

    mn_dt, mx_dt = get_date_bounds(parameter)
    start_date   = st.date_input("Start date", value=mn_dt.date(),
                                 min_value=mn_dt.date(), max_value=mx_dt.date(), key="sb_start")
    end_date     = st.date_input("End date",   value=mx_dt.date(),
                                 min_value=mn_dt.date(), max_value=mx_dt.date(), key="sb_end")
    st.divider()
    st.caption(f"DB: {DB_PATH.name}")

# Validate date range before doing anything else
if start_date > end_date:
    st.error("Start date must be ≤ end date.")
    st.stop()

# Convert to timestamps (end gets bumped to midnight so the day is included)
start_ts = pd.to_datetime(start_date)
end_ts   = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)


# Top-level tabs 

tab_map, tab_eda, tab_dl = st.tabs(["Map", "Data Analysis", "Download Data"])



# MAP TAB

with tab_map:
    st.subheader("Map")

    left, right = st.columns([2, 1])
    with left:
        map_mode = st.radio("Map mode", ["Latest per location"], horizontal=True, key="map_mode")
    with right:
        limit = st.slider("Max points", 200, 5000, 1200, 200, key="map_limit")

    with st.spinner("Loading map points..."):
        df_map = load_latest_points_presentation(parameter, limit)

    if df_map.empty:
        st.warning("No rows returned. Try widening the date range.")
        st.stop()

    # Clean up values and add WHO bands
    df_map             = df_map.copy()
    df_map["value_num"]= pd.to_numeric(df_map["value"], errors="coerce")
    df_map             = df_map.dropna(subset=["value_num", "lat", "lon"])
    df_map["datetime"] = pd.to_datetime(df_map["datetime"], errors="coerce")
    df_map             = add_who_band(df_map, parameter)

    # Friendly display name for the hoverbox
    PARAM_LABELS = {"pm25": "PM₂.₅", "pm10": "PM₁₀", "no2": "NO₂",
                    "o3": "O₃", "so2": "SO₂", "co": "CO"}
    param_name = PARAM_LABELS.get(str(parameter).lower().strip(), str(parameter).upper())

    # Build colour map from bands actually present in the data
    present_bands = df_map["who_band"].dropna().unique().tolist()
    band_labels   = sorted([b for b in present_bands if b != "WHO bands unavailable"])
    if "WHO bands unavailable" in present_bands:
        band_labels.append("WHO bands unavailable")

    colour_map = {lab: WHO_PALETTE.get(lab[0], "#999999") for lab in band_labels}

    fig = px.scatter_mapbox(
        df_map,
        lat="lat", lon="lon",
        color="who_band",
        category_orders={"who_band": band_labels},
        color_discrete_map=colour_map,
        hover_name="location",
        custom_data=["datetime", "value_num", "units", "who_band", "location_id"],
        zoom=3, height=650,
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Station ID: %{customdata[4]}<br>"
            "Last measurement: %{customdata[0]|%d %b %Y, %H:%M}<br>"
            f"{param_name}: %{{customdata[1]:.2f}} %{{customdata[2]}}<br>"
            "WHO band: %{customdata[3]}<extra></extra>"
        )
    )
    fig.update_layout(
        mapbox_style="open-street-map",
        margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            title="WHO guideline bands",
            x=0.01, y=0.99, xanchor="left", yanchor="top",
            bgcolor="rgba(255,246,249,0.85)", bordercolor="#f3c6d6", borderwidth=1,
            font=dict(size=11), itemsizing="constant",
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Raw map sample (first 50 rows)"):
        st.dataframe(df_map.head(50), use_container_width=True)



# DATA ANALYSIS TAB

with tab_eda:
    st.subheader("Data analysis")

    with st.spinner("Loading stations..."):
        stations_df = get_station_options(parameter, start_ts, end_ts)

    if stations_df.empty:
        st.warning("No stations available for this parameter/date range.")
        st.stop()

    # Build readable "City (id=123)" labels
    stations_df          = stations_df.copy()
    stations_df["location"] = stations_df["location"].astype(str)
    stations_df["label"] = stations_df["location"] + " (id=" + stations_df["location_id"].astype(str) + ")"
    all_labels           = stations_df["label"].tolist()

    # Station selector lives in the sidebar so it persists across inner tabs
    with st.sidebar:
        st.header("Location")
        station_label = st.selectbox("Station", all_labels, key="sb_station")
        smooth        = st.selectbox(
            "Daily smoothing",
            ["Daily (no smoothing)", "7-day rolling mean", "30-day rolling mean"],
            index=1, key="sb_smooth",
        )

    station_id = int(stations_df.loc[stations_df["label"] == station_label, "location_id"].iloc[0])
    city       = stations_df.loc[stations_df["location_id"] == station_id, "location"].iloc[0]

    # Build inner tab list dynamically — only show tabs with data
    inner_tabs = ["Compare stations"]
    if has_station_timeseries(parameter, station_id, start_ts, end_ts):
        inner_tabs += ["Time series", "Distribution"]
    inner_tabs.append("Coverage")
    if has_city_wide(city, start_ts, end_ts):
        inner_tabs += ["Correlations", "ML"]
    if has_wind(city, start_ts, end_ts):
        inner_tabs.append("Wind rose")

    tabs = dict(zip(inner_tabs, st.tabs(inner_tabs)))


    #Time series

    if "Time series" in tabs:
        with tabs["Time series"]:
            raw = load_station_timeseries(parameter, station_id, start_ts, end_ts)
            if raw.empty:
                st.warning("No data for this station.")
            else:
                units_val = raw["units"].dropna().iloc[0] if raw["units"].notna().any() else ""
                ts        = prep_daily_series(raw[["day", "value"]], start_ts, end_ts, smooth)
                suffix    = ts.attrs.get("suffix", "")

                fig_ts = px.line(
                    ts, x="day", y="value_plot",
                    title=f"{parameter.upper()} – {city} (id={station_id}){suffix}",
                    labels={"value_plot": f"{parameter} ({units_val})", "day": "Date"},
                    markers=False,
                )
                st.plotly_chart(style_fig(fig_ts), width="stretch")
                st.caption(f"Missing days: {int(ts['value'].isna().sum()):,} (shown as gaps)")
                download_csv_button(ts, "Download time series (CSV)",
                                    f"timeseries_{parameter}_{city}_{start_date}_{end_date}.csv", "dl_ts")
                tab_intro("Time series",
                          "Daily average concentrations for one station. "
                          "Smoothing highlights longer-term trends; gaps show missing data.")


    # Compare stations 

    with tabs["Compare stations"]:
        st.caption("Compare two stations for the same parameter and date window.")

        smooth_cmp = st.selectbox("Smoothing (compare)",
                                  ["None", "7-day rolling mean", "30-day rolling mean"],
                                  index=1, key="cmp_smooth")
        col_a, col_b = st.columns(2)
        with col_a:
            label_a = st.selectbox("Station A", all_labels, key="cmp_a")
        with col_b:
            label_b = st.selectbox("Station B", all_labels,
                                   index=min(1, len(all_labels) - 1), key="cmp_b")

        id_a = int(stations_df.loc[stations_df["label"] == label_a, "location_id"].iloc[0])
        id_b = int(stations_df.loc[stations_df["label"] == label_b, "location_id"].iloc[0])

        if id_a == id_b:
            st.info("Pick two different stations.")
        else:
            city_a = stations_df.loc[stations_df["location_id"] == id_a, "location"].iloc[0]
            city_b = stations_df.loc[stations_df["location_id"] == id_b, "location"].iloc[0]
            raw_a  = load_station_timeseries(parameter, id_a, start_ts, end_ts)
            raw_b  = load_station_timeseries(parameter, id_b, start_ts, end_ts)

            if raw_a.empty or raw_b.empty:
                st.warning("One of the stations has no data in this date range.")
            else:
                units_val = ""
                for d in (raw_a, raw_b):
                    if "units" in d.columns and d["units"].notna().any():
                        units_val = d["units"].dropna().iloc[0]; break

                # Combine both series into one long DataFrame for Plotly
                ser_a = prep_daily_series(raw_a[["day", "value"]], start_ts, end_ts, smooth_cmp)
                ser_b = prep_daily_series(raw_b[["day", "value"]], start_ts, end_ts, smooth_cmp)
                ser_a["station"] = f"{city_a} (id={id_a})"
                ser_b["station"] = f"{city_b} (id={id_b})"
                combined = pd.concat([ser_a, ser_b], ignore_index=True)

                fig_cmp = px.line(combined, x="day", y="value_plot", color="station",
                                  title=f"{parameter.upper()} – Station comparison",
                                  labels={"value_plot": f"{parameter} ({units_val})",
                                          "day": "Date", "station": "Station"})
                st.plotly_chart(style_fig(fig_cmp), width="stretch")
                tab_intro("Compare stations",
          "Plots two stations on the same chart so you can spot local differences. "
          "A persistent gap between lines often reflects different surroundings, "
          "such as roadside vs background. Occasional divergences may indicate "
          "a localised pollution episode at one site, or a period where one instrument "
          "was offline or miscalibrated.")

    # Distribution

    if "Distribution" in tabs:
        with tabs["Distribution"]:
            dist_df = load_station_timeseries(parameter, station_id, start_ts, end_ts)
            if dist_df.empty:
                st.warning("No distribution data for this range.")
            else:
                units_dist = dist_df["units"].dropna().iloc[0] if dist_df["units"].notna().any() else ""
                
                dist_df["day"] = pd.to_datetime(dist_df["day"])
                dist_df["month"] = dist_df["day"].dt.strftime("%b")
                dist_df["month_num"] = dist_df["day"].dt.month
                month_order = dist_df.sort_values("month_num")["month"].unique().tolist()

                fig_dist = px.box(
                    dist_df,
                    x="month",
                    y="value",
                    category_orders={"month": month_order},
                    points="outliers",
                    title=f"Monthly distribution of {parameter.upper()} daily averages – {city} ({units_dist})",
                    labels={"value": f"{parameter} ({units_dist})", "month": "Month"},
                )
                st.plotly_chart(style_fig(fig_dist, height=420), width="stretch")
                tab_intro("Distribution",
                        "Each box shows the spread of daily average concentrations within that calendar month, "
                        "across all years in the selected date range. "
                        "The line inside is the median; the box covers the middle 50% of values (interquartile range). "
                        "Whiskers extend to 1.5× the IQR. "
                        "Points beyond the whiskers are outliers and may reflect genuine pollution episodes, "
                        "dust events, or sensor faults. "
                        "Consistent seasonal patternss, such as elevated PM₂.₅ in autumn months, due to popular festivals such as Diwali,"
                        "are visible across years if the date range spans multiple years.")

    # Coverage

        with tabs["Coverage"]:
            cov_df = load_coverage_stats(parameter, start_ts, end_ts)
            if cov_df.empty:
                st.warning("No coverage data.")
            else:
                top_n = st.slider("Show top N cities", 10, 200, 30, 10, key="cov_topn")
                fig_cov = px.bar(cov_df.head(top_n).iloc[::-1], x="days_available", y="location",
                            orientation="h",
                            title=f"Top {top_n} cities by days available ({parameter})",
                            labels={"days_available": "Days with data", "location": "City"})
                
                fig_cov.update_layout(height=420 + top_n * 6)
                st.plotly_chart(style_fig(fig_cov, height=420 + top_n * 6), width="stretch")

            tab_intro("Coverage",
                    "Ranks cities by how many days of data exist in the selected window. "
                    "Cities with high coverage are more suitable for trend analysis. "
                    "Low coverage may mean the station is new, was temporarily offline, "
                    "or reports infrequently.")

        # Wind rose

        if "Wind rose" in tabs:
            with tabs["Wind rose"]:
                df_wind = load_city_wind_daily(city, start_ts, end_ts)
                fig_wr  = wind_rose(df_wind, f"Wind rose – {city}")
                if fig_wr is None:
                    st.warning("No wind data for this city/date range.")
                else:
                    st.plotly_chart(fig_wr, width="stretch")

                tab_intro("Wind rose",
                        "Each wedge points in the direction the wind is blowing from. "
                        "Wedge length shows how often wind comes from that direction. "
                        "Colours show wind speed bands in m/s. "
                        "If the dominant direction points towards a known source such as a motorway "
                        "or industrial area, that can help explain elevated readings at this station.")
                

        # Correlations
        if "Correlations" in tabs:
            with tabs["Correlations"]:
                df_corr = load_city_wide_daily(city, start_ts, end_ts)
                if df_corr.empty:
                    st.warning("No multi-pollutant data for this station.")
                else:
                    num_cols = [c for c in ["pm25","pm10","no2","o3","co","so2",
                                            "temperature","relativehumidity"]
                                if c in df_corr.columns and df_corr[c].notna().any()]
                    if len(num_cols) < 2:
                        st.warning("Need at least two pollutants with data.")
                    else:
                        # Heatmap overview
                        corr_matrix = df_corr[num_cols].corr()
                        fig_corr = px.imshow(
                            corr_matrix,
                            text_auto=".2f",
                            color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1,
                            title=f"Pearson correlation matrix – {city}",
                        )
                        st.plotly_chart(style_fig(fig_corr, height=500), width="stretch")

                        st.divider()

                        # Pairwise scatter
                        col1, col2 = st.columns(2)
                        with col1:
                            param_x = st.selectbox("Parameter X", num_cols, key="corr_x")
                        with col2:
                            param_y = st.selectbox("Parameter Y", num_cols,
                                                   index=min(1, len(num_cols) - 1), key="corr_y")

                        if param_x == param_y:
                            st.info("Choose two different parameters.")
                        else:
                            df_pair = df_corr[["day", param_x, param_y]].dropna()
                            if df_pair.empty:
                                st.warning("No overlapping days with both variables present.")
                            else:
                                r = df_pair[param_x].corr(df_pair[param_y])
                                m1, m2, m3 = st.columns(3)
                                m1.metric("City", city)
                                m2.metric("Days compared", f"{len(df_pair):,}")
                                m3.metric("Pearson r", f"{r:.2f}")

                                fig_scatter = px.scatter(
                                    df_pair, x=param_x, y=param_y, trendline="ols",
                                    title=f"{param_x.upper()} vs {param_y.upper()} – {city}",
                                    labels={param_x: param_x.upper(), param_y: param_y.upper()}
                                )
                                st.plotly_chart(style_fig(fig_scatter, height=420), width="stretch")

                                with st.expander("How to interpret this correlation", expanded=True):
                                    st.markdown(f"**{param_x.upper()} vs {param_y.upper()}**")
                                    st.markdown(correlation_interpretation(param_x, param_y))

                        tab_intro("Correlations",
                                  "The heatmap shows Pearson r for all pollutant pairs simultaneously: "
                                  "red indicates a positive correlation, blue a negative one. \n\n "
                                  "Use the dropdowns below to explore any pair in detail with a scatter plot and trendline. "
                                  "Pearson r ranges from -1 (perfect inverse) to +1 (perfect positive). "
                                  "Values near 0 indicate no linear relationship. \n\n"
                                  "Correlations reflect statistical association only not neccessarily causation.")
                
                


        # ML

        if "ML" in tabs:
            with tabs["ML"]:
                st.caption("Using presentation.daily_city_wide_core (city-level daily features).")

                df_ml = load_city_wide_daily(city, start_ts, end_ts)
                if df_ml.empty:
                    st.warning("No data for this city/date range.")
                    st.stop()

                all_features = [c for c in
                                ["pm25","pm10","no2","o3","co","so2","temperature","relativehumidity"]
                                if c in df_ml.columns]

                sub_lr, sub_if = st.tabs(["Linear regression", "Isolation Forest"])

                tab_intro("Simple modelling tools",
                        "Linear Regression fits a straight-line relationship between your chosen features (X) "
                        "and a target variable (y). R² tells you how much variance is explained, where 1.0 is "
                        "perfect and 0.0 means the model adds nothing. MAE is the average error in the target's "
                        "own units. A high R² indicates a strong statistical association but does not imply that "
                        "one pollutant causes another. Correlated variables often share a common driver such as "
                        "meteorology or a shared emission source. The model should only be treated as exploratory. \n\n"
                        "Isolation Forest scores each day by how unusual its combination of values looks relative "
                        "to the rest of the record. More negative scores indicate more anomalous days. "
                        "A flagged day is statistically unusual, not necessarily a confirmed pollution episode. "
                        "Always cross-reference with the time series, coverage tabs and literature before drawing conclusions. "
                        "Flagged days may reflect genuine pollution episodes, extreme weather, or sensor faults. "
                        "Hover over points to see the feature values for that day.")

                # Linear regression
                with sub_lr:
                    target   = st.selectbox("Target (y)",   all_features, key="lr_target")
                    features = st.multiselect("Features (X)",
                                            [c for c in all_features if c != target],
                                            default=[c for c in all_features if c != target][:4],
                                            key="lr_features")
                    test_size = st.slider("Test split", 0.1, 0.5, 0.2, 0.05, key="lr_test")

                    if not features:
                        st.info("Pick at least one feature.")
                    else:
                        df_lr = df_ml[["day", target] + features].dropna()
                        if len(df_lr) < 30:
                            st.warning("Not enough rows (need ≥ 30).")
                        else:
                            X = df_lr[features].values
                            y = df_lr[target].values
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, test_size=test_size, random_state=42)

                            model = Pipeline([("scaler", StandardScaler()), ("lr", LinearRegression())])
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                            c1, c2, c3 = st.columns(3)
                            c1.metric("Rows used", f"{len(df_lr):,}")
                            c2.metric("R²",  f"{r2_score(y_test, y_pred):.3f}")
                            c3.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.3f}")

                # Isolation Forest
                with sub_if:
                    feats_if = st.multiselect(
                        "Features",
                        all_features,
                        default=[c for c in ["pm25","pm10","no2","o3","co","so2"]
                                if c in all_features],
                        key="if_features",
                    )
                    contamination = st.slider("Expected anomaly fraction", 0.005, 0.2, 0.03, 0.005, key="if_cont")

                    if len(feats_if) < 2:
                        st.info("Choose at least two features.")
                    else:
                        df_if = df_ml[["day"] + feats_if].dropna()
                        if len(df_if) < 50:
                            st.warning("Need ~50+ rows.")
                        else:
                            pipe = Pipeline([
                                ("scaler", StandardScaler()),
                                ("iso", IsolationForest(n_estimators=300, contamination=contamination,
                                                        random_state=42, n_jobs=-1)),
                            ])
                            pipe.fit(df_if[feats_if].values)

                            Xs     = pipe.named_steps["scaler"].transform(df_if[feats_if].values)
                            scores = pipe.named_steps["iso"].decision_function(Xs)

                            out_if = df_if.copy()
                            out_if["anomaly_score"] = scores

                            fig_if = px.scatter(out_if, x="day", y="anomaly_score",
                                                hover_data=feats_if,
                                                title=f"Isolation Forest anomaly score – {city}")
                            st.plotly_chart(style_fig(fig_if), width="stretch")


        with tab_dl:
            st.subheader("Download data")
            st.caption("Download multiple parameters across multiple stations as a single CSV (daily averages).")

            # Pick parameters (multi)
            params_all = get_parameters()
            dl_params = st.multiselect(
                "Parameters",
                params_all,
                default=[p for p in ["pm25", "no2", "o3"] if p in params_all] or [params_all[0]],
                key="dl_params",
            )
            if not dl_params:
                st.info("Select at least one parameter.")
                st.stop()

            # Date range (use bounds of first selected parameter)
            mn_dt, mx_dt = get_date_bounds(dl_params[0])
            c1, c2 = st.columns(2)
            with c1:
                dl_start = st.date_input("Start date", value=mn_dt.date(), min_value=mn_dt.date(), max_value=mx_dt.date(), key="dl_start")
            with c2:
                dl_end = st.date_input("End date", value=mx_dt.date(), min_value=mn_dt.date(), max_value=mx_dt.date(), key="dl_end")

            if dl_start > dl_end:
                st.error("Start date must be <= end date.")
                st.stop()

            dl_start_ts = pd.to_datetime(dl_start)
            dl_end_ts = pd.to_datetime(dl_end)

            # Pick stations (multi)
            stn_df = get_station_options(dl_params[0], dl_start_ts, dl_end_ts)
            if stn_df.empty:
                st.warning("No stations available for that date range.")
                st.stop()

            stn_df = stn_df.copy()
            stn_df["location"] = stn_df["location"].astype(str)
            stn_df["label"] = stn_df["location"] + " (id=" + stn_df["location_id"].astype(str) + ")"
            station_labels = stn_df["label"].tolist()

            dl_stations = st.multiselect(
                "Stations",
                station_labels,
                default=station_labels[: min(5, len(station_labels))],
                key="dl_stations",
            )
            if not dl_stations:
                st.info("Select at least one station.")
                st.stop()

            location_ids = stn_df.loc[stn_df["label"].isin(dl_stations), "location_id"].astype(int).tolist()

            # Pull data
            with st.spinner("Preparing download..."):
                df_out = load_download_daily_multi(dl_params, location_ids, dl_start_ts, dl_end_ts)

            if df_out.empty:
                st.warning("No rows returned for that selection.")
                st.stop()

            st.success(f"Rows: {len(df_out):,} • Stations: {df_out['location_id'].nunique():,} • Parameters: {df_out['parameter'].nunique():,}")
            st.dataframe(df_out.head(200), width="stretch")

            download_csv_button(
                df_out,
                label="Download selected data (CSV)",
                filename=f"daily_multi_{dl_start}_{dl_end}.csv",
                key="dl_multi_btn",
            )
