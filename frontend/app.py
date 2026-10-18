from __future__ import annotations

import os
import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "Butterfly_AI_25_Areas.csv"
# The currently documented Mac SSH tunnel exposes remote port 8000 locally as 8001.
API_BASE = os.getenv("BUTTERFLY_API_URL", "http://127.0.0.1:8001").rstrip("/")
MODEL_FEATURES = [
    "population", "zone_area_km2", "population_density_per_km2",
    "road_length_km", "road_density_km_per_km2",
    "road_capacity_vehicles_per_hour", "traffic_volume_vehicles_per_hour",
    "avg_vehicle_speed_kmh", "avg_travel_time_min", "metro_station_count",
    "metro_ridership_daily", "bus_stop_count", "bus_route_count",
    "distance_to_nearest_transit_km", "job_count", "business_count",
    "office_count", "commercial_land_area_km2", "office_floor_area_sqm",
    "residential_building_count", "housing_units", "average_rent_inr_month",
    "housing_demand_index_0_100", "school_count", "hospital_count",
    "park_count", "water_body_count", "latitude", "longitude",
    "new_jobs_scenario", "estimated_commute_share",
    "estimated_private_vehicle_share", "estimated_public_transport_share",
    "estimated_new_commuters_scenario",
    "estimated_private_vehicle_commuters_scenario",
    "estimated_public_transport_commuters_scenario",
]

# This is the signed network from the supplied notebook, kept in its original
# direction and weights. The ripple runs for the same three rounds as training work.
RIPPLE_EDGES = [
    ("Transport", "Traffic", -0.8),
    ("Transport", "Businesses", 0.6),
    ("Transport", "Housing", 0.3),
    ("Traffic", "Pollution", 0.7),
    ("Traffic", "Businesses", -0.4),
    ("Businesses", "Employment", 0.9),
    ("Employment", "Housing", 0.5),
    ("Housing", "Traffic", 0.4),
]
RIPPLE_NODES = ["Transport", "Traffic", "Pollution", "Businesses", "Employment", "Housing"]


st.set_page_config(
    page_title="Butterfly AI · Bengaluru in motion",
    page_icon="🦋",
    layout="wide",
    initial_sidebar_state="collapsed",
)


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { color-scheme: dark; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .stApp { background: radial-gradient(ellipse at 82% 2%, #14254a 0, #0a142a 38%, #080f1e 78%); color: #f1f5ff; }
    [data-testid="stHeader"] { background: rgba(6, 13, 32, .78); }
    [data-testid="stMainBlockContainer"] { max-width: 1420px; padding-top: 2rem; padding-bottom: 4rem; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -.035em; color: #f5f7ff !important; }
    h1 { font-size: clamp(2.7rem, 5vw, 4.55rem) !important; font-weight: 800 !important; line-height: 1.04 !important; }
    h2 { font-size: clamp(1.7rem, 3vw, 2.45rem) !important; font-weight: 750 !important; }
    h3 { font-size: 1.25rem !important; font-weight: 700 !important; }
    p, li, label, .stCaption, [data-testid="stMetricLabel"] { font-size: 1.04rem !important; line-height: 1.6 !important; color: #bdc9e4 !important; }
    [data-testid="stMetricValue"] { font-size: 2.15rem !important; font-weight: 750 !important; color: #f3f6ff !important; }
    [data-testid="stMetric"] { background: linear-gradient(145deg, rgba(19, 38, 77, .78), rgba(10, 22, 48, .82)); border: 1px solid rgba(128, 165, 255, .17); border-radius: 20px; padding: 1.25rem 1.4rem; box-shadow: 0 18px 42px rgba(0,0,0,.14); }
    [data-testid="stMetricLabel"] p { color: #91a6ce !important; font-size: .92rem !important; }
    [data-testid="stMetricDelta"] { font-size: .92rem !important; }
    [data-testid="stSelectbox"], [data-testid="stSlider"] { padding-top: .35rem; }
    [data-testid="stSelectbox"] label, [data-testid="stSlider"] label { font-size: 1.04rem !important; font-weight: 650 !important; color: #e6edff !important; }
    [data-baseweb="select"] > div { background: #101f40; border-color: #284372; border-radius: 12px; min-height: 48px; }
    [data-testid="stForm"] { background: linear-gradient(135deg, rgba(16, 34, 72, .78), rgba(10, 22, 48, .8)); border: 1px solid rgba(120, 158, 244, .22); border-radius: 24px; padding: 1.5rem 1.7rem 1.3rem; }
    .stButton > button, .stFormSubmitButton > button { border: 0 !important; border-radius: 12px !important; min-height: 3.1rem; padding: .65rem 1.25rem !important; font-size: 1.05rem !important; font-weight: 750 !important; color: #071126 !important; background: linear-gradient(100deg, #7cf0d1, #a8f6c5) !important; box-shadow: 0 8px 25px rgba(73, 232, 186, .2); transition: transform .18s ease, box-shadow .18s ease; }
    .stButton > button:hover, .stFormSubmitButton > button:hover { transform: translateY(-2px); box-shadow: 0 12px 32px rgba(73, 232, 186, .32); }
    [data-testid="stAlert"] { border-radius: 15px; }
    hr { border-color: rgba(151, 177, 231, .15) !important; margin: 2rem 0 !important; }
    .eyebrow { color: #73e9cb; text-transform: uppercase; letter-spacing: .16em; font-size: .82rem; font-weight: 800; }
    .hero-copy { color: #b8c6e2; max-width: 770px; font-size: 1.2rem; line-height: 1.75; margin-top: 1.2rem; }
    .pill { display: inline-flex; align-items: center; gap: .5rem; border: 1px solid rgba(248, 195, 109, .35); background: rgba(248, 195, 109, .1); color: #ffd896; border-radius: 999px; padding: .42rem .82rem; font-size: .84rem; font-weight: 800; letter-spacing: .07em; text-transform: uppercase; }
    .status-pill { display: inline-flex; align-items: center; gap: .55rem; border: 1px solid rgba(112, 237, 199, .22); background: rgba(76, 220, 182, .09); color: #a8f6de; border-radius: 999px; padding: .48rem .88rem; font-weight: 700; font-size: .94rem; }
    .status-dot { width: .55rem; height: .55rem; border-radius: 50%; background: #64e6c0; box-shadow: 0 0 15px #64e6c0; }
    .glass-card { background: linear-gradient(145deg, rgba(17, 36, 77, .78), rgba(9, 20, 44, .86)); border: 1px solid rgba(128, 165, 255, .18); border-radius: 22px; padding: 1.35rem 1.5rem; }
    .section-kicker { color: #7de8d0; font-size: .82rem; text-transform: uppercase; letter-spacing: .14em; font-weight: 800; margin-bottom: .45rem; }
    .subtle { color: #91a3c5 !important; }
    .zone-name { font-family: 'Manrope',sans-serif; font-size: 1.34rem; font-weight: 800; color: #f4f7ff; }
    .ripple-value { font-family: 'Manrope',sans-serif; font-size: 1.24rem; font-weight: 800; color: #eef4ff; }
    .tiny-note { font-size: .88rem; color: #93a6c9; line-height: 1.55; }
    .intro-step { height: 100%; border: 1px solid rgba(138, 170, 238, .2); border-radius: 20px; background: rgba(14, 31, 66, .65); padding: 1.15rem 1.2rem; }
    .intro-num { font-size: .8rem; color: #72e5cb; font-weight: 850; letter-spacing: .12em; }
    .welcome-stage { min-height: min(76vh, 780px); display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; position:relative; overflow:hidden; border:1px solid rgba(140,171,235,.12); border-radius:30px; background:radial-gradient(ellipse at 50% 43%,rgba(28,62,110,.42),rgba(6,13,27,.92) 65%); padding:3rem 1.5rem; }
    .welcome-stage:before { content:''; position:absolute; inset:0; opacity:.62; background-image:radial-gradient(1px 1px at 12% 22%,#b7d6ff 99%,transparent),radial-gradient(2px 2px at 78% 20%,#8aaaff 99%,transparent),radial-gradient(1px 1px at 31% 72%,#d4c8ff 99%,transparent),radial-gradient(1px 1px at 88% 67%,#9cefe2 99%,transparent),radial-gradient(2px 2px at 59% 83%,#bfd5ff 99%,transparent),radial-gradient(1px 1px at 20% 85%,#bfd5ff 99%,transparent); background-size:340px 260px; animation: star-drift 24s linear infinite; pointer-events:none; }
    .welcome-content { position:relative; z-index:1; }
    .butterfly-wrap { display:flex; justify-content:center; align-items:center; min-height:230px; position:relative; }
    .butterfly-wrap:before { content:''; position:absolute; width:210px; height:210px; border-radius:50%; background:#5ee9d0; filter:blur(78px); opacity:.17; }
    .butterfly { position:relative; width:min(88%,330px); animation: fly-in 1.4s cubic-bezier(.2,.75,.2,1) both, float 3.6s 1.4s ease-in-out infinite; filter: drop-shadow(0 18px 30px rgba(75,229,209,.22)); }
    .brand-wordmark { display:flex; justify-content:center; align-items:baseline; gap:.03em; font:800 clamp(2.7rem,7vw,5rem)/1.05 Manrope,sans-serif; letter-spacing:-.065em; color:#f3f7ff; text-shadow:0 0 26px rgba(121,177,255,.22); }
    .brand-wordmark span { display:inline-block; opacity:0; transform:translateY(18px); animation: letter-draw .48s ease-out forwards; }
    .brand-wordmark .space { width:.28em; }
    .welcome-caption { color:#b5c4dd; font-size:1.15rem; margin:.9rem auto 1.5rem; }
    .welcome-stage .stButton { position:relative; z-index:1; }
    .wing-left { transform-origin: 50% 50%; animation: flap-left .72s ease-in-out infinite alternate; }
    .wing-right { transform-origin: 50% 50%; animation: flap-right .72s ease-in-out infinite alternate; }
    .glow-orbit { animation: orbit 11s linear infinite; transform-origin: 180px 130px; }
    @keyframes flap-left { from { transform: rotateY(0deg) rotate(-3deg); } to { transform: rotateY(47deg) rotate(-11deg); } }
    @keyframes flap-right { from { transform: rotateY(0deg) rotate(3deg); } to { transform: rotateY(-47deg) rotate(11deg); } }
    @keyframes float { 0%,100% { transform: translateY(0) rotate(-1deg); } 50% { transform: translateY(-13px) rotate(1deg); } }
    @keyframes orbit { to { transform: rotate(360deg); } }
    @keyframes letter-draw { to { opacity:1; transform:translateY(0); } }
    @keyframes fly-in { from { opacity:0; transform:translate(-110px,70px) scale(.72) rotate(-18deg); } to { opacity:1; transform:translate(0,0) scale(1) rotate(0); } }
    @keyframes star-drift { to { background-position: 180px -120px; } }
    @media (max-width: 700px) { [data-testid="stMainBlockContainer"] { padding: 1.1rem 1rem 3rem; } .hero-copy { font-size: 1.08rem; } [data-testid="stMetricValue"] { font-size: 1.65rem !important; } }
    </style>
    """,
    unsafe_allow_html=True,
)


def butterfly_svg() -> str:
    return """<svg class="butterfly" viewBox="0 0 360 250" role="img" aria-label="A teal and violet butterfly in flight">
      <defs>
        <linearGradient id="wingA" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#88f4dc"/><stop offset=".56" stop-color="#35c9dc"/><stop offset="1" stop-color="#6d82ff"/></linearGradient>
        <linearGradient id="wingB" x1="0" y1="1" x2="1" y2="0"><stop stop-color="#6b8cff"/><stop offset=".62" stop-color="#bd83ff"/><stop offset="1" stop-color="#f19bd6"/></linearGradient>
        <radialGradient id="wingGlow"><stop stop-color="#fff" stop-opacity=".8"/><stop offset="1" stop-color="#fff" stop-opacity="0"/></radialGradient>
      </defs>
      <circle class="glow-orbit" cx="180" cy="130" r="104" fill="none" stroke="#80ffe4" stroke-opacity=".18" stroke-dasharray="2 11" stroke-width="2"/>
      <g class="wing-left">
        <path d="M174 120C143 64 75 34 42 58 10 83 46 151 93 157 54 165 47 207 76 219 114 234 164 183 180 143Z" fill="url(#wingA)" stroke="#b8fff0" stroke-opacity=".75" stroke-width="2"/>
        <path d="M164 132C128 91 82 72 54 77M163 140C119 153 91 179 84 202" fill="none" stroke="#eafffc" stroke-opacity=".55" stroke-width="2"/>
        <ellipse cx="88" cy="105" rx="25" ry="17" fill="url(#wingGlow)" opacity=".42"/>
        <circle cx="105" cy="180" r="5" fill="#e7fffa" opacity=".8"/>
      </g>
      <g class="wing-right">
        <path d="M186 120C217 64 285 34 318 58 350 83 314 151 267 157 306 165 313 207 284 219 246 234 196 183 180 143Z" fill="url(#wingB)" stroke="#dfd5ff" stroke-opacity=".8" stroke-width="2"/>
        <path d="M196 132C232 91 278 72 306 77M197 140C241 153 269 179 276 202" fill="none" stroke="#fff" stroke-opacity=".55" stroke-width="2"/>
        <ellipse cx="272" cy="106" rx="25" ry="17" fill="url(#wingGlow)" opacity=".4"/>
        <circle cx="255" cy="180" r="5" fill="#f6f0ff" opacity=".8"/>
      </g>
      <path d="M180 104C171 122 172 155 180 174 188 155 189 122 180 104Z" fill="#f5edff" stroke="#e1dcff" stroke-width="2"/>
      <circle cx="180" cy="101" r="5" fill="#f7f4ff"/>
      <path d="M177 98Q157 76 147 78M183 98Q203 76 213 78" fill="none" stroke="#eee8ff" stroke-width="2" stroke-linecap="round"/>
      <circle cx="147" cy="78" r="3" fill="#8cf3dc"/><circle cx="213" cy="78" r="3" fill="#d4a5ff"/>
      <circle cx="47" cy="43" r="3" fill="#8af3df"/><circle cx="316" cy="38" r="2" fill="#fff" opacity=".8"/><circle cx="302" cy="226" r="3" fill="#b7a4ff"/>
    </svg>"""


@st.cache_data
def load_areas() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    required = {"zone_name", "latitude", "longitude", "congestion_index_0_100", "data_quality"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"The supplied zone CSV is missing expected fields: {', '.join(sorted(missing))}")
    return df


def signed_ripple(initial_change: float) -> dict[str, float]:
    nodes = list(dict.fromkeys([node for edge in RIPPLE_EDGES for node in edge[:2]]))
    impact = {node: 0.0 for node in nodes}
    current = impact.copy()
    current["Traffic"] = initial_change
    impact["Traffic"] = initial_change
    for _ in range(3):
        next_round = {node: 0.0 for node in nodes}
        for source, target, weight in RIPPLE_EDGES:
            next_round[target] += current[source] * weight
        for node in nodes:
            impact[node] += next_round[node]
        current = next_round
    return impact


def predict(features: dict[str, Any]) -> float:
    payload = {name: float(features[name]) for name in MODEL_FEATURES}
    response = requests.post(f"{API_BASE}/predict", json=payload, timeout=18)
    if response.status_code >= 400:
        try:
            detail = response.json().get("detail", "The prediction API returned an error.")
        except (ValueError, AttributeError):
            detail = "The prediction API returned an error."
        raise RuntimeError(f"{detail} (HTTP {response.status_code})")
    result = response.json()
    return float(result["prediction"])


@st.cache_data(ttl=20, show_spinner=False)
def check_api() -> tuple[bool, str]:
    try:
        response = requests.get(f"{API_BASE}/health", timeout=3)
        response.raise_for_status()
        return True, "Prediction API connected"
    except requests.RequestException:
        return False, "Prediction API not reachable"


def draw_intro() -> None:
    chars = "BUTTERFLY AI"
    wordmark = "".join(
        f'<span style="animation-delay:{1.15 + i * 0.075:.3f}s">{char}</span>' if char != " " else '<span class="space" aria-hidden="true"></span>'
        for i, char in enumerate(chars)
    )
    st.markdown(
        f'<section class="welcome-stage"><div class="welcome-content"><div class="eyebrow">A little more room to imagine</div><div class="butterfly-wrap">{butterfly_svg()}</div><div class="brand-wordmark" aria-label="Butterfly AI">{wordmark}</div><p class="welcome-caption">A city is made of small changes, connected.</p></div></section>',
        unsafe_allow_html=True,
    )
    _, center, _ = st.columns([1, 2, 1])
    with center:
        if st.button("Enter the city lab  →", type="primary", use_container_width=True, key="enter_city_lab_center"):
            st.session_state.entered_dashboard = True
            st.rerun()
        st.markdown('<p class="tiny-note" style="text-align:center;margin-top:1rem">Butterfly AI · Bengaluru neighbourhoods, explored with care</p>', unsafe_allow_html=True)


def map_chart(df: pd.DataFrame, selected_zone: str) -> None:
    areas = [
        {
            "id": str(row.zone_id),
            "name": str(row.zone_name),
            "lat": float(row.latitude),
            "lon": float(row.longitude),
            "congestion": float(row.congestion_index_0_100),
        }
        for row in df.itertuples(index=False)
    ]
    active = next((area for area in areas if area["name"] == selected_zone), areas[0])
    map_html = r"""<!doctype html><html><head><meta charset="utf-8">
<link href="https://unpkg.com/maplibre-gl@5.12.0/dist/maplibre-gl.css" rel="stylesheet">
<style>
html,body,#map{margin:0;width:100%;height:100%;background:#101b2d;font-family:Arial,sans-serif}
.map-label{position:absolute;top:14px;left:14px;z-index:2;padding:9px 13px;border:1px solid #ffffff22;border-radius:12px;background:#0b1425e8;color:#eaf2ff;font-size:13px;box-shadow:0 8px 20px #0005}
.map-label b{color:#90e6d5}.maplibregl-popup-content{background:#12213b;color:#f2f6ff;border:1px solid #345079;border-radius:12px;padding:13px 16px;line-height:1.5}.maplibregl-popup-close-button{color:#dce8fa;font-size:18px}.maplibregl-popup-tip{border-top-color:#12213b!important;border-bottom-color:#12213b!important}.maplibregl-ctrl-group{background:#152746!important;border:1px solid #345079!important}.maplibregl-ctrl-group button{filter:invert(.9)}.maplibregl-ctrl-attrib{background:#0b1425cc!important;color:#c4d2e9!important}.maplibregl-ctrl-attrib a{color:#aad4ff!important}
</style></head><body><div id="map"></div><div class="map-label"><b>25 neighbourhoods</b> · select a point to see its sample reference</div>
<script src="https://unpkg.com/maplibre-gl@5.12.0/dist/maplibre-gl.js"></script><script>
const areas=__AREAS__; const active=__ACTIVE__;
const map=new maplibregl.Map({container:'map',style:'https://tiles.openfreemap.org/styles/dark',center:[active.lon,active.lat],zoom:10.5,attributionControl:false});
map.addControl(new maplibregl.NavigationControl({showCompass:false}),'top-right');
map.addControl(new maplibregl.AttributionControl({compact:true,customAttribution:'© OpenFreeMap · © OpenStreetMap contributors'}),'bottom-right');
map.on('load',()=>{
  const bounds = new maplibregl.LngLatBounds();
  areas.forEach(area => bounds.extend([area.lon, area.lat]));
  map.fitBounds(bounds, {padding: 48, maxZoom: 10.7, duration: 0});
});
map.on('load',()=>{
 const features=areas.map(a=>({type:'Feature',geometry:{type:'Point',coordinates:[a.lon,a.lat]},properties:{id:a.id,name:a.name,congestion:a.congestion,selected:a.name===active.name}}));
 map.addSource('neighbourhoods',{type:'geojson',data:{type:'FeatureCollection',features}});
 map.addLayer({id:'area-glow',type:'circle',source:'neighbourhoods',paint:{'circle-radius':['case',['get','selected'],19,13],'circle-color':['case',['get','selected'],'#f8dda0',['step',['get','congestion'],'#65dfc8',28,'#719bf2',34,'#f29b83']],'circle-opacity':0.2,'circle-blur':0.65}});
 map.addLayer({id:'area-points',type:'circle',source:'neighbourhoods',paint:{'circle-radius':['case',['get','selected'],8,5.5],'circle-color':['case',['get','selected'],'#ffe09a',['step',['get','congestion'],'#65dfc8',28,'#719bf2',34,'#f29b83']],'circle-stroke-color':'#e9f5ff','circle-stroke-width':1.5,'circle-opacity':0.95}});
 map.on('mouseenter','area-points',()=>map.getCanvas().style.cursor='pointer'); map.on('mouseleave','area-points',()=>map.getCanvas().style.cursor='');
 map.on('click','area-points',e=>{const p=e.features[0].properties;const content=document.createElement('div');const title=document.createElement('strong');title.textContent=p.name;const detail=document.createElement('div');detail.textContent='Sample congestion reference: '+p.congestion+' / 100';const note=document.createElement('small');note.textContent='Choose this area in the list below to run a scenario.';content.append(title,detail,note);new maplibregl.Popup({closeButton:true,offset:12}).setLngLat(e.lngLat).setDOMContent(content).addTo(map);});
});
map.on('error',()=>{document.getElementById('map').style.background='#17243a'});
</script></body></html>"""
    map_html = map_html.replace("__AREAS__", json.dumps(areas, ensure_ascii=False).replace("</", "<\\/"))
    map_html = map_html.replace("__ACTIVE__", json.dumps(active, ensure_ascii=False).replace("</", "<\\/"))
    components.html(map_html, height=530, scrolling=False)


def show_result(result: dict[str, Any]) -> None:
    delta = result["delta"]
    positive = delta > 0
    direction = "higher" if positive else "lower" if delta < 0 else "unchanged"
    change_description = "unchanged" if delta == 0 else f"{direction} by {abs(delta):.2f} points"
    accent = "#ffae91" if positive else "#76ead0"
    st.markdown("### The ripple has landed")
    st.markdown(
        f'<div class="glass-card"><div class="section-kicker">{result["zone"]} · model estimate</div><div style="display:flex;gap:2rem;align-items:flex-end;flex-wrap:wrap"><div><div class="tiny-note">Baseline congestion</div><div class="ripple-value">{result["baseline"]:.2f}</div></div><div style="color:#7e94bd;font-size:1.4rem">→</div><div><div class="tiny-note">Scenario congestion</div><div class="ripple-value">{result["scenario"]:.2f}</div></div><div style="margin-left:auto"><div class="tiny-note">Change</div><div style="font:800 2rem Manrope;color:{accent}">{delta:+.2f}</div></div></div><p class="tiny-note" style="margin:.9rem 0 0">The model estimates congestion {change_description} for this scenario. Ripple values show the notebook’s signed network response.</p></div>',
        unsafe_allow_html=True,
    )
    impacts = result["ripple"]
    cols = st.columns(3, gap="medium")
    for i, node in enumerate(RIPPLE_NODES):
        value = impacts[node]
        with cols[i % 3]:
            sign = "up" if value > 0 else "down" if value < 0 else "steady"
            color = "#ffab8d" if value > 0 else "#72e8cd" if value < 0 else "#95a8cb"
            st.markdown(
                f'<div class="glass-card" style="padding:1rem 1.15rem;margin:.25rem 0"><div class="section-kicker">{node}</div><div class="ripple-value" style="color:{color}">{value:+.2f}</div><div class="tiny-note">{sign}stream impact</div></div>',
                unsafe_allow_html=True,
            )


def draw_dashboard() -> None:
    df = load_areas()
    api_ok, api_message = check_api()

    st.markdown('<div class="eyebrow">A neighbourhood view · Bengaluru</div>', unsafe_allow_html=True)
    st.title("The city, close to home.")
    intro_left, intro_right = st.columns([3, 1], vertical_alignment="center")
    with intro_left:
        st.markdown('<p class="hero-copy" style="margin-top:0">Pick a place you know. Try a small change to the daily commute and see how its effects travel across the city.</p>', unsafe_allow_html=True)
    with intro_right:
        status_color = "#a8f6de" if api_ok else "#ffd896"
        dot = "#64e6c0" if api_ok else "#f3bd67"
        st.markdown(f'<div class="status-pill" style="color:{status_color}"><span class="status-dot" style="background:{dot};box-shadow:0 0 15px {dot}"></span>{api_message}</div>', unsafe_allow_html=True)

    st.markdown('<span class="pill">For exploration · sample values</span>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">Take a look around</div>', unsafe_allow_html=True)
    st.markdown("## Neighbourhoods on the map")
    zones = df["zone_name"].tolist()
    default_zone = "Whitefield" if "Whitefield" in zones else zones[0]
    if "selected_zone" not in st.session_state or st.session_state.selected_zone not in zones:
        st.session_state.selected_zone = default_zone
    selected_zone = st.session_state.selected_zone
    map_chart(df, selected_zone)
    pick, map_note = st.columns([1.1, 2.2], vertical_alignment="center")
    with pick:
        selected_zone = st.selectbox("Area to explore", zones, key="selected_zone", label_visibility="visible")
    with map_note:
        st.markdown('<p class="tiny-note">Drag or zoom to explore. Click a point for its details, then choose that neighbourhood in the list to run a scenario.</p>', unsafe_allow_html=True)
    st.markdown('<p class="tiny-note">The supplied area file labels its rows synthetic and illustrative. This map is a prototype, not a live or validated city measure.</p>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">A small what-if</div>', unsafe_allow_html=True)
    st.markdown("## What if more trips used public transport?")
    st.markdown('<p class="hero-copy" style="font-size:1.05rem;margin-top:0">Choose how much of the commute to shift. The model compares the selected neighbourhood before and after, then traces the change through connected city systems.</p>', unsafe_allow_html=True)

    with st.form("scenario_controls"):
        shift_pp = st.slider(
            "Shift daily trips from private vehicles to public transport",
            min_value=0,
            max_value=20,
            value=5,
            step=1,
            format="%d pp",
            help="Follows the supplied notebook scenario: reduce private-vehicle share and raise public-transport share by the same amount.",
        )
        st.markdown(f'<div class="tiny-note">Scenario disturbance: <strong style="color:#88f0d6">{shift_pp} percentage points</strong> moved toward public transport.</div>', unsafe_allow_html=True)
        submitted = st.form_submit_button("Run the ripple simulation  →", use_container_width=True)

    if submitted:
        zone = df.loc[df["zone_name"].eq(selected_zone)].iloc[0]
        base = {name: float(zone[name]) for name in MODEL_FEATURES}
        scenario = base.copy()
        shift = shift_pp / 100.0
        if shift > base["estimated_private_vehicle_share"]:
            st.error("This shift is greater than the area’s private-vehicle share. Choose a smaller amount.")
        else:
            scenario["estimated_private_vehicle_share"] -= shift
            scenario["estimated_public_transport_share"] += shift
            with st.spinner("Asking the model to score both city states…"):
                try:
                    baseline_score = predict(base)
                    scenario_score = predict(scenario)
                    delta = scenario_score - baseline_score
                    st.session_state["scenario_result"] = {
                        "zone": selected_zone,
                        "baseline": baseline_score,
                        "scenario": scenario_score,
                        "delta": delta,
                        "ripple": signed_ripple(delta),
                        "shift_pp": shift_pp,
                    }
                except requests.RequestException:
                    st.session_state.pop("scenario_result", None)
                    st.error(f"Could not reach the prediction API at {API_BASE}. Start the backend or set BUTTERFLY_API_URL to its reachable address, then try again.")
                except (RuntimeError, KeyError, TypeError, ValueError) as exc:
                    st.session_state.pop("scenario_result", None)
                    st.error(f"The prediction request did not complete: {exc}")

    result = st.session_state.get("scenario_result")
    if result:
        st.markdown("<br>", unsafe_allow_html=True)
        show_result(result)
        st.markdown('<p class="tiny-note">Predictions depend on the fitted model and these illustrative inputs. The signed network shows directional relationships from the notebook; it is a conceptual ripple, not a causal forecast.</p>', unsafe_allow_html=True)
    else:
        st.info("Choose an area, set the slider, and run the simulation to see the model score and six-system ripple.")

    with st.expander("About the data and model"):
        st.write(
            "The dashboard reads the supplied `Butterfly_AI_25_Areas.csv`. The file labels the 25 rows as `synthetic` and says they are illustrative, not verified Bengaluru statistics. Scenario scores come from the existing FastAPI `/predict` endpoint. Ripple effects use the signed edges and three rounds from `butterflyAI-ML-Ripple-API.ipynb`."
        )
        st.code(f"Backend API: {API_BASE}\nPrediction route: POST /predict", language="text")

    st.markdown("<hr>", unsafe_allow_html=True)
    footer_left, footer_right = st.columns([3, 1])
    with footer_left:
        st.markdown('<p class="tiny-note">🦋 Butterfly AI · a city-systems prototype</p>', unsafe_allow_html=True)
    with footer_right:
        if st.button("Replay welcome", key="replay_welcome"):
            st.session_state.entered_dashboard = False
            st.rerun()


if "entered_dashboard" not in st.session_state:
    st.session_state.entered_dashboard = False

if not st.session_state.entered_dashboard:
    draw_intro()
else:
    draw_dashboard()
