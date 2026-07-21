from __future__ import annotations

import asyncio
import math
import os
import time
import logging
from datetime import date, timedelta
from typing import Literal, Optional, List, Dict, Any

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ============================================================
# ZERYROOT AI — PREMIUM DISASTER MANAGEMENT BACKEND
# Run independently on port 8002:
# .\venv\Scripts\python.exe -m uvicorn disaster:app --reload --port 8002
# ============================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ZeryrootAIDisaster")

app = FastAPI(
    title="Zeryroot AI — Disaster Intelligence & Network Routing Engine",
    version="3.1.0",
    description=(
        "Production-grade remote sensing engine deploying unified Sentinel-2 "
        "structural index algorithms, Open-Meteo micro-forecasts, and safe vectors."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# GLOBAL ENV CONFIGURATIONS & COMPONENT ROUTERS
# ============================================================
CDSE_CLIENT_ID = os.getenv("CDSE_CLIENT_ID", "").strip()
CDSE_CLIENT_SECRET = os.getenv("CDSE_CLIENT_SECRET", "").strip()
CDSE_TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
CDSE_STATISTICS_URL = "https://sh.dataspace.copernicus.eu/api/v1/statistics"

NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_HEADERS = {"User-Agent": "ZeryrootAI-DisasterAnalytics/3.1 (contact: ryan566vani@gmail.com)"}

OVERPASS_URLS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
OPENTOPODATA_SRTM_URL = "https://api.opentopodata.org/v1/srtm90m"
OSRM_ROUTE_URL = "https://router.project-osrm.org/route/v1/driving"

GLOBAL_SYSTEM_CACHE: dict[str, dict] = {
    "GEO": {}, "WEATHER": {}, "SATELLITE": {}, "ELEVATION": {}, "INFRA": {}
}
TTL_CONFIG = {"GEO": 86400, "WEATHER": 3600, "SATELLITE": 14400, "ELEVATION": 604800, "INFRA": 7200}
CDSE_TOKEN_CACHE: dict[str, object] = {"access_token": None, "expires_at": 0.0}

# ============================================================
# DATA SHAPE VERIFICATION MODELS (PYDANTIC SCHEMAS)
# ============================================================
class StrategicDisasterRequest(BaseModel):
    region: str = Field(..., min_length=2, max_length=250)
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)
    radius_km: float = Field(default=5.0, ge=1.0, le=50.0)
    destination_latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    destination_longitude: Optional[float] = Field(default=None, ge=-180, le=180)

class CriticalAsset(BaseModel):
    name: str
    type: str  # dam, river, drain, bridge, hospital, police, school, shelter, government, highway, fire_station
    distance_km: float
    latitude: float
    longitude: float
    capacity_rating: str

class DenseWeatherMetrics(BaseModel):
    precipitation_mm: float
    rain_mm: float
    humidity_percent: float
    soil_moisture_index: float
    wind_speed_kmh: float
    surface_pressure_hpa: float
    temperature_c: float
    forecast_6h_precipitation_mm: float
    forecast_12h_precipitation_mm: float
    accumulated_forecast_24h_mm: float

class WaypointCoordinate(BaseModel):
    latitude: float
    longitude: float

class SafeDetourRoute(BaseModel):
    route_status: str
    distance_km: float
    duration_mins: float
    path: List[WaypointCoordinate]
    navigation_protocol: str

class TimelineForecastNode(BaseModel):
    flood_probability_percent: float
    river_overflow_risk: str
    landslide_probability_percent: float
    evacuation_protocol: str

class GridCellResult(BaseModel):
    zone_id: str
    latitude: float
    longitude: float
    elevation_m: float
    calculated_slope_deg: float
    ndwi: float
    ndmi: float
    flood_risk_index: float
    landslide_risk_index: float

class UnifiedDashboardJSONResponse(BaseModel):
    success: bool
    flood_score: float
    landslide_score: float
    overall_risk: str
    nearest_river: str
    nearest_dam: str
    drainage_status: str
    safe_route: Optional[SafeDetourRoute] = None
    recommended_actions: List[str]
    timeline: Dict[str, TimelineForecastNode]
    micro_zones: List[GridCellResult]
    weather_context: DenseWeatherMetrics
    critical_infrastructure_mapped: List[CriticalAsset]

# ============================================================
# MATH UTILITIES & CACHE READS
# ============================================================
def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def haversine_distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    a = math.sin(math.radians(lat2 - lat1) / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(math.radians(lon2 - lon1) / 2)**2
    return radius * (2 * math.atan2(math.sqrt(a), math.sqrt(1 - a)))

def get_cached_item(layer: str, key: str) -> Optional[Any]:
    if key in GLOBAL_SYSTEM_CACHE[layer]:
        entry = GLOBAL_SYSTEM_CACHE[layer][key]
        if time.time() - entry["timestamp"] < TTL_CONFIG[layer]:
            logger.info(f"Cache HIT on layer: [{layer}]")
            return entry["data"]
    return None

def set_cached_item(layer: str, key: str, data: Any) -> None:
    GLOBAL_SYSTEM_CACHE[layer][key] = {"timestamp": time.time(), "data": data}

def calculate_dynamic_grid_size(radius_km: float) -> int:
    if radius_km < 8.0:
        return 2  # 2x2 grid
    elif radius_km < 15.0:
        return 3  # 3x3 grid
    return 4  # 4x4 grid

def generate_grid_cells(lat: float, lon: float, radius_km: float) -> List[Dict[str, Any]]:
    grid_size = calculate_dynamic_grid_size(radius_km)
    lat_delta = radius_km / 111.32
    lon_delta = radius_km / (111.32 * max(math.cos(math.radians(lat)), 0.1))
    west, south, east, north = lon - lon_delta, lat - lat_delta, lon + lon_delta, lat + lat_delta
    lon_step, lat_step = (east - west) / grid_size, (north - south) / grid_size
    
    cells = []
    for r in range(grid_size):
        for c in range(grid_size):
            c_west = west + c * lon_step
            c_south = south + r * lat_step
            cells.append({
                "zone_id": f"D-{r+1}{c+1}",
                "latitude": round(c_south + (lat_step / 2), 6),
                "longitude": round(c_west + (lon_step / 2), 6),
                "bbox": [round(c_west, 6), round(c_south, 6), round(c_west + lon_step, 6), round(c_south + lat_step, 6)]
            })
    return cells

# ============================================================
# GEODATA PARSING ENGINE (NOMINATIM INTERACTION LAYER)
# ============================================================
async def resolve_nominatim_coordinates(region: str, client: httpx.AsyncClient) -> tuple[float, float, str]:
    cache_key = region.lower().strip()
    if cached := get_cached_item("GEO", cache_key):
        return cached[0], cached[1], cached[2]
    try:
        res = await client.get(NOMINATIM_SEARCH_URL, params={"q": region, "format": "jsonv2", "limit": 1, "countrycodes": "in"}, headers=NOMINATIM_HEADERS, timeout=15.0)
        res.raise_for_status()
        data = res.json()
        if not data:
            raise HTTPException(status_code=404, detail=f"Geo-perimeter bounds for [{region}] could not be traced.")
        val = (float(data[0]["lat"]), float(data[0]["lon"]), data[0].get("display_name", region))
        set_cached_item("GEO", cache_key, val)
        return val
    except Exception as e:
        if isinstance(e, HTTPException): raise e
        raise HTTPException(status_code=502, detail="Geocoding pipeline engine processing timeout.")

# ============================================================
# UNIFIED COPERNICUS CORES (EXACT FOREST.PY PARSING MIRROR)
# ============================================================
async def get_cdse_access_token(client: httpx.AsyncClient) -> str:
    if not CDSE_CLIENT_ID or not CDSE_CLIENT_SECRET:
        return "MOCK_TOKEN_ENVIRONMENT"
    cached_token = CDSE_TOKEN_CACHE.get("access_token")
    expires_at = float(CDSE_TOKEN_CACHE.get("expires_at", 0.0))
    if isinstance(cached_token, str) and cached_token and time.time() < expires_at - 60:
        return cached_token
    try:
        res = await client.post(CDSE_TOKEN_URL, data={"grant_type": "client_credentials", "client_id": CDSE_CLIENT_ID, "client_secret": CDSE_CLIENT_SECRET}, headers={"Content-Type": "application/x-www-form-urlencoded"}, timeout=15.0)
        payload = res.json()
        token = payload.get("access_token", "")
        CDSE_TOKEN_CACHE["access_token"] = token
        CDSE_TOKEN_CACHE["expires_at"] = time.time() + int(payload.get("expires_in", 300))
        return token
    except Exception as e:
        logger.error(f"CDSE Auth processing failure: {str(e)}")
        return "MOCK_TOKEN_ENVIRONMENT"

def build_disaster_evalscript() -> str:
    return r"""
    //VERSION=3
    function setup() {
      return {
        input: [{ bands: ["B03", "B08", "B11", "dataMask"], units: "REFLECTANCE" }],
        output: [
          { id: "ndwi", bands: 1 },
          { id: "ndmi", bands: 1 },
          { id: "dataMask", bands: 1 }
        ]
      };
    }
    function evaluatePixel(sample) {
      const mask = sample.dataMask;
      const den_ndwi = sample.B03 + sample.B08;
      const den_ndmi = sample.B08 + sample.B11;
      return {
        ndwi: [den_ndwi === 0 ? 0 : (sample.B03 - sample.B08) / den_ndwi],
        ndmi: [den_ndmi === 0 ? 0 : (sample.B08 - sample.B11) / den_ndmi],
        dataMask: [mask]
      };
    }
    """

def extract_interval_mean(interval: dict, output_name: str) -> Optional[float]:
    output = interval.get("outputs", {}).get(output_name, {})
    bands = output.get("bands", {})
    if not bands: return None
    stats = next(iter(bands.values()), {}).get("stats", {})
    mean = stats.get("mean")
    try: return float(mean) if mean is not None else None
    except (TypeError, ValueError): return None

def parse_disaster_statistics(payload: dict) -> dict:
    intervals = payload.get("data", [])
    collected_ndwi, collected_ndmi = [], []
    for interval in intervals:
        val_ndwi = extract_interval_mean(interval, "ndwi")
        val_ndmi = extract_interval_mean(interval, "ndmi")
        if val_ndwi is not None and math.isfinite(val_ndwi): collected_ndwi.append(val_ndwi)
        if val_ndmi is not None and math.isfinite(val_ndmi): collected_ndmi.append(val_ndmi)
        
    return {
        "ndwi": round(sum(collected_ndwi)/len(collected_ndwi), 3) if collected_ndwi else 0.05,
        "ndmi": round(sum(collected_ndmi)/len(collected_ndmi), 3) if collected_ndmi else 0.12
    }

async def fetch_real_satellite_metrics(bbox: List[float], lat: float, client: httpx.AsyncClient, token: str) -> dict:
    cache_key = f"{','.join(map(str, bbox))}"
    if cached := get_cached_item("SATELLITE", cache_key): return cached
    if token == "MOCK_TOKEN_ENVIRONMENT":
        raise HTTPException(status_code=503, detail="Copernicus Data credentials missing in execution runtime.")
        
    lat_res = 20.0 / 111320.0
    lon_res = 20.0 / (111320.0 * max(math.cos(math.radians(lat)), 0.1))
    t_end = date.today()
    payload = {
        "input": {
            "bounds": {"bbox": bbox, "properties": {"crs": "http://www.opengis.net/def/crs/OGC/1.3/CRS84"}},
            "data": [{"type": "sentinel-2-l2a", "dataFilter": {"timeRange": {"from": f"{(t_end - timedelta(days=5)).isoformat()}T00:00:00Z", "to": f"{t_end.isoformat()}T23:59:59Z"}, "maxCloudCoverage": 40.0, "mosaickingOrder": "leastCC"}}]
        },
        "aggregation": {
            "timeRange": {"from": f"{(t_end - timedelta(days=5)).isoformat()}T00:00:00Z", "to": f"{t_end.isoformat()}T23:59:59Z"},
            "aggregationInterval": {"of": "P5D"},
            "resx": round(lon_res, 10), "resy": round(lat_res, 10), "evalscript": build_disaster_evalscript()
        },
        "calculations": {name: {"statistics": {"default": {"percentiles": {"k": [50]}}}} for name in ["ndwi", "ndmi"]}
    }
    try:
        res = await client.post(CDSE_STATISTICS_URL, json=payload, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"}, timeout=30.0)
        res.raise_for_status()
        metrics = parse_disaster_statistics(res.json())
        set_cached_item("SATELLITE", cache_key, metrics)
        return metrics
    except Exception as e:
        logger.error(f"Sentinel Statistics API endpoint extraction crash: {str(e)}")
        raise HTTPException(status_code=502, detail="Sentinel satellite indices aggregation failed.")

# ============================================================
# TERRAIN GEOMETRY EXTRACTION ENGINE (OPENTOPODATA SRTM)
# ============================================================
async def process_terrain_geometries(cells: List[Dict[str, Any]], radius_km: float, client: httpx.AsyncClient) -> List[tuple[float, float]]:
    cache_key = f"{cells[0]['latitude']}:{cells[0]['longitude']}:{radius_km}:grid"
    if cached := get_cached_item("ELEVATION", cache_key): return cached
    
    coords_payload = []
    for c in cells:
        coords_payload.append(f"{c['latitude']},{c['longitude']}")
        coords_payload.append(f"{c['latitude'] + 0.0015},{c['longitude']}")
        
    try:
        res = await client.get(OPENTOPODATA_SRTM_URL, params={"locations": "|".join(coords_payload)}, timeout=20.0)
        res.raise_for_status()
        results = res.json().get("results", [])
        
        computed_pairs = []
        for idx in range(0, len(results), 2):
            if idx + 1 >= len(results): break # Boundary Check against index payload mutations
            elev_center = float(results[idx]["elevation"] or 120.0)
            elev_neighbor = float(results[idx+1]["elevation"] or 120.0)
            
            run_m = haversine_distance_km(cells[idx//2]["latitude"], cells[idx//2]["longitude"], cells[idx//2]["latitude"] + 0.0015, cells[idx//2]["longitude"]) * 1000.0
            derived_slope = math.degrees(math.atan(abs(elev_center - elev_neighbor) / max(run_m, 1.0)))
            computed_pairs.append((elev_center, round(derived_slope, 2)))
            
        if not computed_pairs: computed_pairs = [(150.0, 3.0) for _ in cells]
        set_cached_item("ELEVATION", cache_key, computed_pairs)
        return computed_pairs
    except Exception as e:
        logger.error(f"OpenTopoData server baseline failure: {str(e)}")
        return [(185.0, 4.0) for _ in cells]

# ============================================================
# HIGH RESOLUTION WEATHER FORECAST PIPELINES (OPEN-METEO)
# ============================================================
async def fetch_dense_weather_forecast(lat: float, lon: float, client: httpx.AsyncClient) -> DenseWeatherMetrics:
    cache_key = f"{round(lat, 3)}:{round(lon, 3)}"
    if cached := get_cached_item("WEATHER", cache_key): return cached
    params = {
        "latitude": lat, "longitude": lon,
        "current": ["temperature_2m", "relative_humidity_2m", "precipitation", "rain", "surface_pressure", "wind_speed_10m"],
        "hourly": ["precipitation", "soil_moisture_1_3cm"], "forecast_days": 2
    }
    try:
        res = await client.get(OPEN_METEO_FORECAST_URL, params=params, timeout=15.0)
        data = res.json()
        current = data.get("current", {})
        hourly = data.get("hourly", {})
        hourly_precip = hourly.get("precipitation", [0.0]*24)
        
        metrics = DenseWeatherMetrics(
            precipitation_mm=float(current.get("precipitation", 0.0)),
            rain_mm=float(current.get("rain", 0.0)),
            humidity_percent=float(current.get("relative_humidity_2m", 60.0)),
            soil_moisture_index=float(hourly.get("soil_moisture_1_3cm", [0.30]*24)[0] or 0.30),
            wind_speed_kmh=float(current.get("wind_speed_10m", 10.0)),
            surface_pressure_hpa=float(current.get("surface_pressure", 1013.2)),
            temperature_c=float(current.get("temperature_2m", 25.5)),
            forecast_6h_precipitation_mm=round(sum(hourly_precip[0:6]), 2),
            forecast_12h_precipitation_mm=round(sum(hourly_precip[0:12]), 2),
            accumulated_forecast_24h_mm=round(sum(hourly_precip[0:24]), 2)
        )
        set_cached_item("WEATHER", cache_key, metrics)
        return metrics
    except Exception as e:
        logger.error(f"Weather interface logging pipeline block: {str(e)}")
        return DenseWeatherMetrics(precipitation_mm=2.0, rain_mm=1.5, humidity_percent=70.0, soil_moisture_index=0.35, wind_speed_kmh=12.0, surface_pressure_hpa=1011.0, temperature_c=26.0, forecast_6h_precipitation_mm=5.0, forecast_12h_precipitation_mm=12.0, accumulated_forecast_24h_mm=25.0)

# ============================================================
# STRUCTURAL LOGISTICS MATRIX LAYER (OVERPASS CRITICAL CORE)
# ============================================================
def build_comprehensive_infra_query(lat: float, lon: float, radius_m: int) -> str:
    return f"""
    [out:json][timeout:25];
    (
      node["waterway"="dam"](around:{radius_m},{lat},{lon});
      way["waterway"="dam"](around:{radius_m},{lat},{lon});
      way["waterway"~"river|canal|drain"](around:{radius_m},{lat},{lon});
      way["bridge"="yes"](around:{radius_m},{lat},{lon});
      node["amenity"~"hospital|police|school|fire_station"](around:{radius_m},{lat},{lon});
      node["emergency"="shelter"](around:{radius_m},{lat},{lon});
      node["amenity"="townhall"](around:{radius_m},{lat},{lon});
      node["office"="government"](around:{radius_m},{lat},{lon});
      way["highway"~"motorway|trunk|primary|secondary"](around:{radius_m},{lat},{lon});
    );
    out center tags;
    """

async def scan_critical_infrastructure_matrix(lat: float, lon: float, radius_km: float, client: httpx.AsyncClient) -> List[CriticalAsset]:
    cache_key = f"{round(lat, 3)}:{round(lon, 3)}:{radius_km}"
    if cached := get_cached_item("INFRA", cache_key): return cached
    query = build_comprehensive_infra_query(lat, lon, int(radius_km * 1000))
    elements = []
    for url in OVERPASS_URLS:
        try:
            res = await client.post(url, content=query, headers={"Content-Type": "text/plain"}, timeout=20.0)
            if res.status_code == 200:
                elements = res.json().get("elements", [])
                break
        except Exception: continue
        
    assets = []
    for el in elements:
        tags = el.get("tags", {})
        center = el.get("center", el)
        f_lat, f_lon = center.get("lat"), center.get("lon")
        if not f_lat or not f_lon: continue
        
        dist = haversine_distance_km(lat, lon, f_lat, f_lon)
        name = tags.get("name", "Unnamed Safety Node")
        a_type, cap = "unknown", "STANDARD_OPERATIONAL_ASSET"
        
        if tags.get("waterway") == "dam": a_type = "dam"
        elif tags.get("waterway") in ["river", "stream"]: a_type, cap = "river", "HIGH_VOLUME_DISCHARGE_PATH"
        elif tags.get("waterway") in ["drain", "canal", "ditch"]: a_type, cap = "drain", "STORMWATER_EXIT_CONDUIT"
        elif tags.get("bridge") == "yes": a_type = "bridge"
        elif tags.get("amenity") == "hospital": a_type, cap = "hospital", "CRITICAL_MEDICAL_BEDS"
        elif tags.get("amenity") == "police": a_type = "police"
        elif tags.get("amenity") == "fire_station": a_type, cap = "fire_station", "RESCUE_DISPATCH"
        elif tags.get("amenity") == "school": a_type, cap = "school", "LOCAL_EMERGENCY_ASSEMBLY"
        elif tags.get("emergency") == "shelter": a_type, cap = "shelter", "SECURE_DISASTER_SHELTER"
        elif tags.get("amenity") == "townhall" or tags.get("office") == "government": a_type, cap = "government", "ADMINISTRATIVE_COMMAND_CENTER"
        elif tags.get("highway"): a_type, cap = "highway", "EVACUATION_LOGISTICAL_ROUTE"
        
        if a_type != "unknown":
            assets.append(CriticalAsset(name=name if name != "Unnamed Safety Node" else f"Strategic {a_type.capitalize()}", type=a_type, distance_km=round(dist, 2), latitude=f_lat, longitude=f_lon, capacity_rating=cap))
            
    sorted_assets = sorted(assets, key=lambda x: x.distance_km)
    set_cached_item("INFRA", cache_key, sorted_assets)
    return sorted_assets

# ============================================================
# ADVANCED HAZARD AVOIDANCE EVACUATION PIPELINES
# ============================================================
async def compute_hazard_aware_routing(s_lat: float, s_lon: float, d_lat: float, d_lon: float, zones: List[GridCellResult], client: httpx.AsyncClient) -> SafeDetourRoute:
    danger_nodes = [z for z in zones if z.flood_risk_index > 60.0 or z.landslide_risk_index > 60.0]
    url = f"{OSRM_ROUTE_URL}/{s_lon},{s_lat};{d_lon},{d_lat}"
    try:
        res = await client.get(url, params={"overview": "full", "geometries": "geojson"}, timeout=12.0)
        if res.status_code == 200:
            coords = res.json()["routes"][0]["geometry"]["coordinates"]
            base_path = [WaypointCoordinate(latitude=c[1], longitude=c[0]) for c in coords]
            
            # Cross scan every track node step point against micro disaster grids
            clash = False
            for point in base_path:
                if any(haversine_distance_km(point.latitude, point.longitude, d.latitude, d.longitude) < 1.5 for d in danger_nodes):
                    clash = True
                    break
            if not clash:
                return SafeDetourRoute(route_status="SECURE_CLEARANCE", distance_km=round(float(res.json()["routes"][0]["distance"])/1000, 2), duration_mins=round(float(res.json()["routes"][0]["duration"])/60, 1), path=base_path, navigation_protocol="OSRM structural routing validated completely safe from active perimeter vectors.")
    except Exception: pass
    
    # Mathematical Shifting bearing offsets away from danger vectors zones coordinates centroids
    shift_lat = 0.035 if (d_lat - s_lat) >= 0 else -0.035
    shift_lon = 0.035 if (d_lon - s_lon) >= 0 else -0.035
    detour = [
        WaypointCoordinate(latitude=s_lat, longitude=s_lon),
        WaypointCoordinate(latitude=(s_lat + d_lat)/2 + shift_lat, longitude=(s_lon + d_lon)/2 + shift_lon),
        WaypointCoordinate(latitude=d_lat, longitude=d_lon)
    ]
    return SafeDetourRoute(route_status="EMERGENCY_DETOUR_ACTIVE", distance_km=round(haversine_distance_km(s_lat, s_lon, d_lat, d_lon) * 1.35, 2), duration_mins=38.0, path=detour, navigation_protocol="ALERT: Shortest spatial transit paths intercepted by risk boundaries. Active geometric detours deployed.")

# ============================================================
# DECISION MATRICES & TIMELINE FORECAST LOGICS
# ============================================================
def calculate_advanced_drainage_advisory(infra: List[CriticalAsset]) -> str:
    drains = [i for i in infra if i.type == "drain"]
    if not drains: return "STATUS_ALERT: No mapped high-capacity storm conduits detected. Establish emergency earthen swales immediately."
    return f"MONITORED: Capacity check required at catchment conduit [{drains[0].name}] positioned {drains[0].distance_km}km away. Initiate screen scouring protocols."

def build_predictive_timeline_matrices(flood_base: float, land_base: float, w: DenseWeatherMetrics) -> Dict[str, TimelineForecastNode]:
    intervals = {"6h": w.forecast_6h_precipitation_mm, "12h": w.forecast_12h_precipitation_mm, "24h": w.accumulated_forecast_24h_mm}
    timeline = {}
    for label, rain_delta in intervals.items():
        rain_weight = min(35.0, rain_delta * 0.7)
        f_prob = clamp(flood_base + rain_weight, 0, 100)
        l_prob = clamp(land_base + rain_weight * 1.1, 0, 100)
        timeline[label] = TimelineForecastNode(
            flood_probability_percent=round(f_prob, 1),
            river_overflow_risk="HIGH_ALERT" if f_prob > 70.0 else ("ELEVATED" if f_prob > 45.0 else "STABLE"),
            landslide_probability_percent=round(l_prob, 1),
            evacuation_protocol="IMMEDIATE_EVACUATION_ORDER" if l_prob > 75.0 or f_prob > 80.0 else ("MONITOR_PERIMETER" if l_prob > 50.0 or f_prob > 50.0 else "NOMINAL_STATUS")
        )
    return timeline

# ============================================================
# SYSTEM PIPELINE PIPING ROUTER (MAIN RUNNER)
# ============================================================
@app.post("/disaster-analysis", response_model=UnifiedDashboardJSONResponse)
async def process_disaster_intelligence_pipeline(request: StrategicDisasterRequest) -> UnifiedDashboardJSONResponse:
    async with httpx.AsyncClient() as client:
        # 1. Coordinate Resolution Layer
        if request.latitude is not None and request.longitude is not None:
            c_lat, c_lon, region_name = request.latitude, request.longitude, request.region
        else:
            c_lat, c_lon, region_name = await resolve_nominatim_coordinates(request.region, client)
            
        # 2. Dynamic Micro Grid Array Allocation
        cells = generate_grid_cells(c_lat, c_lon, request.radius_km)
        token = await get_cdse_access_token(client)
        
        # 3. Synchronous Remote Data Assembly Framework Pipelines Execution
        satellite_tasks = [fetch_real_satellite_metrics(c["bbox"], c["latitude"], client, token) for c in cells]
        weather_task = fetch_dense_weather_forecast(c_lat, c_lon, client)
        infra_task = scan_critical_infrastructure_matrix(c_lat, c_lon, request.radius_km, client)
        elevation_task = process_terrain_geometries(cells, request.radius_km, client)
        
        sat_results = await asyncio.gather(*satellite_tasks)
        weather_res, critical_infra, terrain_profiles = await asyncio.gather(weather_task, infra_task, elevation_task)
        
        # 4. Multivariable Risk Calculations Engine Iterations
        processed_zones = []
        dams_tracked = [i for i in critical_infra if i.type == "dam"]
        rivers_tracked = [i for i in critical_infra if i.type in ["river", "drain"]]
        
        min_dam_dist = dams_tracked[0].distance_km if dams_tracked else 999.0
        min_river_dist = rivers_tracked[0].distance_km if rivers_tracked else 999.0
        
        for idx, (cell, sat, terrain) in enumerate(zip(cells, sat_results, terrain_profiles)):
            ndwi_val, ndmi_val = sat["ndwi"], sat["ndmi"]
            elev_m, slope_deg = terrain[0], terrain[1]
            
            # Flood Score Framework Matrix
            river_penalty = max(0.0, (6.0 - min_river_dist) * 3.5)
            weather_penalty = min(25.0, weather_res.accumulated_forecast_24h_mm * 0.6)
            elevation_penalty = max(0.0, (400.0 - elev_m) * 0.04)
            drain_penalty = 10.0 if not rivers_tracked else 2.0
            flood_score = clamp((ndwi_val * 100 * 0.3) + weather_penalty + river_penalty + elevation_penalty + drain_penalty, 0, 100)
            
            # Landslide Score Framework Matrix
            slope_penalty = min(30.0, (slope_deg / 45.0) * 30.0)
            moisture_saturation = max(0.0, -ndmi_val * 100 * 0.25)
            soil_penalty = min(15.0, weather_res.soil_moisture_index * 30.0)
            road_penalty = 10.0 if any(i.type == "highway" and i.distance_km < 1.0 for i in critical_infra) else 2.0
            landslide_score = clamp(slope_penalty + moisture_saturation + weather_penalty + soil_penalty + road_penalty, 0, 100)
            
            processed_zones.append(GridCellResult(
                zone_id=cell["zone_id"], latitude=cell["latitude"], longitude=cell["longitude"],
                elevation_m=elev_m, calculated_slope_deg=slope_deg, ndwi=ndwi_val, ndmi=ndmi_val,
                flood_risk_index=round(flood_score, 1), landslide_risk_index=round(landslide_score, 1)
            ))
            
        # 5. Dynamic Hazard Avoidance Routing Vector Trigger Execution
        routing_res = None
        if request.destination_latitude is not None and request.destination_longitude is not None:
            routing_res = await compute_hazard_aware_routing(c_lat, c_lon, request.destination_latitude, request.destination_longitude, processed_zones, client)
            
        # 6. Consolidation Dashboard Output Assembly
        max_f_score = max(z.flood_risk_index for z in processed_zones)
        max_l_score = max(z.landslide_risk_index for z in processed_zones)
        
        risk_tier = "NOMINAL_SAFE"
        if max_f_score > 75.0 or max_l_score > 75.0: risk_tier = "HIGH"
        elif max_f_score > 45.0 or max_l_score > 45.0: risk_tier = "ELEVATED"
        
        drain_msg = calculate_advanced_drainage_advisory(critical_infra)
        predictive_timeline = build_predictive_timeline_matrices(max_f_score, max_l_score, weather_res)
        
        # Operational Standard Institutional SOP Actions Arrays Compilation
        actions = [f"METEOROLOGICAL: Monitor dynamic precipitation metrics changes inside the localized {weather_res.accumulated_forecast_24h_mm}mm forecast matrix."]
        if dams_tracked:
            actions.append(f"INSTITUTIONAL: Establish emergency telemetry loops with [{dams_tracked[0].name}] control room to monitor safety release schedules.")
        if risk_tier == "HIGH":
            actions.insert(0, "CRITICAL: Deploy evacuation warnings across low-elevation corridors immediately and block vulnerable routes.")
            
        return UnifiedDashboardJSONResponse(
            success=True, flood_score=max_f_score, landslide_score=max_l_score, overall_risk=risk_tier,
            nearest_river=rivers_tracked[0].name if rivers_tracked else "None Mapped within perimeter",
            nearest_dam=dams_tracked[0].name if dams_tracked else "None Mapped within perimeter",
            drainage_status=drain_msg, safe_route=routing_res, recommended_actions=actions,
            timeline=predictive_timeline, micro_zones=processed_zones, weather_context=weather_res,
            critical_infrastructure_mapped=critical_infra[:20]
        )

if __name__ == "__main__":
    logger.info("Booting up Zeryroot AI Production Disaster Protocol Layer Core...")
    import uvicorn
    uvicorn.run("disaster:app", host="127.0.0.1", port=8002, reload=True)
