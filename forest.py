from __future__ import annotations

import asyncio
import math
import os
import time
from datetime import date, timedelta
from typing import Literal, Optional

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


# ============================================================
# ZERYROOT AI — FOREST INTELLIGENCE BACKEND
# Run independently on port 8001:
# .\venv\Scripts\python.exe -m uvicorn forest:app --reload --port 8001
# ============================================================

app = FastAPI(
    title="Zeryroot AI — Forest Intelligence",
    version="1.1.0",
    description=(
        "Sentinel-2 based forest health, degradation, cause attribution, "
        "zone classification and ecological recovery recommendation backend."
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
# CONFIGURATION
# Uses the same terminal environment variables as main.py.
# ============================================================

CDSE_CLIENT_ID = os.getenv("CDSE_CLIENT_ID", "").strip()
CDSE_CLIENT_SECRET = os.getenv("CDSE_CLIENT_SECRET", "").strip()

CDSE_TOKEN_URL = os.getenv(
    "CDSE_TOKEN_URL",
    (
        "https://identity.dataspace.copernicus.eu/"
        "auth/realms/CDSE/protocol/openid-connect/token"
    ),
).strip()

CDSE_STATISTICS_URL = os.getenv(
    "CDSE_STATISTICS_URL",
    "https://sh.dataspace.copernicus.eu/api/v1/statistics",
).strip()

CDSE_COLLECTION_TYPE = os.getenv(
    "CDSE_COLLECTION_TYPE",
    "sentinel-2-l2a",
).strip()

CDSE_MAX_CLOUD_COVERAGE = float(
    os.getenv("CDSE_MAX_CLOUD_COVERAGE", "35")
)

FOREST_REQUEST_TIMEOUT_SECONDS = float(
    os.getenv("FOREST_REQUEST_TIMEOUT_SECONDS", "90")
)

NOMINATIM_SEARCH_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_HEADERS = {
    "User-Agent": (
        "ZeryrootAI-Forest/1.1 "
        "(contact: ryan566vani@gmail.com)"
    ),
    "Accept-Language": "en",
}

OVERPASS_URLS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

OVERPASS_HEADERS = {
    "User-Agent": (
        "ZeryrootAI-Forest/1.1 "
        "(contact: ryan566vani@gmail.com)"
    )
}

CDSE_TOKEN_CACHE: dict[str, object] = {
    "access_token": None,
    "expires_at": 0.0,
}

OPEN_METEO_ARCHIVE_URL = os.getenv(
    "OPEN_METEO_ARCHIVE_URL",
    "https://archive-api.open-meteo.com/v1/archive",
).strip()

OVERPASS_CACHE_TTL_SECONDS = int(
    os.getenv("OVERPASS_CACHE_TTL_SECONDS", "1800")
)

OVERPASS_CACHE: dict[str, dict] = {}
CLIMATE_CACHE: dict[str, dict] = {}


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

SeverityType = Literal[
    "HEALTHY",
    "EARLY_STRESS",
    "MODERATE_DEGRADATION",
    "HIGH_DEGRADATION",
    "CRITICAL_DEGRADATION",
]

CauseType = Literal[
    "HUMAN",
    "CLIMATE",
    "MIXED",
    "UNCERTAIN",
]

MapColorType = Literal[
    "GREEN",
    "YELLOW",
    "ORANGE",
    "RED",
    "PURPLE",
]


class ForestAnalysisRequest(BaseModel):
    region: str = Field(min_length=2, max_length=250)

    latitude: Optional[float] = Field(
        default=None,
        ge=-90,
        le=90,
    )
    longitude: Optional[float] = Field(
        default=None,
        ge=-180,
        le=180,
    )

    radius_km: float = Field(
        default=5,
        ge=1,
        le=100,
    )

    grid_size: int = Field(
        default=2,
        ge=2,
        le=4,
        description=(
            "2 creates 4 zones, 3 creates 9 zones, "
            "and 4 creates 16 zones."
        ),
    )

    baseline_start_date: Optional[date] = None
    baseline_end_date: Optional[date] = None

    current_start_date: Optional[date] = None
    current_end_date: Optional[date] = None

    include_human_pressure_scan: bool = True


class RecoveryAction(BaseModel):
    priority: Literal[
        "IMMEDIATE",
        "SHORT_TERM",
        "LONG_TERM",
    ]
    action: str
    reason: str


class ForestZoneResult(BaseModel):
    zone_id: str
    latitude: float
    longitude: float
    bbox: list[float]

    severity: SeverityType
    map_color: MapColorType
    forest_health_score: float

    ndvi_baseline: Optional[float] = None
    ndvi_current: Optional[float] = None
    ndvi_change: Optional[float] = None

    ndmi_baseline: Optional[float] = None
    ndmi_current: Optional[float] = None
    ndmi_change: Optional[float] = None

    ndbi_baseline: Optional[float] = None
    ndbi_current: Optional[float] = None
    ndbi_change: Optional[float] = None

    nbr_baseline: Optional[float] = None
    nbr_current: Optional[float] = None
    nbr_change: Optional[float] = None

    forest_cover_baseline_percent: Optional[float] = None
    forest_cover_current_percent: Optional[float] = None
    forest_cover_change_percent: Optional[float] = None

    cloud_ratio_percent: Optional[float] = None
    data_confidence_percent: float

    cause_type: CauseType
    likely_cause: str
    cause_confidence_percent: float

    human_pressure_evidence: dict
    climate_evidence: dict
    attribution_quality: str
    observed_evidence: list[str]
    cause_explanation: list[str]
    recovery_actions: list[RecoveryAction]


class ForestAnalysisResponse(BaseModel):
    success: bool
    analysis_type: str
    provider: dict

    region: str
    center: dict
    radius_km: float
    grid_size: int

    baseline_period: dict
    current_period: dict

    overall_forest_health_score: float
    degradation_detected: bool

    healthy_zone_percent: float
    stressed_zone_percent: float
    degraded_zone_percent: float
    critical_zone_percent: float

    dominant_cause_type: CauseType
    zones: list[ForestZoneResult]

    overall_summary: list[str]
    recommended_next_steps: list[str]
    limitations: list[str]


# ============================================================
# COMMON HELPERS
# ============================================================

def cdse_configured() -> bool:
    return bool(
        CDSE_CLIENT_ID
        and CDSE_CLIENT_SECRET
        and CDSE_TOKEN_URL
        and CDSE_STATISTICS_URL
    )


def forest_provider_status() -> dict:
    return {
        "configured": cdse_configured(),
        "provider_name": (
            "Copernicus Data Space Ecosystem Sentinel Hub"
        ),
        "mode": (
            "CDSE_SENTINEL_HUB"
            if cdse_configured()
            else "NOT_CONFIGURED"
        ),
        "collection": CDSE_COLLECTION_TYPE,
        "maximum_cloud_coverage_percent": (
            CDSE_MAX_CLOUD_COVERAGE
        ),
        "indices": [
            "NDVI",
            "NDMI",
            "NDBI",
            "NBR",
            "Sentinel-2 SCL",
        ],
        "climate_context_provider": "Open-Meteo historical archive",
        "fake_results_enabled": False,
    }


def clamp(
    value: float,
    minimum: float,
    maximum: float,
) -> float:
    return max(minimum, min(maximum, value))


def safe_round(
    value: Optional[float],
    digits: int = 4,
) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def difference(
    current: Optional[float],
    baseline: Optional[float],
) -> Optional[float]:
    if current is None or baseline is None:
        return None
    return current - baseline


def build_bbox_from_radius(
    latitude: float,
    longitude: float,
    radius_km: float,
) -> list[float]:
    latitude_delta = radius_km / 111.32

    longitude_scale = max(
        math.cos(math.radians(latitude)),
        0.1,
    )

    longitude_delta = (
        radius_km
        / (111.32 * longitude_scale)
    )

    return [
        round(longitude - longitude_delta, 7),
        round(latitude - latitude_delta, 7),
        round(longitude + longitude_delta, 7),
        round(latitude + latitude_delta, 7),
    ]


def generate_grid_cells(
    latitude: float,
    longitude: float,
    radius_km: float,
    grid_size: int,
) -> list[dict]:
    full_bbox = build_bbox_from_radius(
        latitude,
        longitude,
        radius_km,
    )

    west, south, east, north = full_bbox

    longitude_step = (east - west) / grid_size
    latitude_step = (north - south) / grid_size

    cells = []

    for row in range(grid_size):
        for column in range(grid_size):
            cell_west = west + column * longitude_step
            cell_east = cell_west + longitude_step

            cell_south = south + row * latitude_step
            cell_north = cell_south + latitude_step

            center_latitude = (
                cell_south + cell_north
            ) / 2

            center_longitude = (
                cell_west + cell_east
            ) / 2

            cells.append(
                {
                    "zone_id": (
                        f"F-{row + 1}{column + 1}"
                    ),
                    "row": row,
                    "column": column,
                    "latitude": round(
                        center_latitude,
                        7,
                    ),
                    "longitude": round(
                        center_longitude,
                        7,
                    ),
                    "bbox": [
                        round(cell_west, 7),
                        round(cell_south, 7),
                        round(cell_east, 7),
                        round(cell_north, 7),
                    ],
                }
            )

    return cells


def resolve_analysis_periods(
    request: ForestAnalysisRequest,
) -> tuple[date, date, date, date]:
    current_end = (
        request.current_end_date
        or date.today()
    )

    current_start = (
        request.current_start_date
        or current_end - timedelta(days=89)
    )

    baseline_end = (
        request.baseline_end_date
        or current_end.replace(
            year=current_end.year - 1
        )
    )

    baseline_start = (
        request.baseline_start_date
        or current_start.replace(
            year=current_start.year - 1
        )
    )

    if current_start > current_end:
        raise HTTPException(
            status_code=422,
            detail=(
                "current_start_date must be before "
                "current_end_date."
            ),
        )

    if baseline_start > baseline_end:
        raise HTTPException(
            status_code=422,
            detail=(
                "baseline_start_date must be before "
                "baseline_end_date."
            ),
        )

    return (
        baseline_start,
        baseline_end,
        current_start,
        current_end,
    )


async def geocode_region(
    region: str,
    client: httpx.AsyncClient,
) -> dict:
    try:
        response = await client.get(
            NOMINATIM_SEARCH_URL,
            params={
                "q": region,
                "format": "jsonv2",
                "limit": 1,
                "countrycodes": "in",
                "addressdetails": 1,
            },
            headers=NOMINATIM_HEADERS,
            timeout=20.0,
        )

        response.raise_for_status()
        payload = response.json()

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="Location geocoding timed out.",
        ) from exc

    except (
        httpx.HTTPStatusError,
        httpx.RequestError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to resolve the supplied forest region."
            ),
        ) from exc

    if not payload:
        raise HTTPException(
            status_code=404,
            detail=(
                "No Indian location matched the supplied "
                "forest region."
            ),
        )

    result = payload[0]

    return {
        "latitude": float(result["lat"]),
        "longitude": float(result["lon"]),
        "display_name": result.get(
            "display_name",
            region,
        ),
        "address": result.get("address", {}),
    }


# ============================================================
# CDSE AUTHENTICATION
# ============================================================

async def get_cdse_access_token(
    client: httpx.AsyncClient,
) -> str:
    if not cdse_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "CDSE credentials are not configured in "
                "the current forest backend terminal."
            ),
        )

    cached_token = CDSE_TOKEN_CACHE.get(
        "access_token"
    )

    expires_at = float(
        CDSE_TOKEN_CACHE.get(
            "expires_at",
            0.0,
        )
    )

    if (
        isinstance(cached_token, str)
        and cached_token
        and time.time() < expires_at - 60
    ):
        return cached_token

    try:
        response = await client.post(
            CDSE_TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": CDSE_CLIENT_ID,
                "client_secret": CDSE_CLIENT_SECRET,
            },
            headers={
                "Content-Type": (
                    "application/x-www-form-urlencoded"
                )
            },
            timeout=20.0,
        )

        response.raise_for_status()
        payload = response.json()

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                "CDSE OAuth token request timed out."
            ),
        ) from exc

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "CDSE OAuth authentication failed with "
                f"HTTP {exc.response.status_code}. "
                "Check the client ID and secret."
            ),
        ) from exc

    except (
        httpx.RequestError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to obtain a valid CDSE OAuth token."
            ),
        ) from exc

    access_token = payload.get("access_token")
    expires_in = int(
        payload.get("expires_in", 300)
    )

    if not access_token:
        raise HTTPException(
            status_code=502,
            detail=(
                "CDSE token response did not include "
                "an access token."
            ),
        )

    CDSE_TOKEN_CACHE["access_token"] = (
        access_token
    )

    CDSE_TOKEN_CACHE["expires_at"] = (
        time.time() + expires_in
    )

    return access_token


# ============================================================
# SENTINEL-2 FOREST EVALSCRIPT
# ============================================================

def build_forest_evalscript() -> str:
    return r"""
//VERSION=3
function setup() {
  return {
    input: [{
      bands: [
        "B03", "B04", "B08",
        "B11", "B12", "SCL", "dataMask"
      ],
      units: [
        "REFLECTANCE",
        "REFLECTANCE",
        "REFLECTANCE",
        "REFLECTANCE",
        "REFLECTANCE",
        "DN",
        "DN"
      ]
    }],
    output: [
      { id: "ndvi", bands: 1 },
      { id: "ndmi", bands: 1 },
      { id: "ndbi", bands: 1 },
      { id: "nbr", bands: 1 },
      { id: "forest", bands: 1 },
      { id: "valid", bands: 1 },
      { id: "cloud", bands: 1 },
      { id: "dataMask", bands: 1 }
    ]
  };
}

function safeIndex(a, b) {
  const denominator = a + b;
  return denominator === 0
    ? 0
    : (a - b) / denominator;
}

function evaluatePixel(sample) {
  const scl = sample.SCL;

  const isCloud = (
    scl === 3 ||
    scl === 8 ||
    scl === 9 ||
    scl === 10 ||
    scl === 11
  ) ? 1 : 0;

  const valid = (
    sample.dataMask === 1 &&
    isCloud === 0 &&
    scl !== 6
  ) ? 1 : 0;

  const ndvi = safeIndex(
    sample.B08,
    sample.B04
  );

  const ndmi = safeIndex(
    sample.B08,
    sample.B11
  );

  const ndbi = safeIndex(
    sample.B11,
    sample.B08
  );

  const nbr = safeIndex(
    sample.B08,
    sample.B12
  );

  const forest = (
    valid === 1 &&
    ndvi >= 0.52 &&
    ndmi >= 0.05
  ) ? 1 : 0;

  return {
    ndvi: [valid === 1 ? ndvi : 0],
    ndmi: [valid === 1 ? ndmi : 0],
    ndbi: [valid === 1 ? ndbi : 0],
    nbr: [valid === 1 ? nbr : 0],
    forest: [forest],
    valid: [valid],
    cloud: [isCloud],
    dataMask: [valid]
  };
}
"""


def extract_interval_mean(
    interval: dict,
    output_name: str,
) -> Optional[float]:
    output = (
        interval
        .get("outputs", {})
        .get(output_name, {})
    )

    bands = output.get("bands", {})

    if not bands:
        return None

    first_band = next(
        iter(bands.values()),
        {},
    )

    stats = first_band.get("stats", {})
    mean = stats.get("mean")

    try:
        return float(mean)
    except (TypeError, ValueError):
        return None


def average_available(
    values: list[Optional[float]],
) -> Optional[float]:
    available = [
        float(value)
        for value in values
        if value is not None
        and math.isfinite(float(value))
    ]

    if not available:
        return None

    return sum(available) / len(available)


def parse_forest_statistics(
    payload: dict,
) -> dict:
    intervals = payload.get("data", [])

    output_names = [
        "ndvi",
        "ndmi",
        "ndbi",
        "nbr",
        "forest",
        "valid",
        "cloud",
    ]

    collected: dict[
        str,
        list[Optional[float]],
    ] = {
        name: []
        for name in output_names
    }

    interval_dates = []

    for interval in intervals:
        interval_dates.append(
            interval.get("interval", {})
        )

        for output_name in output_names:
            collected[output_name].append(
                extract_interval_mean(
                    interval,
                    output_name,
                )
            )

    result = {
        name: average_available(values)
        for name, values in collected.items()
    }

    result["intervals_received"] = len(intervals)
    result["interval_dates"] = interval_dates

    if result["forest"] is not None:
        result["forest_percent"] = (
            result["forest"] * 100
        )
    else:
        result["forest_percent"] = None

    if result["cloud"] is not None:
        result["cloud_percent"] = (
            result["cloud"] * 100
        )
    else:
        result["cloud_percent"] = None

    return result


async def fetch_forest_statistics(
    bbox: list[float],
    center_latitude: float,
    start_date: date,
    end_date: date,
    client: httpx.AsyncClient,
    token: str,
) -> dict:
    latitude_resolution_degrees = (
        20.0 / 111_320.0
    )

    longitude_resolution_degrees = (
        20.0
        / (
            111_320.0
            * max(
                math.cos(
                    math.radians(
                        center_latitude
                    )
                ),
                0.1,
            )
        )
    )

    payload = {
        "input": {
            "bounds": {
                "bbox": bbox,
                "properties": {
                    "crs": (
                        "http://www.opengis.net/"
                        "def/crs/OGC/1.3/CRS84"
                    )
                },
            },
            "data": [
                {
                    "type": (
                        CDSE_COLLECTION_TYPE
                    ),
                    "dataFilter": {
                        "timeRange": {
                            "from": (
                                f"{start_date.isoformat()}"
                                "T00:00:00Z"
                            ),
                            "to": (
                                f"{end_date.isoformat()}"
                                "T23:59:59Z"
                            ),
                        },
                        "maxCloudCoverage": (
                            CDSE_MAX_CLOUD_COVERAGE
                        ),
                        "mosaickingOrder": (
                            "leastCC"
                        ),
                    },
                }
            ],
        },
        "aggregation": {
            "timeRange": {
                "from": (
                    f"{start_date.isoformat()}"
                    "T00:00:00Z"
                ),
                "to": (
                    f"{end_date.isoformat()}"
                    "T23:59:59Z"
                ),
            },
            "aggregationInterval": {
                "of": "P30D"
            },
            "resx": round(
                longitude_resolution_degrees,
                10,
            ),
            "resy": round(
                latitude_resolution_degrees,
                10,
            ),
            "evalscript": (
                build_forest_evalscript()
            ),
        },
        "calculations": {
            name: {
                "statistics": {
                    "default": {
                        "percentiles": {
                            "k": [25, 50, 75]
                        }
                    }
                }
            }
            for name in [
                "ndvi",
                "ndmi",
                "ndbi",
                "nbr",
                "forest",
                "valid",
                "cloud",
            ]
        },
    }

    try:
        response = await client.post(
            CDSE_STATISTICS_URL,
            json=payload,
            headers={
                "Authorization": (
                    f"Bearer {token}"
                ),
                "Content-Type": (
                    "application/json"
                ),
            },
            timeout=(
                FOREST_REQUEST_TIMEOUT_SECONDS
            ),
        )

        response.raise_for_status()
        response_payload = response.json()

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                "CDSE forest statistics request "
                "timed out."
            ),
        ) from exc

    except httpx.HTTPStatusError as exc:
        detail = exc.response.text[:700]

        raise HTTPException(
            status_code=502,
            detail=(
                "CDSE Statistical API returned "
                f"HTTP {exc.response.status_code}. "
                f"{detail}"
            ),
        ) from exc

    except (
        httpx.RequestError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to obtain valid Sentinel-2 "
                "forest statistics."
            ),
        ) from exc

    parsed = parse_forest_statistics(
        response_payload
    )

    parsed["source"] = (
        "Copernicus Data Space Ecosystem "
        "Sentinel-2 L2A"
    )

    parsed["period"] = {
        "start": start_date.isoformat(),
        "end": end_date.isoformat(),
    }

    return parsed



# ============================================================
# CLIMATE CONTEXT
# ============================================================

def _climate_cache_key(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
) -> str:
    return (
        f"{round(latitude, 2)}:{round(longitude, 2)}:"
        f"{start_date.isoformat()}:{end_date.isoformat()}"
    )


async def fetch_climate_period(
    latitude: float,
    longitude: float,
    start_date: date,
    end_date: date,
    client: httpx.AsyncClient,
) -> dict:
    cache_key = _climate_cache_key(
        latitude,
        longitude,
        start_date,
        end_date,
    )

    if cache_key in CLIMATE_CACHE:
        return CLIMATE_CACHE[cache_key]

    try:
        response = await client.get(
            OPEN_METEO_ARCHIVE_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "daily": (
                    "temperature_2m_mean,"
                    "temperature_2m_max,"
                    "precipitation_sum"
                ),
                "timezone": "UTC",
            },
            timeout=25.0,
        )
        response.raise_for_status()
        payload = response.json()
        daily = payload.get("daily", {})

        temperatures = [
            float(value)
            for value in daily.get(
                "temperature_2m_mean",
                [],
            )
            if value is not None
        ]

        maximum_temperatures = [
            float(value)
            for value in daily.get(
                "temperature_2m_max",
                [],
            )
            if value is not None
        ]

        precipitation = [
            float(value)
            for value in daily.get(
                "precipitation_sum",
                [],
            )
            if value is not None
        ]

        result = {
            "status": "SUCCESS",
            "provider": "Open-Meteo historical archive",
            "mean_temperature_c": (
                sum(temperatures) / len(temperatures)
                if temperatures
                else None
            ),
            "mean_max_temperature_c": (
                sum(maximum_temperatures)
                / len(maximum_temperatures)
                if maximum_temperatures
                else None
            ),
            "total_precipitation_mm": (
                sum(precipitation)
                if precipitation
                else None
            ),
            "days_received": max(
                len(temperatures),
                len(precipitation),
            ),
            "error": None,
        }

    except (
        httpx.TimeoutException,
        httpx.HTTPStatusError,
        httpx.RequestError,
        ValueError,
    ) as exc:
        result = {
            "status": "FAILED",
            "provider": "Open-Meteo historical archive",
            "mean_temperature_c": None,
            "mean_max_temperature_c": None,
            "total_precipitation_mm": None,
            "days_received": 0,
            "error": str(exc),
        }

    CLIMATE_CACHE[cache_key] = result
    return result


async def fetch_climate_comparison(
    latitude: float,
    longitude: float,
    baseline_start: date,
    baseline_end: date,
    current_start: date,
    current_end: date,
    client: httpx.AsyncClient,
) -> dict:
    baseline, current = await asyncio.gather(
        fetch_climate_period(
            latitude,
            longitude,
            baseline_start,
            baseline_end,
            client,
        ),
        fetch_climate_period(
            latitude,
            longitude,
            current_start,
            current_end,
            client,
        ),
    )

    rainfall_change_percent = None
    temperature_change_c = None
    maximum_temperature_change_c = None

    baseline_rainfall = baseline.get(
        "total_precipitation_mm"
    )
    current_rainfall = current.get(
        "total_precipitation_mm"
    )

    if (
        baseline_rainfall is not None
        and current_rainfall is not None
        and baseline_rainfall > 0
    ):
        rainfall_change_percent = (
            (current_rainfall - baseline_rainfall)
            / baseline_rainfall
            * 100
        )

    if (
        baseline.get("mean_temperature_c")
        is not None
        and current.get("mean_temperature_c")
        is not None
    ):
        temperature_change_c = (
            current["mean_temperature_c"]
            - baseline["mean_temperature_c"]
        )

    if (
        baseline.get("mean_max_temperature_c")
        is not None
        and current.get("mean_max_temperature_c")
        is not None
    ):
        maximum_temperature_change_c = (
            current["mean_max_temperature_c"]
            - baseline["mean_max_temperature_c"]
        )

    return {
        "status": (
            "SUCCESS"
            if (
                baseline.get("status") == "SUCCESS"
                and current.get("status") == "SUCCESS"
            )
            else "PARTIAL"
        ),
        "baseline": baseline,
        "current": current,
        "rainfall_change_percent": safe_round(
            rainfall_change_percent,
            2,
        ),
        "mean_temperature_change_c": safe_round(
            temperature_change_c,
            2,
        ),
        "maximum_temperature_change_c": safe_round(
            maximum_temperature_change_c,
            2,
        ),
    }


# ============================================================
# HUMAN PRESSURE SCAN
# ============================================================

def build_human_pressure_query(
    latitude: float,
    longitude: float,
    radius_m: int,
) -> str:
    return f"""
[out:json][timeout:25];
(
  way["highway"~"motorway|trunk|primary|secondary|tertiary"]
    (around:{radius_m},{latitude},{longitude});

  node["place"~"city|town|village|hamlet"]
    (around:{radius_m},{latitude},{longitude});

  way["landuse"~"residential|industrial|commercial|construction|quarry"]
    (around:{radius_m},{latitude},{longitude});

  relation["landuse"~"residential|industrial|commercial|construction|quarry"]
    (around:{radius_m},{latitude},{longitude});

  node["man_made"="mineshaft"]
    (around:{radius_m},{latitude},{longitude});
);
out center tags;
"""


def get_osm_coordinates(
    element: dict,
) -> tuple[
    Optional[float],
    Optional[float],
]:
    if (
        element.get("lat") is not None
        and element.get("lon") is not None
    ):
        return (
            float(element["lat"]),
            float(element["lon"]),
        )

    center = element.get("center", {})

    if (
        center.get("lat") is not None
        and center.get("lon") is not None
    ):
        return (
            float(center["lat"]),
            float(center["lon"]),
        )

    return None, None


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    radius = 6371.0088

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    delta_phi = math.radians(
        lat2 - lat1
    )

    delta_lambda = math.radians(
        lon2 - lon1
    )

    value = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(delta_lambda / 2) ** 2
    )

    central_angle = 2 * math.atan2(
        math.sqrt(value),
        math.sqrt(1 - value),
    )

    return radius * central_angle


def classify_pressure_feature(
    tags: dict,
) -> Optional[str]:
    if tags.get("highway"):
        return "road"

    if tags.get("place"):
        return "settlement"

    landuse = tags.get("landuse")

    if landuse in {
        "residential",
        "industrial",
        "commercial",
        "construction",
    }:
        return "built_up_land"

    if landuse == "quarry":
        return "quarry"

    if tags.get("man_made") == "mineshaft":
        return "mining"

    return None


async def fetch_human_pressure(
    latitude: float,
    longitude: float,
    radius_km: float,
    client: httpx.AsyncClient,
) -> dict:
    cache_key = (
        f"{round(latitude, 3)}:"
        f"{round(longitude, 3)}:"
        f"{round(radius_km, 1)}"
    )

    cached = OVERPASS_CACHE.get(cache_key)
    if (
        cached
        and time.time() - cached["stored_at"]
        < OVERPASS_CACHE_TTL_SECONDS
    ):
        return cached["value"]

    query = build_human_pressure_query(
        latitude,
        longitude,
        int(radius_km * 1000),
    )

    errors = []

    for overpass_url in OVERPASS_URLS:
        try:
            response = await client.post(
                overpass_url,
                content=query,
                headers={
                    **OVERPASS_HEADERS,
                    "Content-Type": "text/plain",
                },
                timeout=25.0,
            )

            response.raise_for_status()
            payload = response.json()
            elements = payload.get(
                "elements",
                [],
            )

            break

        except (
            httpx.TimeoutException,
            httpx.HTTPStatusError,
            httpx.RequestError,
            ValueError,
        ) as exc:
            errors.append(
                f"{overpass_url}: {exc}"
            )
    else:
        return {
            "status": "FAILED",
            "feature_count": 0,
            "nearest_road_km": None,
            "nearest_settlement_km": None,
            "nearest_built_up_land_km": None,
            "nearest_quarry_or_mining_km": None,
            "pressure_score": 0,
            "sample_features": [],
            "errors": errors,
        }

    grouped: dict[str, list[dict]] = {
        "road": [],
        "settlement": [],
        "built_up_land": [],
        "quarry": [],
        "mining": [],
    }

    for element in elements:
        tags = element.get("tags", {})
        category = classify_pressure_feature(
            tags
        )

        if category is None:
            continue

        feature_latitude, feature_longitude = (
            get_osm_coordinates(element)
        )

        if (
            feature_latitude is None
            or feature_longitude is None
        ):
            continue

        distance_km = haversine_distance_km(
            latitude,
            longitude,
            feature_latitude,
            feature_longitude,
        )

        grouped[category].append(
            {
                "name": tags.get(
                    "name",
                    "Unnamed feature",
                ),
                "category": category,
                "distance_km": round(
                    distance_km,
                    3,
                ),
                "latitude": feature_latitude,
                "longitude": feature_longitude,
            }
        )

    for features in grouped.values():
        features.sort(
            key=lambda item: item[
                "distance_km"
            ]
        )

    def nearest(
        *categories: str,
    ) -> Optional[float]:
        candidates = []

        for category in categories:
            if grouped[category]:
                candidates.append(
                    grouped[category][0][
                        "distance_km"
                    ]
                )

        return min(candidates) if candidates else None

    nearest_road = nearest("road")
    nearest_settlement = nearest(
        "settlement"
    )
    nearest_built = nearest(
        "built_up_land"
    )
    nearest_extraction = nearest(
        "quarry",
        "mining",
    )

    pressure_score = 0

    if nearest_road is not None:
        pressure_score += (
            30
            if nearest_road <= 1
            else 20
            if nearest_road <= 3
            else 8
        )

    if nearest_settlement is not None:
        pressure_score += (
            25
            if nearest_settlement <= 1
            else 15
            if nearest_settlement <= 3
            else 5
        )

    if nearest_built is not None:
        pressure_score += (
            25
            if nearest_built <= 1
            else 15
            if nearest_built <= 3
            else 5
        )

    if nearest_extraction is not None:
        pressure_score += (
            35
            if nearest_extraction <= 2
            else 20
            if nearest_extraction <= 5
            else 8
        )

    sample_features = sorted(
        [
            feature
            for features in grouped.values()
            for feature in features
        ],
        key=lambda item: item["distance_km"],
    )[:10]

    result = {
        "status": "SUCCESS",
        "feature_count": len(
            [
                item
                for items in grouped.values()
                for item in items
            ]
        ),
        "nearest_road_km": nearest_road,
        "nearest_settlement_km": (
            nearest_settlement
        ),
        "nearest_built_up_land_km": (
            nearest_built
        ),
        "nearest_quarry_or_mining_km": (
            nearest_extraction
        ),
        "pressure_score": clamp(
            pressure_score,
            0,
            100,
        ),
        "sample_features": sample_features,
        "errors": errors,
    }
    OVERPASS_CACHE[cache_key] = {
        "stored_at": time.time(),
        "value": result,
    }

    return result


# ============================================================
# FOREST HEALTH / DEGRADATION ENGINE
# ============================================================

def calculate_data_confidence(
    current: dict,
    baseline: dict,
) -> float:
    current_cloud = (
        current.get("cloud_percent")
        or 0
    )

    baseline_cloud = (
        baseline.get("cloud_percent")
        or 0
    )

    average_cloud = (
        current_cloud + baseline_cloud
    ) / 2

    interval_factor = min(
        (
            current.get(
                "intervals_received",
                0,
            )
            + baseline.get(
                "intervals_received",
                0,
            )
        ) / 6,
        1,
    )

    confidence = (
        100
        - average_cloud * 0.75
    ) * (0.75 + interval_factor * 0.25)

    return round(
        clamp(confidence, 20, 100),
        1,
    )


def calculate_forest_health_score(
    ndvi_current: Optional[float],
    ndvi_change: Optional[float],
    ndmi_change: Optional[float],
    forest_change_percent: Optional[float],
    nbr_change: Optional[float],
) -> float:
    score = 75.0

    if ndvi_current is not None:
        score += clamp(
            (ndvi_current - 0.45) * 55,
            -20,
            20,
        )

    if ndvi_change is not None:
        score += clamp(
            ndvi_change * 180,
            -35,
            15,
        )

    if ndmi_change is not None:
        score += clamp(
            ndmi_change * 90,
            -18,
            10,
        )

    if forest_change_percent is not None:
        score += clamp(
            forest_change_percent * 1.2,
            -25,
            10,
        )

    if nbr_change is not None:
        score += clamp(
            nbr_change * 80,
            -20,
            10,
        )

    return round(
        clamp(score, 0, 100),
        1,
    )


def classify_severity(
    health_score: float,
    ndvi_change: Optional[float],
    forest_change_percent: Optional[float],
) -> tuple[
    SeverityType,
    MapColorType,
]:
    ndvi_loss = (
        -(ndvi_change or 0)
    )

    forest_loss = (
        -(forest_change_percent or 0)
    )

    if (
        health_score >= 78
        and ndvi_loss < 0.04
        and forest_loss < 4
    ):
        return "HEALTHY", "GREEN"

    if (
        health_score >= 62
        and ndvi_loss < 0.09
        and forest_loss < 9
    ):
        return "EARLY_STRESS", "YELLOW"

    if (
        health_score >= 45
        and ndvi_loss < 0.16
        and forest_loss < 16
    ):
        return (
            "MODERATE_DEGRADATION",
            "ORANGE",
        )

    if (
        health_score >= 25
        and ndvi_loss < 0.25
        and forest_loss < 25
    ):
        return (
            "HIGH_DEGRADATION",
            "RED",
        )

    return (
        "CRITICAL_DEGRADATION",
        "RED",
    )


def classify_cause(
    ndvi_change: Optional[float],
    ndmi_change: Optional[float],
    ndbi_change: Optional[float],
    nbr_change: Optional[float],
    forest_change_percent: Optional[float],
    pressure: dict,
    climate: dict,
    severity: SeverityType,
) -> dict:
    if severity == "HEALTHY":
        return {
            "cause_type": "UNCERTAIN",
            "likely_cause": "No material degradation detected",
            "confidence": 92.0,
            "attribution_quality": "NOT_REQUIRED",
            "evidence": [
                "Forest health and cover remain stable within the selected comparison periods."
            ],
            "explanation": [
                "Cause attribution is not required because the zone is classified as healthy.",
                "Continue periodic same-season monitoring to detect future change.",
            ],
        }

    ndvi_loss = max(0, -(ndvi_change or 0))
    moisture_loss = max(0, -(ndmi_change or 0))
    built_up_gain = max(0, ndbi_change or 0)
    burn_loss = max(0, -(nbr_change or 0))
    forest_loss = max(
        0,
        -(forest_change_percent or 0),
    )

    pressure_score = float(
        pressure.get("pressure_score", 0)
        or 0
    )

    rainfall_deficit = max(
        0,
        -(
            climate.get(
                "rainfall_change_percent",
                0,
            )
            or 0
        ),
    )

    temperature_gain = max(
        0,
        climate.get(
            "mean_temperature_change_c",
            0,
        )
        or 0,
    )

    maximum_temperature_gain = max(
        0,
        climate.get(
            "maximum_temperature_change_c",
            0,
        )
        or 0,
    )

    human_data_available = (
        pressure.get("status") == "SUCCESS"
    )

    climate_data_available = (
        climate.get("status") == "SUCCESS"
    )

    human_score = (
        built_up_gain * 420
        + pressure_score * 0.65
        + forest_loss * 0.7
    )

    climate_score = (
        moisture_loss * 280
        + burn_loss * 210
        + ndvi_loss * 70
        + min(rainfall_deficit, 60) * 0.45
        + temperature_gain * 8
        + maximum_temperature_gain * 5
    )

    evidence = []
    explanation = []

    if built_up_gain >= 0.012:
        evidence.append(
            "Built-up spectral signal increased during the comparison period."
        )

    if pressure_score >= 15:
        evidence.append(
            "Mapped road, settlement, construction or extraction pressure is present near the zone."
        )

    if moisture_loss >= 0.012:
        evidence.append(
            "Canopy moisture index declined."
        )

    if burn_loss >= 0.025:
        evidence.append(
            "NBR declined, indicating dryness, burn stress or canopy disturbance."
        )

    if forest_loss >= 2:
        evidence.append(
            f"Estimated dense forest cover reduced by {forest_loss:.1f} percentage points."
        )

    if rainfall_deficit >= 10:
        evidence.append(
            f"Rainfall was approximately {rainfall_deficit:.1f}% below the baseline period."
        )

    if temperature_gain >= 0.5:
        evidence.append(
            f"Mean temperature increased by approximately {temperature_gain:.1f}°C."
        )

    if not human_data_available:
        evidence.append(
            "Nearby human-pressure service was unavailable or rate-limited; human attribution confidence was reduced."
        )

    if not climate_data_available:
        evidence.append(
            "Historical weather context was unavailable; climate attribution confidence was reduced."
        )

    if (
        human_score >= 20
        and climate_score >= 18
        and abs(human_score - climate_score)
        <= max(human_score, climate_score) * 0.55
    ):
        cause_type: CauseType = "MIXED"
        likely_cause = (
            "Combined human pressure and climate stress"
        )
        confidence = clamp(
            60
            + min(human_score, 55) * 0.28
            + min(climate_score, 55) * 0.26,
            58,
            94,
        )
        explanation.extend(
            [
                "Vegetation loss coincides with both land-use pressure and moisture, temperature or burn stress.",
                "Climate stress may have weakened the forest while nearby human activity increased fragmentation or disturbance.",
            ]
        )

    elif (
        human_score >= 18
        and human_score
        >= climate_score * 1.15
    ):
        cause_type = "HUMAN"

        if (
            pressure.get(
                "nearest_quarry_or_mining_km"
            )
            is not None
            and pressure[
                "nearest_quarry_or_mining_km"
            ] <= 5
        ):
            likely_cause = (
                "Mining or quarry-linked land disturbance"
            )
        elif built_up_gain >= 0.025:
            likely_cause = (
                "Built-up, road or settlement expansion"
            )
        elif pressure_score >= 20:
            likely_cause = (
                "Road, settlement or land-use pressure"
            )
        else:
            likely_cause = (
                "Possible human forest clearing or fragmentation"
            )

        confidence = clamp(
            58 + min(human_score, 70) * 0.48,
            56,
            95,
        )

        if not human_data_available:
            confidence -= 12

        explanation.extend(
            [
                "Forest or vegetation decline occurs with increasing built-up signal and/or nearby mapped human-pressure features.",
                "The pattern is more consistent with land conversion, fragmentation or clearing than climate stress alone.",
            ]
        )

    elif climate_score >= 15:
        cause_type = "CLIMATE"

        if (
            burn_loss >= 0.05
            and (
                rainfall_deficit >= 10
                or temperature_gain >= 0.5
            )
        ):
            likely_cause = (
                "Fire susceptibility caused by heat and dryness"
            )
        elif rainfall_deficit >= 15:
            likely_cause = (
                "Rainfall deficit and drought-related canopy stress"
            )
        elif moisture_loss >= 0.03:
            likely_cause = (
                "Prolonged canopy moisture stress"
            )
        elif temperature_gain >= 0.8:
            likely_cause = (
                "Heat-related vegetation stress"
            )
        else:
            likely_cause = (
                "Climate-related vegetation stress"
            )

        confidence = clamp(
            58 + min(climate_score, 70) * 0.46,
            55,
            94,
        )

        if not climate_data_available:
            confidence -= 12

        explanation.extend(
            [
                "Vegetation health declined together with moisture, burn, rainfall or temperature indicators.",
                "The likely mechanism is reduced water availability and/or higher heat stress, which weakens canopy condition and can increase fire susceptibility.",
            ]
        )

    else:
        cause_type = "UNCERTAIN"
        likely_cause = (
            "No single cause can be isolated confidently"
        )
        confidence = 48.0
        explanation.extend(
            [
                "The available satellite, weather and nearby-feature signals do not clearly isolate one cause.",
                "Field verification, active-fire records, legal land-use data and higher-resolution imagery are recommended.",
            ]
        )

    available_sources = sum(
        [
            True,
            human_data_available,
            climate_data_available,
        ]
    )

    attribution_quality = (
        "HIGH"
        if available_sources == 3
        and confidence >= 75
        else "MEDIUM"
        if available_sources >= 2
        and confidence >= 58
        else "LOW"
    )

    return {
        "cause_type": cause_type,
        "likely_cause": likely_cause,
        "confidence": round(
            clamp(confidence, 35, 95),
            1,
        ),
        "attribution_quality": attribution_quality,
        "evidence": evidence,
        "explanation": explanation,
    }


def generate_recovery_actions(
    cause_type: CauseType,
    likely_cause: str,
    severity: SeverityType,
) -> list[RecoveryAction]:
    actions: list[RecoveryAction] = []

    if severity == "HEALTHY":
        return [
            RecoveryAction(
                priority="LONG_TERM",
                action=(
                    "Continue same-season satellite monitoring "
                    "and protect existing canopy condition."
                ),
                reason=(
                    "The zone is currently healthy; prevention "
                    "and early detection are more appropriate "
                    "than restoration."
                ),
            )
        ]

    if severity in {
        "HIGH_DEGRADATION",
        "CRITICAL_DEGRADATION",
    }:
        actions.append(
            RecoveryAction(
                priority="IMMEDIATE",
                action=(
                    "Freeze additional disturbance inside the "
                    "detected hotspot and establish a monitored "
                    "ecological buffer."
                ),
                reason=(
                    "Critical zones may continue losing canopy "
                    "without immediate protection."
                ),
            )
        )

    if cause_type == "HUMAN":
        actions.extend(
            [
                RecoveryAction(
                    priority="IMMEDIATE",
                    action=(
                        "Verify road, settlement, mining and "
                        "land-clearing activity through field "
                        "inspection and recent high-resolution "
                        "imagery."
                    ),
                    reason=(
                        "Enforcement action should be based on "
                        "verified land-use evidence."
                    ),
                ),
                RecoveryAction(
                    priority="SHORT_TERM",
                    action=(
                        "Start assisted natural regeneration and "
                        "native mixed-species plantation in "
                        "cleared or fragmented patches."
                    ),
                    reason=(
                        "Native restoration improves canopy "
                        "recovery and habitat connectivity."
                    ),
                ),
                RecoveryAction(
                    priority="LONG_TERM",
                    action=(
                        "Deploy periodic satellite and drone "
                        "monitoring with alerts for repeated "
                        "clearing."
                    ),
                    reason=(
                        "Repeated observation supports early "
                        "detection and enforcement."
                    ),
                ),
            ]
        )

        if (
            "Mining"
            in likely_cause
            or "quarry"
            in likely_cause.lower()
        ):
            actions.append(
                RecoveryAction(
                    priority="SHORT_TERM",
                    action=(
                        "Restore topsoil, stabilize exposed "
                        "slopes and control sediment runoff "
                        "before plantation."
                    ),
                    reason=(
                        "Extraction zones require soil and slope "
                        "repair before vegetation can survive."
                    ),
                )
            )

    elif cause_type == "CLIMATE":
        actions.extend(
            [
                RecoveryAction(
                    priority="IMMEDIATE",
                    action=(
                        "Increase canopy-moisture, fire-weather "
                        "and thermal-stress monitoring."
                    ),
                    reason=(
                        "Climate-stressed forest can deteriorate "
                        "rapidly or become fire-prone."
                    ),
                ),
                RecoveryAction(
                    priority="SHORT_TERM",
                    action=(
                        "Implement contour trenches, check dams, "
                        "mulching and other soil-moisture "
                        "conservation measures where suitable."
                    ),
                    reason=(
                        "Water retention reduces drought stress "
                        "and supports regeneration."
                    ),
                ),
                RecoveryAction(
                    priority="LONG_TERM",
                    action=(
                        "Restore climate-resilient native species "
                        "and maintain fire lines in vulnerable "
                        "zones."
                    ),
                    reason=(
                        "Diverse native species improve long-term "
                        "resilience to heat, drought and fire."
                    ),
                ),
            ]
        )

    elif cause_type == "MIXED":
        actions.extend(
            [
                RecoveryAction(
                    priority="IMMEDIATE",
                    action=(
                        "Combine access control and patrols with "
                        "fire and moisture monitoring."
                    ),
                    reason=(
                        "Mixed zones require simultaneous control "
                        "of human disturbance and climate risk."
                    ),
                ),
                RecoveryAction(
                    priority="SHORT_TERM",
                    action=(
                        "Use assisted regeneration, erosion "
                        "control and water-retention structures."
                    ),
                    reason=(
                        "Integrated ecological repair addresses "
                        "both land damage and moisture stress."
                    ),
                ),
                RecoveryAction(
                    priority="LONG_TERM",
                    action=(
                        "Create a community-supported restoration "
                        "and monitoring programme."
                    ),
                    reason=(
                        "Local participation reduces recurring "
                        "human pressure and supports maintenance."
                    ),
                ),
            ]
        )

    else:
        actions.extend(
            [
                RecoveryAction(
                    priority="IMMEDIATE",
                    action=(
                        "Conduct field verification before "
                        "assigning responsibility or beginning "
                        "major intervention."
                    ),
                    reason=(
                        "Current evidence is insufficient for "
                        "confident cause attribution."
                    ),
                ),
                RecoveryAction(
                    priority="SHORT_TERM",
                    action=(
                        "Add rainfall, temperature, active-fire "
                        "and land-use permit data."
                    ),
                    reason=(
                        "Additional evidence can distinguish "
                        "climate stress from human disturbance."
                    ),
                ),
            ]
        )

    return actions


# ============================================================
# ZONE ANALYSIS
# ============================================================

async def analyze_single_zone(
    cell: dict,
    baseline_start: date,
    baseline_end: date,
    current_start: date,
    current_end: date,
    include_human_pressure: bool,
    client: httpx.AsyncClient,
    token: str,
) -> ForestZoneResult:
    baseline_task = fetch_forest_statistics(
        bbox=cell["bbox"],
        center_latitude=cell["latitude"],
        start_date=baseline_start,
        end_date=baseline_end,
        client=client,
        token=token,
    )

    current_task = fetch_forest_statistics(
        bbox=cell["bbox"],
        center_latitude=cell["latitude"],
        start_date=current_start,
        end_date=current_end,
        client=client,
        token=token,
    )

    climate_task = fetch_climate_comparison(
        latitude=cell["latitude"],
        longitude=cell["longitude"],
        baseline_start=baseline_start,
        baseline_end=baseline_end,
        current_start=current_start,
        current_end=current_end,
        client=client,
    )

    if include_human_pressure:
        pressure_task = fetch_human_pressure(
            latitude=cell["latitude"],
            longitude=cell["longitude"],
            radius_km=3,
            client=client,
        )
    else:
        pressure_task = asyncio.sleep(
            0,
            result={
                "status": "SKIPPED",
                "feature_count": 0,
                "nearest_road_km": None,
                "nearest_settlement_km": None,
                "nearest_built_up_land_km": None,
                "nearest_quarry_or_mining_km": None,
                "pressure_score": 0,
                "sample_features": [],
                "errors": [],
            },
        )

    (
        baseline,
        current,
        pressure,
        climate,
    ) = await asyncio.gather(
        baseline_task,
        current_task,
        pressure_task,
        climate_task,
    )

    ndvi_change = difference(
        current.get("ndvi"),
        baseline.get("ndvi"),
    )

    ndmi_change = difference(
        current.get("ndmi"),
        baseline.get("ndmi"),
    )

    ndbi_change = difference(
        current.get("ndbi"),
        baseline.get("ndbi"),
    )

    nbr_change = difference(
        current.get("nbr"),
        baseline.get("nbr"),
    )

    forest_change_percent = difference(
        current.get("forest_percent"),
        baseline.get("forest_percent"),
    )

    health_score = calculate_forest_health_score(
        ndvi_current=current.get("ndvi"),
        ndvi_change=ndvi_change,
        ndmi_change=ndmi_change,
        forest_change_percent=(
            forest_change_percent
        ),
        nbr_change=nbr_change,
    )

    severity, map_color = classify_severity(
        health_score=health_score,
        ndvi_change=ndvi_change,
        forest_change_percent=(
            forest_change_percent
        ),
    )

    cause = classify_cause(
        ndvi_change=ndvi_change,
        ndmi_change=ndmi_change,
        ndbi_change=ndbi_change,
        nbr_change=nbr_change,
        forest_change_percent=(
            forest_change_percent
        ),
        pressure=pressure,
        climate=climate,
        severity=severity,
    )

    observed_evidence = list(
        cause["evidence"]
    )

    if current.get("ndvi") is not None:
        observed_evidence.insert(
            0,
            (
                "Current mean NDVI is "
                f"{current['ndvi']:.3f}."
            ),
        )

    data_confidence = calculate_data_confidence(
        current,
        baseline,
    )

    return ForestZoneResult(
        zone_id=cell["zone_id"],
        latitude=cell["latitude"],
        longitude=cell["longitude"],
        bbox=cell["bbox"],
        severity=severity,
        map_color=map_color,
        forest_health_score=health_score,
        ndvi_baseline=safe_round(
            baseline.get("ndvi")
        ),
        ndvi_current=safe_round(
            current.get("ndvi")
        ),
        ndvi_change=safe_round(
            ndvi_change
        ),
        ndmi_baseline=safe_round(
            baseline.get("ndmi")
        ),
        ndmi_current=safe_round(
            current.get("ndmi")
        ),
        ndmi_change=safe_round(
            ndmi_change
        ),
        ndbi_baseline=safe_round(
            baseline.get("ndbi")
        ),
        ndbi_current=safe_round(
            current.get("ndbi")
        ),
        ndbi_change=safe_round(
            ndbi_change
        ),
        nbr_baseline=safe_round(
            baseline.get("nbr")
        ),
        nbr_current=safe_round(
            current.get("nbr")
        ),
        nbr_change=safe_round(
            nbr_change
        ),
        forest_cover_baseline_percent=(
            safe_round(
                baseline.get(
                    "forest_percent"
                ),
                2,
            )
        ),
        forest_cover_current_percent=(
            safe_round(
                current.get(
                    "forest_percent"
                ),
                2,
            )
        ),
        forest_cover_change_percent=(
            safe_round(
                forest_change_percent,
                2,
            )
        ),
        cloud_ratio_percent=safe_round(
            current.get(
                "cloud_percent"
            ),
            2,
        ),
        data_confidence_percent=(
            data_confidence
        ),
        cause_type=cause["cause_type"],
        likely_cause=cause[
            "likely_cause"
        ],
        cause_confidence_percent=cause[
            "confidence"
        ],
        human_pressure_evidence=pressure,
        climate_evidence=climate,
        attribution_quality=cause[
            "attribution_quality"
        ],
        observed_evidence=observed_evidence,
        cause_explanation=cause[
            "explanation"
        ],
        recovery_actions=(
            generate_recovery_actions(
                cause_type=cause[
                    "cause_type"
                ],
                likely_cause=cause[
                    "likely_cause"
                ],
                severity=severity,
            )
        ),
    )


def build_overall_summary(
    zones: list[ForestZoneResult],
    dominant_cause: CauseType,
) -> list[str]:
    if not zones:
        return [
            "No valid forest zones were returned."
        ]

    average_health = sum(
        zone.forest_health_score
        for zone in zones
    ) / len(zones)

    critical_count = sum(
        zone.severity
        == "CRITICAL_DEGRADATION"
        for zone in zones
    )

    high_count = sum(
        zone.severity
        == "HIGH_DEGRADATION"
        for zone in zones
    )

    summary = [
        (
            "Average forest health score across "
            f"{len(zones)} zones is "
            f"{average_health:.1f}/100."
        ),
        (
            f"{critical_count} critical and "
            f"{high_count} high-degradation zones "
            "were identified."
        ),
        (
            "The dominant preliminary cause class is "
            f"{dominant_cause}."
        ),
    ]

    if dominant_cause == "HUMAN":
        summary.append(
            "Human-pressure evidence should be verified "
            "using field inspection and current land-use "
            "records before enforcement."
        )

    elif dominant_cause == "CLIMATE":
        summary.append(
            "Rainfall, temperature and active-fire datasets "
            "should be added to confirm the climate mechanism."
        )

    elif dominant_cause == "MIXED":
        summary.append(
            "The area requires both ecological restoration "
            "and control of recurring human disturbance."
        )

    return summary


def dominant_cause_type(
    zones: list[ForestZoneResult],
) -> CauseType:
    if not zones:
        return "UNCERTAIN"

    weighted_counts: dict[
        CauseType,
        float,
    ] = {
        "HUMAN": 0,
        "CLIMATE": 0,
        "MIXED": 0,
        "UNCERTAIN": 0,
    }

    for zone in zones:
        degradation_weight = (
            1
            + (
                100
                - zone.forest_health_score
            ) / 100
        )

        weighted_counts[
            zone.cause_type
        ] += degradation_weight

    return max(
        weighted_counts,
        key=weighted_counts.get,
    )


# ============================================================
# API ENDPOINTS
# ============================================================

@app.get("/")
async def root() -> dict:
    return {
        "success": True,
        "module": (
            "Zeryroot AI Forest Intelligence"
        ),
        "version": "1.1.0",
        "docs": "/docs",
        "status_endpoint": "/forest-status",
        "analysis_endpoint": "/forest-analysis",
    }


@app.get("/forest-status")
async def forest_status() -> dict:
    return {
        "success": True,
        "module": "Forest Intelligence",
        "status": (
            "READY"
            if cdse_configured()
            else "CREDENTIALS_REQUIRED"
        ),
        "provider": forest_provider_status(),
        "capabilities": [
            "Sentinel-2 forest health screening",
            "NDVI vegetation analysis",
            "NDMI canopy moisture analysis",
            "NDBI built-up change analysis",
            "NBR burn and dryness proxy",
            "Historical rainfall and temperature comparison",
            "Green/yellow/orange/red zone mapping",
            "Human/climate/mixed cause classification",
            "Zone-specific recovery recommendations",
        ],
        "important_note": (
            "Cause attribution is preliminary and must be "
            "verified using weather, fire, legal land-use "
            "and field evidence."
        ),
    }


@app.get("/forest-cdse-token-test")
async def forest_cdse_token_test() -> dict:
    async with httpx.AsyncClient() as client:
        token = await get_cdse_access_token(
            client
        )

    return {
        "success": True,
        "provider": (
            "Copernicus Data Space Ecosystem"
        ),
        "authentication": (
            "CLIENT_CREDENTIALS"
        ),
        "token_received": bool(token),
        "token_preview": (
            f"{token[:6]}...{token[-4:]}"
            if len(token) >= 12
            else "received"
        ),
        "secret_exposed": False,
    }


@app.post(
    "/forest-analysis",
    response_model=ForestAnalysisResponse,
)
async def forest_analysis(
    request: ForestAnalysisRequest,
) -> ForestAnalysisResponse:
    (
        baseline_start,
        baseline_end,
        current_start,
        current_end,
    ) = resolve_analysis_periods(request)

    async with httpx.AsyncClient() as client:
        if (
            request.latitude is not None
            and request.longitude is not None
        ):
            center = {
                "latitude": request.latitude,
                "longitude": request.longitude,
                "display_name": request.region,
                "address": {},
                "source": "USER_SUPPLIED_COORDINATES",
            }
        else:
            center = await geocode_region(
                request.region,
                client,
            )

            center["source"] = (
                "OPENSTREETMAP_NOMINATIM"
            )

        token = await get_cdse_access_token(
            client
        )

        cells = generate_grid_cells(
            latitude=center["latitude"],
            longitude=center["longitude"],
            radius_km=request.radius_km,
            grid_size=request.grid_size,
        )

        # Limit simultaneous CDSE requests to avoid provider overload.
        semaphore = asyncio.Semaphore(4)

        async def protected_analysis(
            cell: dict,
        ) -> ForestZoneResult:
            async with semaphore:
                return await analyze_single_zone(
                    cell=cell,
                    baseline_start=baseline_start,
                    baseline_end=baseline_end,
                    current_start=current_start,
                    current_end=current_end,
                    include_human_pressure=(
                        request.include_human_pressure_scan
                    ),
                    client=client,
                    token=token,
                )

        results = await asyncio.gather(
            *[
                protected_analysis(cell)
                for cell in cells
            ],
            return_exceptions=True,
        )

    zones = []
    failed_zones = []

    for cell, result in zip(
        cells,
        results,
    ):
        if isinstance(result, Exception):
            failed_zones.append(
                {
                    "zone_id": cell[
                        "zone_id"
                    ],
                    "error": str(result),
                }
            )
            continue

        zones.append(result)

    if not zones:
        raise HTTPException(
            status_code=502,
            detail={
                "message": (
                    "Forest analysis failed for every "
                    "generated zone."
                ),
                "failed_zones": failed_zones,
            },
        )

    health_scores = [
        zone.forest_health_score
        for zone in zones
    ]

    overall_health = round(
        sum(health_scores)
        / len(health_scores),
        1,
    )

    total_zones = len(zones)

    healthy_count = sum(
        zone.severity == "HEALTHY"
        for zone in zones
    )

    stressed_count = sum(
        zone.severity == "EARLY_STRESS"
        for zone in zones
    )

    degraded_count = sum(
        zone.severity
        in {
            "MODERATE_DEGRADATION",
            "HIGH_DEGRADATION",
        }
        for zone in zones
    )

    critical_count = sum(
        zone.severity
        == "CRITICAL_DEGRADATION"
        for zone in zones
    )

    dominant_cause = dominant_cause_type(
        zones
    )

    next_steps = [
        (
            "Inspect red and orange zones first, beginning "
            "with the lowest forest-health scores."
        ),
        (
            "Validate preliminary cause attribution using "
            "field visits, local forest records and recent "
            "high-resolution imagery."
        ),
        (
            "Repeat the same-season Sentinel-2 comparison "
            "periodically to track whether restoration is "
            "working."
        ),
    ]

    if failed_zones:
        next_steps.append(
            (
                f"{len(failed_zones)} zone(s) failed during "
                "provider analysis and should be retried."
            )
        )

    return ForestAnalysisResponse(
        success=True,
        analysis_type=(
            "sentinel_2_forest_degradation_"
            "and_cause_intelligence"
        ),
        provider=forest_provider_status(),
        region=request.region,
        center=center,
        radius_km=request.radius_km,
        grid_size=request.grid_size,
        baseline_period={
            "start": baseline_start.isoformat(),
            "end": baseline_end.isoformat(),
        },
        current_period={
            "start": current_start.isoformat(),
            "end": current_end.isoformat(),
        },
        overall_forest_health_score=(
            overall_health
        ),
        degradation_detected=any(
            zone.severity != "HEALTHY"
            for zone in zones
        ),
        healthy_zone_percent=round(
            healthy_count
            / total_zones
            * 100,
            2,
        ),
        stressed_zone_percent=round(
            stressed_count
            / total_zones
            * 100,
            2,
        ),
        degraded_zone_percent=round(
            degraded_count
            / total_zones
            * 100,
            2,
        ),
        critical_zone_percent=round(
            critical_count
            / total_zones
            * 100,
            2,
        ),
        dominant_cause_type=dominant_cause,
        zones=zones,
        overall_summary=build_overall_summary(
            zones,
            dominant_cause,
        ),
        recommended_next_steps=next_steps,
        limitations=[
            (
                "Sentinel-2 spectral indicators estimate "
                "vegetation condition; they do not prove "
                "illegal activity or legal responsibility."
            ),
            (
                "Human versus climate attribution is a "
                "rule-based preliminary inference."
            ),
            (
                "Seasonally mismatched dates can create false "
                "degradation signals; compare similar months."
            ),
            (
                "Clouds, shadows, terrain, mixed pixels and "
                "recent harvesting can affect index values."
            ),
            (
                "Weather, active-fire, forest-boundary, "
                "biodiversity and field-survey datasets are "
                "recommended for final decisions."
            ),
        ],
    )
