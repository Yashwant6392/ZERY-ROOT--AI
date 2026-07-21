from __future__ import annotations

import asyncio
import math
import os
from datetime import date, timedelta
from typing import Optional

import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


app = FastAPI(
    title="Zeryroot AI Backend",
    version="2.3.0",
    description=(
        "Geospatial intelligence, infrastructure analysis, "
        "location geocoding, and real GIS feature extraction backend"
    ),
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


NOMINATIM_SEARCH_URL = (
    "https://nominatim.openstreetmap.org/search"
)

OVERPASS_URLS = [
    "https://overpass.kumi.systems/api/interpreter",
    "https://overpass-api.de/api/interpreter",
]

NOMINATIM_HEADERS = {
    "User-Agent": (
        "ZeryrootAI/2.3.0 "
        "(contact: ryan566vani@gmail.com)"
    ),
    "Accept-Language": "en",
}

OVERPASS_HEADERS = {
    "User-Agent": (
        "ZeryrootAI/2.3.0 "
        "(contact: ryan566vani@gmail.com)"
    ),
}

REGION_GIS_CACHE: dict[str, dict] = {}

OVERPASS_REGION_RETRY_ROUNDS = int(
    os.getenv(
        "OVERPASS_REGION_RETRY_ROUNDS",
        "2",
    )
)

SATELLITE_SHORTLIST_PER_REGION = int(
    os.getenv(
        "SATELLITE_SHORTLIST_PER_REGION",
        "2",
    )
)

SATELLITE_LAND_API_URL = os.getenv(
    "SATELLITE_LAND_API_URL",
    "",
).strip()

SATELLITE_LAND_API_KEY = os.getenv(
    "SATELLITE_LAND_API_KEY",
    "",
).strip()

SATELLITE_PROVIDER_NAME = os.getenv(
    "SATELLITE_PROVIDER_NAME",
    "External Satellite Land Service",
).strip()

SATELLITE_REQUEST_TIMEOUT_SECONDS = float(
    os.getenv(
        "SATELLITE_REQUEST_TIMEOUT_SECONDS",
        "25",
    )
)


CDSE_CLIENT_ID = os.getenv(
    "CDSE_CLIENT_ID",
    "",
).strip()

CDSE_CLIENT_SECRET = os.getenv(
    "CDSE_CLIENT_SECRET",
    "",
).strip()

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

CDSE_PROCESS_URL = os.getenv(
    "CDSE_PROCESS_URL",
    "https://sh.dataspace.copernicus.eu/api/v1/process",
).strip()

CDSE_COLLECTION_TYPE = os.getenv(
    "CDSE_COLLECTION_TYPE",
    "sentinel-2-l2a",
).strip()

CDSE_LOOKBACK_DAYS = int(
    os.getenv(
        "CDSE_LOOKBACK_DAYS",
        "180",
    )
)

CDSE_MAX_CLOUD_COVERAGE = float(
    os.getenv(
        "CDSE_MAX_CLOUD_COVERAGE",
        "35",
    )
)

CDSE_TOKEN_CACHE: dict[str, object] = {
    "access_token": None,
    "expires_at": 0.0,
}


class InfraRequest(BaseModel):
    project_type: str
    project_scale: str
    authority: str
    budget: str

    location: str
    coordinates: Optional[str] = None
    area_type: str
    required_land: Optional[str] = None

    ownership: str
    acquisition: str
    sensitive_zone: str
    approval: str

    land_availability: int = Field(ge=0, le=100)
    road_connectivity: int = Field(ge=0, le=100)
    environmental_sensitivity: int = Field(ge=0, le=100)
    disaster_exposure: int = Field(ge=0, le=100)
    population_need: int = Field(ge=0, le=100)

    water: str
    electricity: str
    drainage: str
    beneficiaries: Optional[str] = None

    objective: Optional[str] = None
    requirements: Optional[str] = None


class CompareLocationsRequest(BaseModel):
    project_type: str = Field(
        min_length=2,
        max_length=150,
    )

    candidate_locations: list[str] = Field(
        min_length=2,
        max_length=5,
        description=(
            "Between 2 and 5 candidate Indian locations"
        ),
    )



class LandAvailabilityInput(BaseModel):
    physical_assessment_status: str = Field(
        default="NOT_CONNECTED",
        description=(
            "NOT_CONNECTED, ESTIMATED, VERIFIED_PHYSICAL, "
            "UNSUITABLE, or UNKNOWN"
        ),
    )
    estimated_contiguous_suitable_land_acres: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Estimated physically suitable contiguous land from "
            "satellite/GIS analysis; this is not legal ownership proof."
        ),
    )
    estimated_land_cover_type: Optional[str] = None
    estimated_built_up_ratio_percent: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
    )
    forest_conflict_detected: Optional[bool] = None
    wetland_or_water_conflict_detected: Optional[bool] = None

    official_land_data_available: bool = False
    government_confirmed_land_acres: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Land area confirmed through official government or "
            "authority records."
        ),
    )
    parcel_ids: list[str] = Field(default_factory=list)
    ownership_status: Optional[str] = None
    acquisition_status: Optional[str] = None
    encumbrance_status: Optional[str] = None
    land_use_permission_status: Optional[str] = None
    official_data_source: Optional[str] = None


class SiteRequirements(BaseModel):
    minimum_land_acres: Optional[float] = Field(default=None, gt=0)

    # Legacy field retained for backwards compatibility.
    # Prefer the structured "land" object below.
    available_land_acres: Optional[float] = Field(
        default=None,
        gt=0,
        description=(
            "Legacy user-supplied land figure. It is treated as "
            "unverified unless official land data is also provided."
        ),
    )

    land: LandAvailabilityInput = Field(
        default_factory=LandAvailabilityInput
    )
    maximum_highway_distance_km: Optional[float] = Field(default=None, ge=0)
    maximum_railway_distance_km: Optional[float] = Field(default=None, ge=0)
    maximum_power_distance_km: Optional[float] = Field(default=None, ge=0)
    minimum_settlement_distance_km: Optional[float] = Field(default=None, ge=0)
    minimum_water_distance_km: Optional[float] = Field(default=None, ge=0)
    maximum_water_distance_km: Optional[float] = Field(default=None, ge=0)
    highway_required: bool = True
    railway_required: bool = False
    power_required: bool = True
    industrial_ecosystem_preferred: bool = True
    notes: Optional[str] = None



class SatelliteLandObservation(BaseModel):
    source: str = Field(
        description=(
            "Name of the satellite, remote-sensing, "
            "or government GIS data source"
        )
    )
    status: str = Field(
        default="ESTIMATED",
        description=(
            "ESTIMATED, VERIFIED_PHYSICAL, "
            "UNSUITABLE, or FAILED"
        ),
    )
    region: Optional[str] = None
    zone_id: Optional[str] = None
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
    estimated_contiguous_suitable_land_acres: Optional[float] = Field(
        default=None,
        ge=0,
    )
    dominant_land_cover_type: Optional[str] = None
    built_up_ratio_percent: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
    )
    forest_conflict_detected: Optional[bool] = None
    wetland_or_water_conflict_detected: Optional[bool] = None
    confidence_percent: Optional[float] = Field(
        default=None,
        ge=0,
        le=100,
    )
    observation_date: Optional[str] = None
    methodology: Optional[str] = None
    raw_reference: Optional[str] = None


class SatelliteLandAssessmentRequest(BaseModel):
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)
    radius_km: float = Field(default=5, ge=0.5, le=20)


class AnalyzeSiteRequest(BaseModel):
    project_type: str = Field(min_length=2, max_length=150)
    site_name: str = Field(min_length=2, max_length=200)
    location_query: Optional[str] = Field(default=None, max_length=250)
    expected_district: Optional[str] = Field(default=None, max_length=120)
    expected_state: Optional[str] = Field(default="Uttar Pradesh", max_length=120)
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)
    search_radius_km: float = Field(default=5, ge=1, le=15)
    requirements: SiteRequirements
    project_profile: Optional[DynamicProjectProfile] = None
    satellite_observation: Optional[SatelliteLandObservation] = None
    auto_satellite_analysis: bool = False


class DiscoverSitesRequest(BaseModel):
    project_type: str = Field(min_length=2, max_length=150)
    preferred_regions: list[str] = Field(
        min_length=2,
        max_length=5,
        description="Between 2 and 5 preferred Indian districts or regions",
    )
    expected_state: Optional[str] = Field(default="Uttar Pradesh", max_length=120)
    grid_size: int = Field(default=2, ge=2, le=3)
    screening_radius_km: float = Field(default=5, ge=2, le=10)
    top_zones_per_region: int = Field(default=2, ge=1, le=3)
    requirements: SiteRequirements
    project_scale: Optional[str] = None
    capacity: Optional[str] = None
    raw_materials: list[str] = Field(default_factory=list)
    transport_needs: list[str] = Field(default_factory=list)
    utility_needs: list[str] = Field(default_factory=list)
    safety_requirements: list[str] = Field(default_factory=list)
    special_requirements: list[str] = Field(default_factory=list)
    custom_notes: Optional[str] = None
    project_profile: Optional[DynamicProjectProfile] = None
    satellite_land_observations: list[SatelliteLandObservation] = Field(
        default_factory=list
    )
    auto_satellite_analysis: bool = False


class ProjectCriteriaWeights(BaseModel):
    road_access: float = Field(default=0.20, ge=0, le=1)
    rail_access: float = Field(default=0.10, ge=0, le=1)
    power_access: float = Field(default=0.20, ge=0, le=1)
    water_access: float = Field(default=0.15, ge=0, le=1)
    settlement_safety: float = Field(default=0.15, ge=0, le=1)
    industrial_ecosystem: float = Field(default=0.10, ge=0, le=1)
    data_quality: float = Field(default=0.10, ge=0, le=1)


class ProjectProfileRequest(BaseModel):
    project_type: str = Field(min_length=2, max_length=200)
    project_scale: Optional[str] = Field(default=None, max_length=120)
    capacity: Optional[str] = Field(default=None, max_length=120)
    raw_materials: list[str] = Field(default_factory=list)
    transport_needs: list[str] = Field(default_factory=list)
    utility_needs: list[str] = Field(default_factory=list)
    safety_requirements: list[str] = Field(default_factory=list)
    special_requirements: list[str] = Field(default_factory=list)
    custom_notes: Optional[str] = Field(default=None, max_length=2000)


class DynamicProjectProfile(BaseModel):
    project_family: str
    detected_project_type: str
    profile_source: str
    criteria_weights: ProjectCriteriaWeights
    hard_constraints: list[str] = Field(default_factory=list)
    soft_preferences: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    explanation: list[str] = Field(default_factory=list)


def get_penalty(
    value: str,
    penalties: dict[str, int],
) -> int:
    return penalties.get(value, 0)


def risk_from_value(value: int) -> str:
    if value <= 35:
        return "LOW"

    if value <= 65:
        return "MEDIUM"

    return "HIGH"


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    radius = 6371.0088

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)

    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2) ** 2
        + math.cos(phi1)
        * math.cos(phi2)
        * math.sin(delta_lambda / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a),
    )

    return radius * c


def format_geocode_result(result: dict) -> dict:
    bounding_box = result.get(
        "boundingbox",
        [],
    )

    return {
        "display_name": result.get("display_name"),
        "latitude": float(result["lat"]),
        "longitude": float(result["lon"]),
        "type": result.get("type"),
        "category": result.get("category"),
        "importance": result.get("importance"),
        "address": result.get("address", {}),
        "bounding_box": {
            "south": (
                float(bounding_box[0])
                if len(bounding_box) == 4
                else None
            ),
            "north": (
                float(bounding_box[1])
                if len(bounding_box) == 4
                else None
            ),
            "west": (
                float(bounding_box[2])
                if len(bounding_box) == 4
                else None
            ),
            "east": (
                float(bounding_box[3])
                if len(bounding_box) == 4
                else None
            ),
        },
    }


async def fetch_geocode_results(
    location: str,
    client: httpx.AsyncClient,
    limit: int = 3,
) -> list[dict]:
    cleaned_location = location.strip()

    if len(cleaned_location) < 2:
        return []

    params = {
        "q": cleaned_location,
        "format": "jsonv2",
        "limit": limit,
        "countrycodes": "in",
        "addressdetails": 1,
    }

    response = await client.get(
        NOMINATIM_SEARCH_URL,
        params=params,
        headers=NOMINATIM_HEADERS,
    )

    response.raise_for_status()

    return response.json()


def get_element_coordinates(
    element: dict,
) -> tuple[Optional[float], Optional[float]]:
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


def build_overpass_queries(
    latitude: float,
    longitude: float,
    radius_m: int,
) -> dict[str, str]:
    return {
        "transport": f"""
[out:json][timeout:35];
(
    way["highway"~"motorway|trunk|primary"]
       (around:{radius_m},{latitude},{longitude});
    way["railway"="rail"]
       (around:{radius_m},{latitude},{longitude});
    node["railway"="station"]
       (around:{radius_m},{latitude},{longitude});
);
out center tags;
""",
        "water": f"""
[out:json][timeout:35];
(
    way["waterway"~"river|canal"]
       (around:{radius_m},{latitude},{longitude});
    way["natural"="water"]
       (around:{radius_m},{latitude},{longitude});
    relation["natural"="water"]
       (around:{radius_m},{latitude},{longitude});
    way["landuse"="reservoir"]
       (around:{radius_m},{latitude},{longitude});
);
out center tags;
""",
        "community": f"""
[out:json][timeout:35];
(
    node["place"~"city|town|village"]
       (around:{radius_m},{latitude},{longitude});
    node["amenity"="hospital"]
       (around:{radius_m},{latitude},{longitude});
    node["amenity"="school"]
       (around:{radius_m},{latitude},{longitude});
);
out center tags;
""",
        "industry_power": f"""
[out:json][timeout:35];
(
    way["landuse"="industrial"]
       (around:{radius_m},{latitude},{longitude});
    relation["landuse"="industrial"]
       (around:{radius_m},{latitude},{longitude});
    node["power"~"plant|substation"]
       (around:{radius_m},{latitude},{longitude});
    way["power"~"plant|substation"]
       (around:{radius_m},{latitude},{longitude});
);
out center tags;
""",
    }


async def execute_overpass_query(
    query: str,
    batch_name: str,
    client: httpx.AsyncClient,
) -> tuple[list[dict], dict]:
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
                timeout=20.0,
            )

            response.raise_for_status()
            payload = response.json()
            elements = payload.get("elements", [])

            return elements, {
                "batch": batch_name,
                "status": "SUCCESS",
                "server": overpass_url,
                "element_count": len(elements),
                "errors": errors,
            }

        except httpx.TimeoutException:
            errors.append(
                f"{overpass_url}: timeout after 20 seconds"
            )

        except httpx.HTTPStatusError as exc:
            errors.append(
                f"{overpass_url}: HTTP {exc.response.status_code}"
            )

        except httpx.RequestError as exc:
            errors.append(
                f"{overpass_url}: connection error - {exc}"
            )

        except ValueError:
            errors.append(
                f"{overpass_url}: invalid JSON response"
            )

    return [], {
        "batch": batch_name,
        "status": "FAILED",
        "server": None,
        "element_count": 0,
        "errors": errors,
    }


async def fetch_overpass_elements_batched(
    latitude: float,
    longitude: float,
    radius_m: int,
    client: httpx.AsyncClient,
) -> tuple[list[dict], list[dict]]:
    queries = build_overpass_queries(
        latitude,
        longitude,
        radius_m,
    )

    merged_elements = []
    batch_statuses = []
    seen = set()

    for batch_name, query in queries.items():
        elements, status = await execute_overpass_query(
            query,
            batch_name,
            client,
        )
        batch_statuses.append(status)

        for element in elements:
            key = (
                element.get("type"),
                element.get("id"),
            )
            if key in seen:
                continue

            seen.add(key)
            merged_elements.append(element)

        await asyncio.sleep(0.2)

    return merged_elements, batch_statuses


def identify_feature_category(
    tags: dict,
) -> Optional[str]:
    highway = tags.get("highway")

    if highway in {
        "motorway",
        "trunk",
        "primary",
    }:
        return "major_roads"

    railway = tags.get("railway")

    if railway in {
        "rail",
        "station",
    }:
        return "railways"

    waterway = tags.get("waterway")

    if waterway in {
        "river",
        "canal",
    }:
        return "water_features"

    if tags.get("natural") == "water":
        return "water_features"

    if tags.get("landuse") == "reservoir":
        return "water_features"

    place = tags.get("place")

    if place in {
        "city",
        "town",
        "village",
    }:
        return "settlements"

    if tags.get("amenity") == "hospital":
        return "hospitals"

    if tags.get("amenity") == "school":
        return "schools"

    if tags.get("landuse") == "industrial":
        return "industrial_areas"

    power = tags.get("power")

    if power in {
        "plant",
        "substation",
    }:
        return "power_infrastructure"

    return None


def summarize_gis_features(
    elements: list[dict],
    center_latitude: float,
    center_longitude: float,
) -> dict:
    category_names = [
        "major_roads",
        "railways",
        "water_features",
        "settlements",
        "hospitals",
        "schools",
        "industrial_areas",
        "power_infrastructure",
    ]

    feature_groups = {
        category: []
        for category in category_names
    }

    seen_osm_objects = set()

    for element in elements:
        osm_key = (
            element.get("type"),
            element.get("id"),
        )

        if osm_key in seen_osm_objects:
            continue

        seen_osm_objects.add(osm_key)

        tags = element.get("tags", {})

        category = identify_feature_category(tags)

        if category is None:
            continue

        latitude, longitude = (
            get_element_coordinates(element)
        )

        if latitude is None or longitude is None:
            continue

        distance_km = haversine_distance_km(
            center_latitude,
            center_longitude,
            latitude,
            longitude,
        )

        feature_groups[category].append(
            {
                "osm_type": element.get("type"),
                "osm_id": element.get("id"),
                "name": tags.get(
                    "name",
                    "Unnamed feature",
                ),
                "latitude": latitude,
                "longitude": longitude,
                "distance_km": round(
                    distance_km,
                    3,
                ),
                "tags": {
                    key: value
                    for key, value in tags.items()
                    if key in {
                        "name",
                        "highway",
                        "railway",
                        "waterway",
                        "natural",
                        "landuse",
                        "place",
                        "amenity",
                        "power",
                    }
                },
            }
        )

    summary = {}

    for category, features in feature_groups.items():
        sorted_features = sorted(
            features,
            key=lambda item: item["distance_km"],
        )

        summary[category] = {
            "count": len(sorted_features),
            "nearest": (
                sorted_features[0]
                if sorted_features
                else None
            ),
            "sample_features": sorted_features[:5],
        }

    return summary


async def get_gis_profile(
    latitude: float,
    longitude: float,
    radius_km: float,
    client: httpx.AsyncClient,
) -> dict:
    radius_m = int(radius_km * 1000)

    elements, batch_statuses = await fetch_overpass_elements_batched(
        latitude,
        longitude,
        radius_m,
        client,
    )

    feature_summary = summarize_gis_features(
        elements,
        latitude,
        longitude,
    )

    successful_batches = sum(
        1
        for item in batch_statuses
        if item["status"] == "SUCCESS"
    )

    return {
        "center": {
            "latitude": latitude,
            "longitude": longitude,
        },
        "search_radius_km": radius_km,
        "osm_elements_received": len(elements),
        "features": feature_summary,
        "batch_statuses": batch_statuses,
        "successful_batches": successful_batches,
        "total_batches": len(batch_statuses),
        "partial_results": successful_batches < len(batch_statuses),
        "data_source": "OpenStreetMap data via Overpass API",
    }



def normalize_text(value: Optional[str]) -> str:
    if not value:
        return ""
    return " ".join(value.lower().strip().replace(",", " ").split())


def validate_geocode_match(
    formatted_result: dict,
    expected_district: Optional[str],
    expected_state: Optional[str],
) -> dict:
    address = formatted_result.get("address", {})
    expected_district_n = normalize_text(expected_district)
    expected_state_n = normalize_text(expected_state)

    district_values = [
        normalize_text(address.get("state_district")),
        normalize_text(address.get("county")),
        normalize_text(address.get("district")),
        normalize_text(address.get("city_district")),
    ]
    state_value = normalize_text(address.get("state"))

    district_match = (
        True
        if not expected_district_n
        else any(
            value and (
                expected_district_n in value
                or value in expected_district_n
            )
            for value in district_values
        )
    )

    state_match = (
        True
        if not expected_state_n
        else bool(
            state_value
            and (
                expected_state_n in state_value
                or state_value in expected_state_n
            )
        )
    )

    if district_match and state_match:
        status = "VERIFIED"
        confidence = "HIGH"
    elif state_match and not expected_district_n:
        status = "STATE_VERIFIED"
        confidence = "MEDIUM"
    elif state_match:
        status = "DISTRICT_MISMATCH"
        confidence = "LOW"
    else:
        status = "STATE_MISMATCH"
        confidence = "LOW"

    return {
        "status": status,
        "confidence": confidence,
        "district_match": district_match,
        "state_match": state_match,
        "resolved_address": address,
    }


def nearest_distance(features: dict, category: str) -> Optional[float]:
    nearest = features.get(category, {}).get("nearest")
    return nearest.get("distance_km") if nearest else None


def score_distance(
    distance_km: Optional[float],
    excellent_km: float,
    acceptable_km: float,
    missing_score: int,
) -> int:
    if distance_km is None:
        return missing_score
    if distance_km <= excellent_km:
        return 100
    if distance_km <= acceptable_km:
        ratio = (
            (distance_km - excellent_km)
            / max(acceptable_km - excellent_km, 0.001)
        )
        return round(100 - ratio * 40)
    return max(0, round(60 - (distance_km - acceptable_km) * 8))


def requirement_check(
    name: str,
    actual,
    operator: str,
    required,
    unit: str,
) -> dict:
    if actual is None:
        return {
            "requirement": name,
            "status": "UNKNOWN",
            "actual": None,
            "required": required,
            "unit": unit,
        }

    passed = (
        actual <= required
        if operator == "max"
        else actual >= required
    )

    return {
        "requirement": name,
        "status": "PASS" if passed else "FAIL",
        "actual": actual,
        "required": required,
        "unit": unit,
    }



PROJECT_PROFILE_REGISTRY: list[dict] = [
    {
        "family": "Chemical and Biofuel Processing",
        "keywords": [
            "ethanol", "distillery", "chemical plant",
            "petrochemical", "fertilizer", "refinery",
            "paint plant", "pharmaceutical bulk drug",
        ],
        "weights": {
            "road_access": 0.17,
            "rail_access": 0.10,
            "power_access": 0.18,
            "water_access": 0.20,
            "settlement_safety": 0.20,
            "industrial_ecosystem": 0.10,
            "data_quality": 0.05,
        },
        "hard_constraints": [
            "Adequate settlement safety buffer",
            "Reliable water source",
            "Reliable power supply",
        ],
        "soft_preferences": [
            "Rail freight access",
            "Existing industrial ecosystem",
        ],
    },
    {
        "family": "Heavy Manufacturing",
        "keywords": [
            "steel plant", "cement plant", "foundry",
            "heavy engineering", "metal plant",
            "automobile manufacturing",
        ],
        "weights": {
            "road_access": 0.20,
            "rail_access": 0.18,
            "power_access": 0.20,
            "water_access": 0.12,
            "settlement_safety": 0.12,
            "industrial_ecosystem": 0.13,
            "data_quality": 0.05,
        },
        "hard_constraints": [
            "Heavy-load road access",
            "High-capacity power availability",
        ],
        "soft_preferences": [
            "Rail freight access",
            "Nearby industrial suppliers",
        ],
    },
    {
        "family": "Electronics and Semiconductor Manufacturing",
        "keywords": [
            "semiconductor", "chip fab", "electronics plant",
            "electronics manufacturing", "display fab",
        ],
        "weights": {
            "road_access": 0.14,
            "rail_access": 0.05,
            "power_access": 0.25,
            "water_access": 0.20,
            "settlement_safety": 0.10,
            "industrial_ecosystem": 0.21,
            "data_quality": 0.05,
        },
        "hard_constraints": [
            "High-reliability power supply",
            "High-quality water availability",
        ],
        "soft_preferences": [
            "Electronics supplier ecosystem",
            "Airport and highway connectivity",
        ],
    },
    {
        "family": "Data Infrastructure",
        "keywords": [
            "data center", "cloud campus",
            "server farm", "digital infrastructure",
        ],
        "weights": {
            "road_access": 0.10,
            "rail_access": 0.02,
            "power_access": 0.35,
            "water_access": 0.12,
            "settlement_safety": 0.08,
            "industrial_ecosystem": 0.18,
            "data_quality": 0.15,
        },
        "hard_constraints": [
            "Redundant power availability",
            "Reliable digital connectivity",
        ],
        "soft_preferences": [
            "Low-congestion access",
            "Nearby technical workforce",
        ],
    },
    {
        "family": "Food and Agro Processing",
        "keywords": [
            "food processing", "dairy plant",
            "cold storage", "rice mill",
            "sugar mill", "agro processing",
        ],
        "weights": {
            "road_access": 0.22,
            "rail_access": 0.08,
            "power_access": 0.15,
            "water_access": 0.18,
            "settlement_safety": 0.12,
            "industrial_ecosystem": 0.15,
            "data_quality": 0.10,
        },
        "hard_constraints": [
            "Reliable water availability",
            "Raw-material logistics access",
        ],
        "soft_preferences": [
            "Nearby agricultural supply base",
            "Cold-chain ecosystem",
        ],
    },
    {
        "family": "Textile and Apparel Manufacturing",
        "keywords": [
            "textile plant", "garment factory",
            "spinning mill", "dyeing unit",
            "apparel park",
        ],
        "weights": {
            "road_access": 0.18,
            "rail_access": 0.07,
            "power_access": 0.18,
            "water_access": 0.20,
            "settlement_safety": 0.12,
            "industrial_ecosystem": 0.15,
            "data_quality": 0.10,
        },
        "hard_constraints": [
            "Reliable power",
            "Adequate process water",
        ],
        "soft_preferences": [
            "Labour availability",
            "Existing textile cluster",
        ],
    },
    {
        "family": "Logistics and Warehousing",
        "keywords": [
            "logistics park", "warehouse",
            "freight terminal", "distribution center",
            "industrial logistics",
        ],
        "weights": {
            "road_access": 0.35,
            "rail_access": 0.20,
            "power_access": 0.10,
            "water_access": 0.05,
            "settlement_safety": 0.10,
            "industrial_ecosystem": 0.15,
            "data_quality": 0.05,
        },
        "hard_constraints": [
            "Direct major-road connectivity",
        ],
        "soft_preferences": [
            "Rail freight access",
            "Nearby industrial demand",
        ],
    },
    {
        "family": "Industrial Parks and Integrated Manufacturing Zones",
        "keywords": [
            "industrial park", "manufacturing zone",
            "industrial corridor", "special economic zone",
            "sez", "industrial estate",
        ],
        "weights": {
            "road_access": 0.22,
            "rail_access": 0.13,
            "power_access": 0.18,
            "water_access": 0.12,
            "settlement_safety": 0.12,
            "industrial_ecosystem": 0.18,
            "data_quality": 0.05,
        },
        "hard_constraints": [
            "Large contiguous land requirement",
            "Reliable utility availability",
        ],
        "soft_preferences": [
            "Multi-modal transport access",
            "Expansion potential",
        ],
    },
    {
        "family": "Energy and Utility Infrastructure",
        "keywords": [
            "power plant", "solar park", "wind farm",
            "battery plant", "green hydrogen",
            "waste to energy",
        ],
        "weights": {
            "road_access": 0.15,
            "rail_access": 0.07,
            "power_access": 0.25,
            "water_access": 0.12,
            "settlement_safety": 0.16,
            "industrial_ecosystem": 0.15,
            "data_quality": 0.10,
        },
        "hard_constraints": [
            "Grid connectivity",
            "Safety buffer",
        ],
        "soft_preferences": [
            "Existing utility corridor",
            "Low-conflict land",
        ],
    },
]


def normalize_weights(weights: dict[str, float]) -> dict[str, float]:
    total = sum(max(value, 0) for value in weights.values())

    if total <= 0:
        return {
            "road_access": 0.20,
            "rail_access": 0.10,
            "power_access": 0.20,
            "water_access": 0.15,
            "settlement_safety": 0.15,
            "industrial_ecosystem": 0.10,
            "data_quality": 0.10,
        }

    return {
        key: round(max(value, 0) / total, 4)
        for key, value in weights.items()
    }


def match_project_registry(project_type: str) -> Optional[dict]:
    normalized = normalize_text(project_type)

    for profile in PROJECT_PROFILE_REGISTRY:
        if any(
            keyword in normalized
            for keyword in profile["keywords"]
        ):
            return profile

    return None


def infer_project_profile(
    request: ProjectProfileRequest,
) -> DynamicProjectProfile:
    matched = match_project_registry(
        request.project_type
    )

    if matched:
        family = matched["family"]
        weights = dict(matched["weights"])
        hard_constraints = list(
            matched["hard_constraints"]
        )
        soft_preferences = list(
            matched["soft_preferences"]
        )
        profile_source = "VALIDATED_RULE_REGISTRY"
    else:
        family = "General Industrial Development"
        weights = {
            "road_access": 0.20,
            "rail_access": 0.10,
            "power_access": 0.20,
            "water_access": 0.15,
            "settlement_safety": 0.15,
            "industrial_ecosystem": 0.10,
            "data_quality": 0.10,
        }
        hard_constraints = []
        soft_preferences = []
        profile_source = "RULE_BASED_CUSTOM_INFERENCE"

    combined_text = normalize_text(
        " ".join(
            [
                request.project_type,
                request.project_scale or "",
                request.capacity or "",
                " ".join(request.raw_materials),
                " ".join(request.transport_needs),
                " ".join(request.utility_needs),
                " ".join(request.safety_requirements),
                " ".join(request.special_requirements),
                request.custom_notes or "",
            ]
        )
    )

    if any(
        word in combined_text
        for word in ["water intensive", "high water", "process water"]
    ):
        weights["water_access"] += 0.08
        hard_constraints.append(
            "High process-water availability"
        )

    if any(
        word in combined_text
        for word in ["continuous power", "high power", "uninterrupted power"]
    ):
        weights["power_access"] += 0.08
        hard_constraints.append(
            "High-reliability power availability"
        )

    if any(
        word in combined_text
        for word in ["rail freight", "rail siding", "bulk cargo"]
    ):
        weights["rail_access"] += 0.07
        soft_preferences.append(
            "Rail freight connectivity"
        )

    if any(
        word in combined_text
        for word in ["hazardous", "explosive", "toxic", "chemical storage"]
    ):
        weights["settlement_safety"] += 0.10
        hard_constraints.append(
            "Enhanced settlement safety buffer"
        )

    if any(
        word in combined_text
        for word in ["highway critical", "truck movement", "road freight"]
    ):
        weights["road_access"] += 0.07
        hard_constraints.append(
            "Direct major-road connectivity"
        )

    weights = normalize_weights(weights)

    missing_information = []

    if not request.project_scale:
        missing_information.append("project_scale")
    if not request.capacity:
        missing_information.append("capacity")
    if not request.utility_needs:
        missing_information.append("utility_needs")
    if not request.transport_needs:
        missing_information.append("transport_needs")

    assumptions = [
        "Profile weights are preliminary planning assumptions.",
        "Legal, engineering, environmental, and statutory standards must be verified separately.",
    ]

    explanation = [
        f"Project classified under: {family}.",
        f"Profile source: {profile_source}.",
        "Weights were generated from the project type and supplied operational requirements.",
    ]

    return DynamicProjectProfile(
        project_family=family,
        detected_project_type=request.project_type,
        profile_source=profile_source,
        criteria_weights=ProjectCriteriaWeights(
            **weights
        ),
        hard_constraints=list(
            dict.fromkeys(hard_constraints)
        ),
        soft_preferences=list(
            dict.fromkeys(soft_preferences)
        ),
        assumptions=assumptions,
        missing_information=missing_information,
        explanation=explanation,
    )


def resolve_project_profile(
    project_type: str,
    project_profile: Optional[DynamicProjectProfile],
    project_scale: Optional[str] = None,
    capacity: Optional[str] = None,
    raw_materials: Optional[list[str]] = None,
    transport_needs: Optional[list[str]] = None,
    utility_needs: Optional[list[str]] = None,
    safety_requirements: Optional[list[str]] = None,
    special_requirements: Optional[list[str]] = None,
    custom_notes: Optional[str] = None,
) -> DynamicProjectProfile:
    if project_profile is not None:
        normalized = normalize_weights(
            project_profile.criteria_weights.model_dump()
        )
        return project_profile.model_copy(
            update={
                "criteria_weights": ProjectCriteriaWeights(
                    **normalized
                )
            }
        )

    return infer_project_profile(
        ProjectProfileRequest(
            project_type=project_type,
            project_scale=project_scale,
            capacity=capacity,
            raw_materials=raw_materials or [],
            transport_needs=transport_needs or [],
            utility_needs=utility_needs or [],
            safety_requirements=safety_requirements or [],
            special_requirements=special_requirements or [],
            custom_notes=custom_notes,
        )
    )



def cdse_configured() -> bool:
    return bool(
        CDSE_CLIENT_ID
        and CDSE_CLIENT_SECRET
        and CDSE_TOKEN_URL
        and CDSE_STATISTICS_URL
    )


def satellite_provider_configured() -> bool:
    return bool(
        cdse_configured()
        or SATELLITE_LAND_API_URL
    )


def satellite_provider_status() -> dict:
    if cdse_configured():
        mode = "CDSE_SENTINEL_HUB"
        provider_name = (
            "Copernicus Data Space Ecosystem "
            "Sentinel Hub"
        )
    elif SATELLITE_LAND_API_URL:
        mode = "EXTERNAL_PROVIDER"
        provider_name = SATELLITE_PROVIDER_NAME
    else:
        mode = "NOT_CONFIGURED"
        provider_name = SATELLITE_PROVIDER_NAME

    return {
        "configured": satellite_provider_configured(),
        "provider_name": provider_name,
        "mode": mode,
        "cdse_client_id_configured": bool(
            CDSE_CLIENT_ID
        ),
        "cdse_client_secret_configured": bool(
            CDSE_CLIENT_SECRET
        ),
        "cdse_token_url_configured": bool(
            CDSE_TOKEN_URL
        ),
        "cdse_statistics_url_configured": bool(
            CDSE_STATISTICS_URL
        ),
        "generic_api_url_configured": bool(
            SATELLITE_LAND_API_URL
        ),
        "generic_api_key_configured": bool(
            SATELLITE_LAND_API_KEY
        ),
        "collection": (
            CDSE_COLLECTION_TYPE
            if cdse_configured()
            else None
        ),
        "lookback_days": (
            CDSE_LOOKBACK_DAYS
            if cdse_configured()
            else None
        ),
        "fake_results_enabled": False,
    }


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


def bbox_area_acres(
    bbox: list[float],
    latitude: float,
) -> float:
    west, south, east, north = bbox

    width_km = (
        abs(east - west)
        * 111.32
        * max(
            math.cos(math.radians(latitude)),
            0.1,
        )
    )
    height_km = abs(north - south) * 111.32
    square_km = width_km * height_km

    return square_km * 247.105381


async def get_cdse_access_token(
    client: httpx.AsyncClient,
) -> str:
    if not cdse_configured():
        raise HTTPException(
            status_code=503,
            detail=(
                "CDSE credentials are not configured in "
                "the current backend terminal."
            ),
        )

    import time

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
            detail="CDSE OAuth token request timed out.",
        ) from exc

    except httpx.HTTPStatusError as exc:
        safe_detail = (
            "CDSE OAuth authentication failed with "
            f"HTTP {exc.response.status_code}. Check the "
            "client ID/secret and restart Uvicorn from the "
            "same terminal where variables were set."
        )
        raise HTTPException(
            status_code=502,
            detail=safe_detail,
        ) from exc

    except (httpx.RequestError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to obtain a valid CDSE OAuth token."
            ),
        ) from exc

    access_token = payload.get("access_token")
    expires_in = int(payload.get("expires_in", 300))

    if not access_token:
        raise HTTPException(
            status_code=502,
            detail=(
                "CDSE token response did not include "
                "an access token."
            ),
        )

    CDSE_TOKEN_CACHE["access_token"] = access_token
    CDSE_TOKEN_CACHE["expires_at"] = (
        time.time() + expires_in
    )

    return access_token


def build_cdse_evalscript() -> str:
    return r"""
//VERSION=3
function setup() {
  return {
    input: [{
      bands: [
        "B02", "B03", "B04",
        "B08", "B11", "SCL", "dataMask"
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
      { id: "valid", bands: 1 },
      { id: "water", bands: 1 },
      { id: "vegetation", bands: 1 },
      { id: "built_up", bands: 1 },
      { id: "bare_open", bands: 1 },
      { id: "cloud", bands: 1 },
      { id: "dataMask", bands: 1 }
    ]
  };
}

function evaluatePixel(sample) {
  const denominatorNdvi = sample.B08 + sample.B04;
  const denominatorNdwi = sample.B03 + sample.B08;
  const denominatorNdbi = sample.B11 + sample.B08;

  const ndvi = denominatorNdvi === 0
    ? 0
    : (sample.B08 - sample.B04) / denominatorNdvi;

  const ndwi = denominatorNdwi === 0
    ? 0
    : (sample.B03 - sample.B08) / denominatorNdwi;

  const ndbi = denominatorNdbi === 0
    ? 0
    : (sample.B11 - sample.B08) / denominatorNdbi;

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
    isCloud === 0
  ) ? 1 : 0;

  const water = (
    valid === 1 &&
    (scl === 6 || ndwi > 0.20)
  ) ? 1 : 0;

  const vegetation = (
    valid === 1 &&
    water === 0 &&
    ndvi > 0.42
  ) ? 1 : 0;

  const builtUp = (
    valid === 1 &&
    water === 0 &&
    vegetation === 0 &&
    ndbi > 0.08 &&
    ndvi < 0.35
  ) ? 1 : 0;

  const bareOpen = (
    valid === 1 &&
    water === 0 &&
    vegetation === 0 &&
    builtUp === 0 &&
    ndvi >= 0.05 &&
    ndvi <= 0.42
  ) ? 1 : 0;

  return {
    valid: [valid],
    water: [water],
    vegetation: [vegetation],
    built_up: [builtUp],
    bare_open: [bareOpen],
    cloud: [isCloud],
    dataMask: [sample.dataMask]
  };
}
"""


def extract_statistical_mean(
    interval_output: dict,
    output_name: str,
) -> Optional[float]:
    output = (
        interval_output
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


async def fetch_cdse_land_observation(
    latitude: float,
    longitude: float,
    radius_km: float,
    client: httpx.AsyncClient,
) -> SatelliteLandObservation:
    token = await get_cdse_access_token(client)

    bbox = build_bbox_from_radius(
        latitude,
        longitude,
        radius_km,
    )

    end_date = date.today()
    start_date = end_date - timedelta(
        days=CDSE_LOOKBACK_DAYS
    )

    # The Statistical API interprets resx/resy in the
    # units of the selected CRS. Because this request uses
    # CRS84, the resolution must be expressed in degrees,
    # not metres. These values approximate a 20 m pixel.
    latitude_resolution_degrees = (
        20.0 / 111_320.0
    )
    longitude_resolution_degrees = (
        20.0
        / (
            111_320.0
            * max(
                math.cos(math.radians(latitude)),
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
                    "type": CDSE_COLLECTION_TYPE,
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
            "evalscript": build_cdse_evalscript(),
        },
        "calculations": {
            name: {
                "statistics": {
                    "default": {
                        "percentiles": {
                            "k": [50]
                        }
                    }
                }
            }
            for name in [
                "valid",
                "water",
                "vegetation",
                "built_up",
                "bare_open",
                "cloud",
            ]
        },
    }

    try:
        response = await client.post(
            CDSE_STATISTICS_URL,
            json=payload,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            timeout=SATELLITE_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        result = response.json()

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                "CDSE Sentinel-2 land analysis timed out."
            ),
        ) from exc

    except httpx.HTTPStatusError as exc:
        detail = (
            "CDSE Statistical API returned "
            f"HTTP {exc.response.status_code}."
        )
        try:
            error_payload = exc.response.json()
            api_message = (
                error_payload.get("error", {})
                .get("message")
                or error_payload.get("message")
            )
            if api_message:
                detail += f" {api_message}"
        except ValueError:
            pass

        raise HTTPException(
            status_code=502,
            detail=detail,
        ) from exc

    except (httpx.RequestError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to process a valid CDSE "
                "Sentinel-2 statistical response."
            ),
        ) from exc

    intervals = result.get("data", [])
    usable_intervals = []

    for interval in intervals:
        outputs = interval.get("outputs", {})
        valid_mean = extract_statistical_mean(
            interval,
            "valid",
        )

        if (
            valid_mean is not None
            and valid_mean > 0.20
            and outputs
        ):
            usable_intervals.append(
                (
                    valid_mean,
                    interval,
                )
            )

    if not usable_intervals:
        raise HTTPException(
            status_code=404,
            detail=(
                "No sufficiently clear Sentinel-2 "
                "observation was available for this area "
                "and date range."
            ),
        )

    usable_intervals.sort(
        key=lambda item: item[0],
        reverse=True,
    )
    valid_mean, best_interval = usable_intervals[0]

    def ratio(output_name: str) -> float:
        value = extract_statistical_mean(
            best_interval,
            output_name,
        )
        if value is None:
            return 0.0
        return max(
            0.0,
            min(
                1.0,
                value / max(valid_mean, 0.0001),
            ),
        )

    water_ratio = ratio("water")
    vegetation_ratio = ratio("vegetation")
    built_ratio = ratio("built_up")
    bare_ratio = ratio("bare_open")
    cloud_mean = (
        extract_statistical_mean(
            best_interval,
            "cloud",
        )
        or 0.0
    )

    # Conservative physical-suitability proxy.
    # Bare/open land is counted fully and only a small
    # fraction of low-density vegetation is counted.
    potential_ratio = max(
        0.0,
        min(
            1.0,
            bare_ratio
            + vegetation_ratio * 0.15,
        ),
    )

    analysis_area_acres = bbox_area_acres(
        bbox,
        latitude,
    )

    # A conservative contiguity discount avoids treating
    # every suitable pixel as one legally available parcel.
    contiguity_factor = 0.55
    estimated_acres = (
        analysis_area_acres
        * potential_ratio
        * contiguity_factor
    )

    dominant_name, dominant_ratio = max(
        {
            "Water": water_ratio,
            "Vegetation / Cropland": (
                vegetation_ratio
            ),
            "Built-up": built_ratio,
            "Bare / Open Land": bare_ratio,
        }.items(),
        key=lambda item: item[1],
    )

    clear_ratio = max(
        0.0,
        min(
            1.0,
            valid_mean,
        ),
    )
    confidence = round(
        (
            clear_ratio * 70
            + min(
                1.0,
                len(usable_intervals) / 3,
            ) * 20
            + (
                1.0
                - min(
                    1.0,
                    cloud_mean,
                )
            ) * 10
        ),
        1,
    )

    interval_info = best_interval.get(
        "interval",
        {}
    )

    wetland_conflict = (
        water_ratio >= 0.08
    )
    forest_conflict = (
        vegetation_ratio >= 0.75
        and bare_ratio < 0.08
    )

    return SatelliteLandObservation(
        source=(
            "Copernicus Data Space Ecosystem "
            "Sentinel-2 L2A"
        ),
        status="ESTIMATED",
        latitude=latitude,
        longitude=longitude,
        estimated_contiguous_suitable_land_acres=round(
            estimated_acres,
            2,
        ),
        dominant_land_cover_type=(
            f"{dominant_name} "
            f"({dominant_ratio * 100:.1f}%)"
        ),
        built_up_ratio_percent=round(
            built_ratio * 100,
            2,
        ),
        forest_conflict_detected=(
            forest_conflict
        ),
        wetland_or_water_conflict_detected=(
            wetland_conflict
        ),
        confidence_percent=confidence,
        observation_date=(
            interval_info.get("to")
            or end_date.isoformat()
        ),
        methodology=(
            "Sentinel-2 L2A Statistical API; "
            "SCL cloud masking plus NDVI, NDWI and "
            "NDBI spectral proxies. Estimated acreage "
            "includes a conservative contiguity discount "
            "and is not a cadastral or legal land record."
        ),
        raw_reference=(
            f"bbox={bbox}; "
            f"analysis_area_acres="
            f"{analysis_area_acres:.2f}; "
            f"bare_open={bare_ratio * 100:.2f}%; "
            f"vegetation={vegetation_ratio * 100:.2f}%; "
            f"water={water_ratio * 100:.2f}%; "
            f"built_up={built_ratio * 100:.2f}%"
        ),
    )


async def fetch_generic_satellite_land_observation(
    latitude: float,
    longitude: float,
    radius_km: float,
    client: httpx.AsyncClient,
) -> SatelliteLandObservation:
    if not SATELLITE_LAND_API_URL:
        raise HTTPException(
            status_code=503,
            detail=(
                "Generic satellite provider is not configured."
            ),
        )

    headers = {
        "Accept": "application/json",
        "User-Agent": (
            "ZeryrootAI/2.3.0 "
            "(satellite-land-adapter)"
        ),
    }

    if SATELLITE_LAND_API_KEY:
        headers["Authorization"] = (
            f"Bearer {SATELLITE_LAND_API_KEY}"
        )
        headers["X-API-Key"] = (
            SATELLITE_LAND_API_KEY
        )

    try:
        response = await client.get(
            SATELLITE_LAND_API_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "radius_km": radius_km,
            },
            headers=headers,
            timeout=SATELLITE_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="Satellite land provider timed out.",
        ) from exc

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Satellite land provider returned "
                f"HTTP {exc.response.status_code}."
            ),
        ) from exc

    except (httpx.RequestError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to obtain a valid generic "
                "satellite response."
            ),
        ) from exc

    if isinstance(payload, dict):
        if isinstance(
            payload.get("observation"),
            dict,
        ):
            payload = payload["observation"]
        elif isinstance(
            payload.get("data"),
            dict,
        ):
            payload = payload["data"]

    if not isinstance(payload, dict):
        raise HTTPException(
            status_code=502,
            detail=(
                "Generic satellite response does not "
                "contain a valid observation object."
            ),
        )

    normalized = {
        "source": payload.get(
            "source",
            SATELLITE_PROVIDER_NAME,
        ),
        "status": payload.get(
            "status",
            "ESTIMATED",
        ),
        "latitude": payload.get(
            "latitude",
            latitude,
        ),
        "longitude": payload.get(
            "longitude",
            longitude,
        ),
        "estimated_contiguous_suitable_land_acres": (
            payload.get(
                "estimated_contiguous_suitable_land_acres"
            )
        ),
        "dominant_land_cover_type": payload.get(
            "dominant_land_cover_type",
            payload.get("land_cover_type"),
        ),
        "built_up_ratio_percent": payload.get(
            "built_up_ratio_percent"
        ),
        "forest_conflict_detected": payload.get(
            "forest_conflict_detected"
        ),
        "wetland_or_water_conflict_detected": (
            payload.get(
                "wetland_or_water_conflict_detected"
            )
        ),
        "confidence_percent": payload.get(
            "confidence_percent"
        ),
        "observation_date": payload.get(
            "observation_date"
        ),
        "methodology": payload.get(
            "methodology"
        ),
        "raw_reference": payload.get(
            "raw_reference"
        ),
    }

    try:
        return SatelliteLandObservation(
            **normalized
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Generic satellite response could not "
                "be validated."
            ),
        ) from exc


async def fetch_satellite_land_observation(
    latitude: float,
    longitude: float,
    radius_km: float,
    client: httpx.AsyncClient,
) -> SatelliteLandObservation:
    if cdse_configured():
        return await fetch_cdse_land_observation(
            latitude,
            longitude,
            radius_km,
            client,
        )

    if SATELLITE_LAND_API_URL:
        return await fetch_generic_satellite_land_observation(
            latitude,
            longitude,
            radius_km,
            client,
        )

    raise HTTPException(
        status_code=503,
        detail=(
            "No satellite provider is configured. "
            "Set CDSE_CLIENT_ID and CDSE_CLIENT_SECRET "
            "or submit a trusted manual observation."
        ),
    )


def apply_satellite_observation_to_requirements(
    requirements: SiteRequirements,
    observation: Optional[
        SatelliteLandObservation
    ],
) -> SiteRequirements:
    if observation is None:
        return requirements

    current_land = requirements.land

    updated_land = current_land.model_copy(
        update={
            "physical_assessment_status": (
                observation.status
            ),
            "estimated_contiguous_suitable_land_acres": (
                observation
                .estimated_contiguous_suitable_land_acres
            ),
            "estimated_land_cover_type": (
                observation.dominant_land_cover_type
            ),
            "estimated_built_up_ratio_percent": (
                observation.built_up_ratio_percent
            ),
            "forest_conflict_detected": (
                observation.forest_conflict_detected
            ),
            "wetland_or_water_conflict_detected": (
                observation
                .wetland_or_water_conflict_detected
            ),
        }
    )

    return requirements.model_copy(
        update={
            "land": updated_land,
        }
    )


def find_supplied_satellite_observation(
    observations: list[
        SatelliteLandObservation
    ],
    region_name: str,
    zone_id: str,
    latitude: float,
    longitude: float,
) -> Optional[SatelliteLandObservation]:
    for observation in observations:
        if (
            observation.region
            and observation.zone_id
            and normalize_text(observation.region)
            == normalize_text(region_name)
            and observation.zone_id.upper()
            == zone_id.upper()
        ):
            return observation

    best = None
    best_distance = float("inf")

    for observation in observations:
        if (
            observation.latitude is None
            or observation.longitude is None
        ):
            continue

        distance = haversine_distance_km(
            latitude,
            longitude,
            observation.latitude,
            observation.longitude,
        )

        if distance < best_distance:
            best_distance = distance
            best = observation

    if best is not None and best_distance <= 0.5:
        return best

    return None


def build_land_assessment(
    requirements: SiteRequirements,
) -> dict:
    minimum_required = requirements.minimum_land_acres
    land = requirements.land

    physical_estimate = (
        land.estimated_contiguous_suitable_land_acres
    )
    official_confirmed = (
        land.government_confirmed_land_acres
    )

    physical_status = (
        land.physical_assessment_status
        or "NOT_CONNECTED"
    ).upper()

    if (
        physical_status == "UNSUITABLE"
        or land.forest_conflict_detected is True
        or land.wetland_or_water_conflict_detected is True
    ):
        physical_decision = "PHYSICALLY_UNSUITABLE"
    elif (
        minimum_required is not None
        and physical_estimate is not None
        and physical_estimate < minimum_required
    ):
        physical_decision = "INSUFFICIENT_PHYSICAL_LAND"
    elif physical_estimate is not None:
        physical_decision = "POTENTIALLY_SUITABLE"
    elif physical_status in {
        "ESTIMATED",
        "VERIFIED_PHYSICAL",
    }:
        physical_decision = "PHYSICAL_AREA_NOT_REPORTED"
    else:
        physical_decision = "NOT_YET_ASSESSED"

    official_available = (
        land.official_land_data_available
        or official_confirmed is not None
        or bool(land.parcel_ids)
    )

    if not official_available:
        legal_decision = "NOT_PROVIDED"
    elif (
        minimum_required is not None
        and official_confirmed is not None
        and official_confirmed < minimum_required
    ):
        legal_decision = "OFFICIALLY_INSUFFICIENT"
    elif (
        minimum_required is not None
        and official_confirmed is not None
        and official_confirmed >= minimum_required
    ):
        legal_decision = "OFFICIALLY_SUFFICIENT"
    else:
        legal_decision = "OFFICIAL_DATA_INCOMPLETE"

    legal_blockers = []

    for label, value in {
        "ownership_status": land.ownership_status,
        "acquisition_status": land.acquisition_status,
        "encumbrance_status": land.encumbrance_status,
        "land_use_permission_status": (
            land.land_use_permission_status
        ),
    }.items():
        normalized = normalize_text(value)

        if any(
            token in normalized
            for token in {
                "disputed",
                "blocked",
                "not permitted",
                "rejected",
                "unavailable",
                "encumbered",
            }
        ):
            legal_blockers.append(
                {
                    "field": label,
                    "value": value,
                }
            )

    if legal_blockers:
        legal_decision = "LEGAL_CONSTRAINT_DETECTED"

    if physical_decision in {
        "PHYSICALLY_UNSUITABLE",
        "INSUFFICIENT_PHYSICAL_LAND",
    }:
        land_decision = "NOT_RECOMMENDED"
    elif legal_decision in {
        "OFFICIALLY_INSUFFICIENT",
        "LEGAL_CONSTRAINT_DETECTED",
    }:
        land_decision = "NOT_RECOMMENDED"
    elif (
        physical_decision == "POTENTIALLY_SUITABLE"
        and legal_decision == "OFFICIALLY_SUFFICIENT"
    ):
        land_decision = "LAND_REQUIREMENT_VERIFIED"
    elif legal_decision == "OFFICIALLY_SUFFICIENT":
        land_decision = "PHYSICAL_ASSESSMENT_REQUIRED"
    elif physical_decision == "POTENTIALLY_SUITABLE":
        land_decision = (
            "REQUIRES_GOVERNMENT_VERIFICATION"
        )
    else:
        land_decision = (
            "REQUIRES_PHYSICAL_AND_GOVERNMENT_VERIFICATION"
        )

    return {
        "minimum_required_acres": minimum_required,
        "physical_land_assessment": {
            "status": physical_status,
            "estimated_contiguous_suitable_land_acres": (
                physical_estimate
            ),
            "land_cover_type": (
                land.estimated_land_cover_type
            ),
            "built_up_ratio_percent": (
                land.estimated_built_up_ratio_percent
            ),
            "forest_conflict_detected": (
                land.forest_conflict_detected
            ),
            "wetland_or_water_conflict_detected": (
                land.wetland_or_water_conflict_detected
            ),
            "decision": physical_decision,
            "source_type": (
                "SATELLITE_GIS_ESTIMATE"
                if physical_estimate is not None
                else "NOT_CONNECTED"
            ),
        },
        "legal_land_assessment": {
            "official_land_data_available": (
                official_available
            ),
            "government_confirmed_land_acres": (
                official_confirmed
            ),
            "parcel_ids": land.parcel_ids,
            "ownership_status": land.ownership_status,
            "acquisition_status": (
                land.acquisition_status
            ),
            "encumbrance_status": (
                land.encumbrance_status
            ),
            "land_use_permission_status": (
                land.land_use_permission_status
            ),
            "official_data_source": (
                land.official_data_source
            ),
            "decision": legal_decision,
            "legal_blockers": legal_blockers,
        },
        "land_decision": land_decision,
        "legacy_available_land_acres": (
            requirements.available_land_acres
        ),
        "important_note": (
            "Satellite/GIS can estimate physical suitability, "
            "but only official government or authority records "
            "can confirm legal availability and parcel status."
        ),
    }


def build_site_analysis(
    project_type: str,
    requirements: SiteRequirements,
    gis_profile: dict,
    project_profile: Optional[DynamicProjectProfile] = None,
) -> dict:
    features = gis_profile.get("features", {})

    road = nearest_distance(features, "major_roads")
    rail = nearest_distance(features, "railways")
    water = nearest_distance(features, "water_features")
    settlement = nearest_distance(features, "settlements")
    industry = nearest_distance(features, "industrial_areas")
    power = nearest_distance(features, "power_infrastructure")

    transport_score = round(
        score_distance(road, 1.5, 8, 15) * 0.6
        + score_distance(rail, 3, 15, 30) * 0.4
    )
    utility_score = score_distance(power, 3, 12, 20)
    industrial_score = score_distance(industry, 3, 12, 35)

    if settlement is None:
        public_safety_score = 40
    elif settlement >= 5:
        public_safety_score = 100
    elif settlement >= 2:
        public_safety_score = 75
    elif settlement >= 1:
        public_safety_score = 50
    else:
        public_safety_score = 20

    if water is None:
        water_score = 45
    elif water < 0.5:
        water_score = 30
    elif water <= 5:
        water_score = 85
    elif water <= 15:
        water_score = 65
    else:
        water_score = 35

    data_quality = round(
        gis_profile.get("successful_batches", 0)
        / max(gis_profile.get("total_batches", 1), 1)
        * 100
    )

    checks = []

    land_assessment = build_land_assessment(
        requirements
    )

    if requirements.minimum_land_acres is not None:
        checks.append(
            requirement_check(
                "Physical contiguous land estimate",
                requirements.land
                .estimated_contiguous_suitable_land_acres,
                "min",
                requirements.minimum_land_acres,
                "acres",
            )
        )

        checks.append(
            requirement_check(
                "Government-confirmed legal land availability",
                requirements.land
                .government_confirmed_land_acres,
                "min",
                requirements.minimum_land_acres,
                "acres",
            )
        )

    if requirements.highway_required and requirements.maximum_highway_distance_km is not None:
        checks.append(
            requirement_check(
                "Maximum highway distance",
                road,
                "max",
                requirements.maximum_highway_distance_km,
                "km",
            )
        )

    if requirements.railway_required and requirements.maximum_railway_distance_km is not None:
        checks.append(
            requirement_check(
                "Maximum railway distance",
                rail,
                "max",
                requirements.maximum_railway_distance_km,
                "km",
            )
        )

    if requirements.power_required and requirements.maximum_power_distance_km is not None:
        checks.append(
            requirement_check(
                "Maximum power distance",
                power,
                "max",
                requirements.maximum_power_distance_km,
                "km",
            )
        )

    if requirements.minimum_settlement_distance_km is not None:
        checks.append(
            requirement_check(
                "Minimum settlement distance",
                settlement,
                "min",
                requirements.minimum_settlement_distance_km,
                "km",
            )
        )

    if requirements.minimum_water_distance_km is not None:
        checks.append(
            requirement_check(
                "Minimum water distance",
                water,
                "min",
                requirements.minimum_water_distance_km,
                "km",
            )
        )

    if requirements.maximum_water_distance_km is not None:
        checks.append(
            requirement_check(
                "Maximum water distance",
                water,
                "max",
                requirements.maximum_water_distance_km,
                "km",
            )
        )

    resolved_profile = (
        project_profile
        or resolve_project_profile(
            project_type=project_type,
            project_profile=None,
        )
    )

    profile_weights = (
        resolved_profile.criteria_weights
        .model_dump()
    )

    weights = {
        "road": profile_weights["road_access"],
        "rail": profile_weights["rail_access"],
        "utility": profile_weights["power_access"],
        "water": profile_weights["water_access"],
        "public_safety": (
            profile_weights["settlement_safety"]
        ),
        "industrial": (
            profile_weights["industrial_ecosystem"]
        ),
        "data_quality": (
            profile_weights["data_quality"]
        ),
    }

    components = {
        "transport": transport_score,
        "utility": utility_score,
        "industrial_ecosystem": industrial_score,
        "public_safety": public_safety_score,
        "water_balance": water_score,
        "data_quality": data_quality,
    }

    road_score = score_distance(
        road,
        1.5,
        8,
        15,
    )
    rail_score = score_distance(
        rail,
        3,
        15,
        30,
    )

    overall = round(
        road_score * weights["road"]
        + rail_score * weights["rail"]
        + components["utility"] * weights["utility"]
        + components["industrial_ecosystem"]
        * weights["industrial"]
        + components["public_safety"]
        * weights["public_safety"]
        + components["water_balance"]
        * weights["water"]
        + components["data_quality"]
        * weights["data_quality"]
    )

    failures = [item for item in checks if item["status"] == "FAIL"]
    unknowns = [item for item in checks if item["status"] == "UNKNOWN"]

    overall = max(0, min(100, overall - len(failures) * 12 - len(unknowns) * 3))

    if failures:
        decision = "NOT_RECOMMENDED"
    elif overall >= 75:
        decision = "RECOMMENDED"
    elif overall >= 55:
        decision = "SUITABLE_WITH_CONDITIONS"
    else:
        decision = "NOT_PREFERRED"

    limitations = []
    conditions = [
        "Field survey and official land-record verification are mandatory.",
        "Environmental and statutory clearances must be checked separately.",
        "Cross-module disaster and forest risk inputs are not yet connected; physical land-cover estimation is also pending.",
    ]

    if gis_profile.get("partial_results"):
        limitations.append("The GIS profile contains partial Overpass results.")
    if settlement is None:
        limitations.append("Settlement and public-safety GIS data is incomplete.")
    if power is None:
        limitations.append("Power-infrastructure proximity could not be verified.")
    if water is None:
        limitations.append("Water-feature proximity could not be verified.")

    return {
        "overall_score": overall,
        "decision_status": decision,
        "component_scores": components,
        "weights_used": weights,
        "requirement_checks": checks,
        "hard_constraint_failures": failures,
        "unknown_requirements": unknowns,
        "evidence": {
            "nearest_major_road_km": road,
            "nearest_railway_km": rail,
            "nearest_water_feature_km": water,
            "nearest_settlement_km": settlement,
            "nearest_industrial_area_km": industry,
            "nearest_power_infrastructure_km": power,
        },
        "limitations": limitations,
        "required_conditions": conditions,
        "land_assessment": land_assessment,
        "project_profile": (
            resolved_profile.model_dump()
        ),
        "dynamic_scoring": True,
    }



def generate_grid_points(
    bounding_box: dict,
    grid_size: int,
) -> list[dict]:
    south = bounding_box.get("south")
    north = bounding_box.get("north")
    west = bounding_box.get("west")
    east = bounding_box.get("east")

    if None in {south, north, west, east}:
        return []

    lat_step = (north - south) / (grid_size + 1)
    lon_step = (east - west) / (grid_size + 1)

    points = []
    zone_number = 1

    for row in range(1, grid_size + 1):
        for column in range(1, grid_size + 1):
            points.append(
                {
                    "zone_id": f"Z{zone_number}",
                    "latitude": round(
                        south + lat_step * row,
                        7,
                    ),
                    "longitude": round(
                        west + lon_step * column,
                        7,
                    ),
                }
            )
            zone_number += 1

    return points


def build_region_overpass_query(
    bounding_box: dict,
) -> str:
    south = bounding_box["south"]
    north = bounding_box["north"]
    west = bounding_box["west"]
    east = bounding_box["east"]

    bbox = f"{south},{west},{north},{east}"

    return f"""
[out:json][timeout:35];
(
    way["highway"~"motorway|trunk|primary"]({bbox});
    way["railway"="rail"]({bbox});
    node["railway"="station"]({bbox});
    way["waterway"~"river|canal"]({bbox});
    way["natural"="water"]({bbox});
    node["place"~"city|town|village"]({bbox});
    way["landuse"="industrial"]({bbox});
    node["power"~"plant|substation"]({bbox});
    way["power"~"plant|substation"]({bbox});
);
out center tags 6000;
"""


async def fetch_region_elements(
    region_name: str,
    bounding_box: dict,
    client: httpx.AsyncClient,
) -> tuple[list[dict], dict]:
    cache_key = (
        f"{normalize_text(region_name)}:"
        f"{bounding_box.get('south')}:"
        f"{bounding_box.get('north')}:"
        f"{bounding_box.get('west')}:"
        f"{bounding_box.get('east')}"
    )

    cached = REGION_GIS_CACHE.get(cache_key)

    if cached is not None:
        return cached["elements"], {
            **cached["status"],
            "cache_hit": True,
        }

    query = build_region_overpass_query(
        bounding_box,
    )

    errors: list[str] = []
    attempts = 0

    retry_rounds = max(
        1,
        OVERPASS_REGION_RETRY_ROUNDS,
    )

    for retry_round in range(retry_rounds):
        urls = (
            OVERPASS_URLS
            if retry_round % 2 == 0
            else list(reversed(OVERPASS_URLS))
        )

        for overpass_url in urls:
            attempts += 1

            try:
                response = await client.post(
                    overpass_url,
                    content=query,
                    headers={
                        **OVERPASS_HEADERS,
                        "Content-Type": "text/plain",
                    },
                    timeout=30.0,
                )

                response.raise_for_status()
                payload = response.json()
                elements = payload.get(
                    "elements",
                    [],
                )

                status = {
                    "status": "SUCCESS",
                    "server": overpass_url,
                    "element_count": len(elements),
                    "errors": errors,
                    "cache_hit": False,
                    "attempts": attempts,
                    "retry_round": retry_round + 1,
                    "recovered_after_retry": (
                        retry_round > 0
                    ),
                }

                REGION_GIS_CACHE[cache_key] = {
                    "elements": elements,
                    "status": status,
                }

                return elements, status

            except httpx.TimeoutException:
                errors.append(
                    f"{overpass_url}: timeout after "
                    "30 seconds"
                )

            except httpx.HTTPStatusError as exc:
                errors.append(
                    f"{overpass_url}: HTTP "
                    f"{exc.response.status_code}"
                )

            except httpx.RequestError as exc:
                errors.append(
                    f"{overpass_url}: connection "
                    f"error - {exc}"
                )

            except ValueError:
                errors.append(
                    f"{overpass_url}: invalid JSON "
                    "response"
                )

        if retry_round < retry_rounds - 1:
            await asyncio.sleep(
                1.5 * (retry_round + 1)
            )

    return [], {
        "status": "FAILED",
        "server": None,
        "element_count": 0,
        "errors": errors,
        "cache_hit": False,
        "attempts": attempts,
        "retry_rounds_completed": retry_rounds,
        "recovered_after_retry": False,
    }

def filter_region_elements_for_zone(
    elements: list[dict],
    latitude: float,
    longitude: float,
    radius_km: float,
) -> list[dict]:
    selected = []

    for element in elements:
        element_lat, element_lon = (
            get_element_coordinates(element)
        )

        if (
            element_lat is None
            or element_lon is None
        ):
            continue

        distance = haversine_distance_km(
            latitude,
            longitude,
            element_lat,
            element_lon,
        )

        if distance <= radius_km:
            selected.append(element)

    return selected


def build_local_zone_profile(
    region_elements: list[dict],
    latitude: float,
    longitude: float,
    radius_km: float,
    region_fetch_status: dict,
) -> dict:
    local_elements = filter_region_elements_for_zone(
        region_elements,
        latitude,
        longitude,
        radius_km,
    )

    feature_summary = summarize_gis_features(
        local_elements,
        latitude,
        longitude,
    )

    fetch_success = (
        region_fetch_status.get("status")
        == "SUCCESS"
    )

    return {
        "center": {
            "latitude": latitude,
            "longitude": longitude,
        },
        "search_radius_km": radius_km,
        "osm_elements_received": len(local_elements),
        "region_elements_available": len(
            region_elements
        ),
        "features": feature_summary,
        "batch_statuses": [
            {
                "batch": "region_level_fetch",
                **region_fetch_status,
            }
        ],
        "successful_batches": (
            1 if fetch_success else 0
        ),
        "total_batches": 1,
        "partial_results": not fetch_success,
        "data_source": (
            "OpenStreetMap data via one "
            "region-level Overpass fetch"
        ),
    }


def get_requirement_groups(
    analysis: dict,
) -> tuple[list[dict], list[dict], list[dict]]:
    checks = analysis.get("requirement_checks", [])

    verified = [
        item for item in checks
        if item.get("status") == "PASS"
    ]
    failed = [
        item for item in checks
        if item.get("status") == "FAIL"
    ]
    unverified = [
        item for item in checks
        if item.get("status") == "UNKNOWN"
    ]

    return verified, failed, unverified


def get_mandatory_unknowns(
    analysis: dict,
    requirements: SiteRequirements,
) -> list[dict]:
    unknowns = analysis.get("unknown_requirements", [])
    mandatory_names = set()

    if requirements.minimum_land_acres is not None:
        mandatory_names.add("Physical contiguous land estimate")
        mandatory_names.add("Government-confirmed legal land availability")
    if requirements.highway_required and requirements.maximum_highway_distance_km is not None:
        mandatory_names.add("Maximum highway distance")
    if requirements.railway_required and requirements.maximum_railway_distance_km is not None:
        mandatory_names.add("Maximum railway distance")
    if requirements.power_required and requirements.maximum_power_distance_km is not None:
        mandatory_names.add("Maximum power distance")
    if requirements.minimum_settlement_distance_km is not None:
        mandatory_names.add("Minimum settlement distance")
    if requirements.minimum_water_distance_km is not None:
        mandatory_names.add("Minimum water distance")
    if requirements.maximum_water_distance_km is not None:
        mandatory_names.add("Maximum water distance")

    return [
        item for item in unknowns
        if item.get("requirement") in mandatory_names
    ]


def build_selection_explanation(analysis: dict) -> list[str]:
    evidence = analysis.get("evidence", {})
    checks = analysis.get("requirement_checks", [])
    reasons = []

    for item in checks:
        if item.get("status") == "PASS":
            reasons.append(
                f"{item.get('requirement')} passed with observed value "
                f"{item.get('actual')} {item.get('unit', '')}."
            )

    if evidence.get("nearest_major_road_km") is not None:
        reasons.append(
            f"Mapped major road is about {evidence['nearest_major_road_km']} km away."
        )
    if evidence.get("nearest_railway_km") is not None:
        reasons.append(
            f"Mapped railway feature is about {evidence['nearest_railway_km']} km away."
        )
    if evidence.get("nearest_power_infrastructure_km") is not None:
        reasons.append(
            f"Mapped power infrastructure is about {evidence['nearest_power_infrastructure_km']} km away."
        )
    if evidence.get("nearest_industrial_area_km") is not None:
        reasons.append(
            f"Mapped industrial area is about {evidence['nearest_industrial_area_km']} km away."
        )

    return reasons[:6]


def build_next_verification_steps(
    mandatory_unknowns: list[dict],
    missing_critical: list[str],
    failed_requirements: list[dict],
) -> list[str]:
    steps = []

    for item in mandatory_unknowns:
        name = item.get("requirement", "Unknown requirement")
        if name in {
            "Physical contiguous land estimate",
            "Government-confirmed legal land availability",
        }:
            steps.append(
                "Verify contiguous usable land through official land records, cadastral maps, and satellite land-cover analysis."
            )
        elif name == "Maximum highway distance":
            steps.append("Verify road access using official road-network data and field inspection.")
        elif name == "Maximum railway distance":
            steps.append("Verify railway access and freight usability through official railway data.")
        elif name == "Maximum power distance":
            steps.append("Verify substation capacity and grid feasibility with the electricity authority.")
        elif name == "Minimum settlement distance":
            steps.append("Verify settlement buffers using population, building-footprint, and revenue-village data.")
        elif "water" in name.lower():
            steps.append("Verify sustainable water availability using official hydrological data.")
        else:
            steps.append(f"Verify: {name}.")

    mapping = {
        "road": "Collect a reliable road-network dataset for this zone.",
        "water": "Collect hydrology and water-resource data for this zone.",
        "settlement": "Collect settlement and population-exposure data for this zone.",
        "power": "Collect official power-grid and substation-capacity data for this zone.",
    }
    for field in missing_critical:
        if field in mapping:
            steps.append(mapping[field])

    for item in failed_requirements:
        steps.append(f"Resolve or redesign around failed requirement: {item.get('requirement')}.")

    steps.extend([
        "Run exact-site verification on the shortlisted coordinates.",
        "Add physical land-cover estimation and consume risk outputs from the Forest and Disaster modules.",
        "Complete field survey and statutory review before any final approval.",
    ])

    return list(dict.fromkeys(steps))


def apply_data_sufficiency_guard(
    analysis: dict,
    gis_profile: dict,
    requirements: SiteRequirements,
) -> dict:
    evidence = analysis.get("evidence", {})
    critical_fields = {
        "road": evidence.get("nearest_major_road_km"),
        "water": evidence.get("nearest_water_feature_km"),
        "settlement": evidence.get("nearest_settlement_km"),
        "power": evidence.get("nearest_power_infrastructure_km"),
    }

    available_count = sum(value is not None for value in critical_fields.values())
    fetch_quality = gis_profile.get("successful_batches", 0) / max(gis_profile.get("total_batches", 1), 1)
    feature_confidence = round((available_count / 4) * 70 + fetch_quality * 30)
    missing_critical = [name for name, value in critical_fields.items() if value is None]

    verified, failed, unverified = get_requirement_groups(analysis)
    mandatory_unknowns = get_mandatory_unknowns(analysis, requirements)
    total_checks = len(analysis.get("requirement_checks", []))
    verified_ratio = len(verified) / total_checks if total_checks else 0
    decision_confidence = round(feature_confidence * 0.7 + verified_ratio * 100 * 0.3)

    guarded_decision = analysis["decision_status"]
    land_assessment = analysis.get(
        "land_assessment",
        {},
    )
    land_decision = land_assessment.get(
        "land_decision"
    )

    if land_decision == "NOT_RECOMMENDED":
        guarded_decision = "NOT_RECOMMENDED"
    elif failed:
        guarded_decision = "NOT_RECOMMENDED"
    elif land_decision == "REQUIRES_GOVERNMENT_VERIFICATION":
        guarded_decision = "REQUIRES_GOVERNMENT_VERIFICATION"
    elif land_decision == "PHYSICAL_ASSESSMENT_REQUIRED":
        guarded_decision = "REQUIRES_PHYSICAL_LAND_ASSESSMENT"
    elif land_decision == "REQUIRES_PHYSICAL_AND_GOVERNMENT_VERIFICATION":
        guarded_decision = "REQUIRES_LAND_VERIFICATION"
    elif mandatory_unknowns:
        guarded_decision = "REQUIRES_VERIFICATION"
    elif feature_confidence < 50:
        guarded_decision = "INSUFFICIENT_DATA"
    elif missing_critical and guarded_decision == "RECOMMENDED":
        guarded_decision = "PRELIMINARY_ONLY"
    elif guarded_decision == "RECOMMENDED" and decision_confidence < 80:
        guarded_decision = "PRELIMINARY_CANDIDATE"

    return {
        **analysis,
        "decision_status": guarded_decision,
        "data_confidence": feature_confidence,
        "decision_confidence": decision_confidence,
        "missing_critical_data": missing_critical,
        "verified_requirements": verified,
        "failed_requirements": failed,
        "unverified_requirements": unverified,
        "mandatory_unverified_requirements": mandatory_unknowns,
        "why_selected": build_selection_explanation(analysis),
        "next_verification_steps": build_next_verification_steps(mandatory_unknowns, missing_critical, failed),
        "screening_level": "COARSE_REGION_LEVEL_SCREENING",
        "decision_integrity": {
            "final_approval_ready": False,
            "reason": "This is a screening result, not a final project approval.",
            "mandatory_unknown_count": len(mandatory_unknowns),
            "failed_requirement_count": len(failed),
            "feature_confidence": feature_confidence,
            "decision_confidence": decision_confidence,
            "land_decision": land_decision,
        },
    }


def get_decision_priority(decision_status: str) -> int:
    priorities = {
        "RECOMMENDED": 7,
        "PRELIMINARY_CANDIDATE": 6,
        "SUITABLE_WITH_CONDITIONS": 5,
        "REQUIRES_GOVERNMENT_VERIFICATION": 5,
        "REQUIRES_PHYSICAL_LAND_ASSESSMENT": 5,
        "REQUIRES_LAND_VERIFICATION": 4,
        "REQUIRES_VERIFICATION": 4,
        "PRELIMINARY_ONLY": 3,
        "NOT_PREFERRED": 2,
        "INSUFFICIENT_DATA": 1,
        "NOT_RECOMMENDED": 0,
    }
    return priorities.get(decision_status, 0)


def zone_sort_key(
    item: dict,
) -> tuple[int, float, float, float]:
    site_analysis = item.get(
        "site_analysis",
        {},
    )

    return (
        get_decision_priority(
            site_analysis.get(
                "decision_status",
                "",
            )
        ),
        site_analysis.get(
            "decision_confidence",
            0,
        ),
        site_analysis.get(
            "data_confidence",
            0,
        ),
        site_analysis.get(
            "overall_score",
            0,
        ),
    )


async def enrich_shortlisted_zones_with_satellite(
    region_name: str,
    zone_results: list[dict],
    data: DiscoverSitesRequest,
) -> dict:
    if not data.auto_satellite_analysis:
        return {
            "requested": False,
            "shortlisted_zone_ids": [],
            "successful_fetches": 0,
            "failed_fetches": 0,
        }

    if not satellite_provider_configured():
        for zone in zone_results:
            if (
                zone.get(
                    "satellite_land_observation"
                )
                is None
            ):
                zone["satellite_land_status"] = (
                    "PROVIDER_NOT_CONFIGURED"
                )

        return {
            "requested": True,
            "shortlisted_zone_ids": [],
            "successful_fetches": 0,
            "failed_fetches": 0,
            "provider_configured": False,
        }

    rankable = [
        zone
        for zone in zone_results
        if zone.get(
            "site_analysis",
            {},
        ).get(
            "data_confidence",
            0,
        ) > 0
    ]

    rankable.sort(
        key=zone_sort_key,
        reverse=True,
    )

    shortlist_count = min(
        len(rankable),
        max(
            1,
            min(
                data.top_zones_per_region,
                SATELLITE_SHORTLIST_PER_REGION,
            ),
        ),
    )

    shortlisted = rankable[:shortlist_count]
    successful = 0
    failed = 0

    async with httpx.AsyncClient() as satellite_client:
        for zone in shortlisted:
            existing_observation = zone.get(
                "satellite_land_observation"
            )

            if existing_observation is not None:
                zone["satellite_land_status"] = (
                    "SUPPLIED"
                )
                continue

            coordinates = zone["coordinates"]

            try:
                observation = (
                    await fetch_satellite_land_observation(
                        coordinates["latitude"],
                        coordinates["longitude"],
                        data.screening_radius_km,
                        satellite_client,
                    )
                )

                observation = observation.model_copy(
                    update={
                        "region": region_name,
                        "zone_id": zone["zone_id"],
                    }
                )

                effective_requirements = (
                    apply_satellite_observation_to_requirements(
                        data.requirements,
                        observation,
                    )
                )

                analysis = build_site_analysis(
                    data.project_type,
                    effective_requirements,
                    zone["gis_profile"],
                    data.project_profile,
                )

                guarded = apply_data_sufficiency_guard(
                    analysis,
                    zone["gis_profile"],
                    effective_requirements,
                )

                zone[
                    "satellite_land_status"
                ] = "FETCHED_AFTER_GIS_SHORTLIST"
                zone[
                    "satellite_land_observation"
                ] = observation.model_dump()
                zone["site_analysis"] = guarded
                successful += 1

            except HTTPException as exc:
                zone[
                    "satellite_land_status"
                ] = f"FAILED: {exc.detail}"
                failed += 1

    shortlisted_ids = {
        zone["zone_id"]
        for zone in shortlisted
    }

    for zone in zone_results:
        if (
            zone["zone_id"]
            not in shortlisted_ids
            and zone.get(
                "satellite_land_observation"
            )
            is None
        ):
            zone["satellite_land_status"] = (
                "SKIPPED_NOT_IN_GIS_SHORTLIST"
            )

    return {
        "requested": True,
        "provider_configured": True,
        "shortlisted_zone_ids": [
            zone["zone_id"]
            for zone in shortlisted
        ],
        "successful_fetches": successful,
        "failed_fetches": failed,
        "strategy": (
            "GIS-first screening followed by "
            "Sentinel-2 analysis only for the "
            "highest-ranked zones in each region."
        ),
    }


def build_final_recommendation(
    overall_candidates: list[dict],
    region_fetch_results: list[dict],
    requested_region_count: int,
) -> Optional[dict]:
    if not overall_candidates:
        return None

    best = overall_candidates[0]
    site = best.get(
        "site_analysis",
        {},
    )
    satellite = best.get(
        "satellite_land_observation"
    )
    evidence = site.get(
        "evidence",
        {},
    )
    land = site.get(
        "land_assessment",
        {},
    )

    successful_regions = [
        item["region"]["input_region"]
        for item in region_fetch_results
        if item["fetch_status"].get(
            "status"
        ) == "SUCCESS"
    ]

    failed_regions = [
        {
            "region": item[
                "region"
            ]["input_region"],
            "status": item[
                "fetch_status"
            ].get("status"),
            "errors": item[
                "fetch_status"
            ].get("errors", []),
            "retry_attempts": item[
                "fetch_status"
            ].get("attempts", 0),
        }
        for item in region_fetch_results
        if item["fetch_status"].get(
            "status"
        ) != "SUCCESS"
    ]

    return {
        "recommendation_type": (
            "PRELIMINARY_SITE_SCREENING"
        ),
        "best_region": best.get("region"),
        "best_zone": best.get("zone_id"),
        "coordinates": best.get(
            "coordinates"
        ),
        "overall_score": site.get(
            "overall_score"
        ),
        "decision_status": site.get(
            "decision_status"
        ),
        "decision_confidence": site.get(
            "decision_confidence"
        ),
        "why_selected": site.get(
            "why_selected",
            [],
        ),
        "infrastructure_evidence": evidence,
        "satellite_evidence": {
            "status": best.get(
                "satellite_land_status"
            ),
            "source": (
                satellite.get("source")
                if satellite
                else None
            ),
            "estimated_physical_land_acres": (
                satellite.get(
                    "estimated_contiguous_suitable_land_acres"
                )
                if satellite
                else None
            ),
            "dominant_land_cover": (
                satellite.get(
                    "dominant_land_cover_type"
                )
                if satellite
                else None
            ),
            "built_up_ratio_percent": (
                satellite.get(
                    "built_up_ratio_percent"
                )
                if satellite
                else None
            ),
            "confidence_percent": (
                satellite.get(
                    "confidence_percent"
                )
                if satellite
                else None
            ),
        },
        "land_decision": land.get(
            "land_decision"
        ),
        "government_verification_required": (
            land.get("land_decision")
            != "LAND_REQUIREMENT_VERIFIED"
        ),
        "regions_requested": (
            requested_region_count
        ),
        "regions_successfully_analyzed": (
            successful_regions
        ),
        "failed_regions": failed_regions,
        "comparison_scope_note": (
            "The recommendation is the best among "
            "regions successfully analyzed in this run."
        ),
        "approval_warning": (
            "This is a prototype screening result, "
            "not a statutory approval. Official land "
            "records, field survey and competent-"
            "authority review remain mandatory."
        ),
    }


async def analyze_region_grid_locally(
    region_name: str,
    points: list[dict],
    region_elements: list[dict],
    region_fetch_status: dict,
    data: DiscoverSitesRequest,
) -> list[dict]:
    results = []

    for point in points:
        profile = build_local_zone_profile(
            region_elements,
            point["latitude"],
            point["longitude"],
            data.screening_radius_km,
            region_fetch_status,
        )

        satellite_observation = (
            find_supplied_satellite_observation(
                data.satellite_land_observations,
                region_name,
                point["zone_id"],
                point["latitude"],
                point["longitude"],
            )
        )

        satellite_status = "NOT_REQUESTED"

        if (
            satellite_observation is None
            and data.auto_satellite_analysis
        ):
            if satellite_provider_configured():
                async with httpx.AsyncClient() as satellite_client:
                    try:
                        satellite_observation = (
                            await fetch_satellite_land_observation(
                                point["latitude"],
                                point["longitude"],
                                data.screening_radius_km,
                                satellite_client,
                            )
                        )
                        satellite_status = "FETCHED"
                    except HTTPException as exc:
                        satellite_status = (
                            f"FAILED: {exc.detail}"
                        )
            else:
                satellite_status = "PROVIDER_NOT_CONFIGURED"

        elif satellite_observation is not None:
            satellite_status = "SUPPLIED"

        effective_requirements = (
            apply_satellite_observation_to_requirements(
                data.requirements,
                satellite_observation,
            )
        )

        analysis = build_site_analysis(
            data.project_type,
            effective_requirements,
            profile,
            data.project_profile,
        )

        guarded = apply_data_sufficiency_guard(
            analysis,
            profile,
            effective_requirements,
        )

        results.append(
            {
                "region": region_name,
                "zone_id": point["zone_id"],
                "coordinates": {
                    "latitude": point["latitude"],
                    "longitude": point["longitude"],
                },
                "zone_status": (
                    "COMPLETED"
                    if region_fetch_status.get(
                        "status"
                    ) == "SUCCESS"
                    else "REGION_FETCH_FAILED"
                ),
                "gis_profile": profile,
                "satellite_land_status": satellite_status,
                "satellite_land_observation": (
                    satellite_observation.model_dump()
                    if satellite_observation
                    else None
                ),
                "site_analysis": guarded,
            }
        )

    return results


@app.get("/")
def home():
    return {
        "message": (
            "Zeryroot AI backend is running"
        ),
        "status": "online",
        "version": "2.3.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Zeryroot AI Backend",
    }


@app.get("/geocode")
async def geocode_location(
    location: str = Query(
        ...,
        min_length=2,
        max_length=200,
        description=(
            "Location name such as "
            "'Kanpur Dehat, Uttar Pradesh'"
        ),
    ),
):
    cleaned_location = location.strip()

    try:
        async with httpx.AsyncClient(
            timeout=15.0
        ) as client:
            results = await fetch_geocode_results(
                cleaned_location,
                client,
                limit=5,
            )

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                "The geocoding service timed out."
            ),
        ) from exc

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "The geocoding service returned "
                f"an error: {exc.response.status_code}"
            ),
        ) from exc

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to connect to the "
                "geocoding service."
            ),
        ) from exc

    if not results:
        raise HTTPException(
            status_code=404,
            detail=(
                "No matching Indian location was "
                f"found for '{cleaned_location}'."
            ),
        )

    locations = [
        format_geocode_result(result)
        for result in results
    ]

    return {
        "success": True,
        "query": cleaned_location,
        "count": len(locations),
        "best_match": locations[0],
        "locations": locations,
        "data_source": (
            "OpenStreetMap Nominatim"
        ),
        "satellite_connected": (
            satellite_provider_configured()
        ),
    }


@app.get("/gis-features")
async def gis_features(
    lat: float = Query(
        ...,
        ge=-90,
        le=90,
    ),
    lon: float = Query(
        ...,
        ge=-180,
        le=180,
    ),
    radius_km: float = Query(
        5,
        ge=1,
        le=15,
        description=(
            "Search radius in kilometres"
        ),
    ),
):
    try:
        timeout = httpx.Timeout(
            25.0,
            connect=10.0,
            read=25.0,
            write=10.0,
            pool=10.0,
        )

        async with httpx.AsyncClient(
            timeout=timeout
        ) as client:
            profile = await get_gis_profile(
                lat,
                lon,
                radius_km,
                client,
            )

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                "The Overpass GIS service timed out. "
                "Try again or use a smaller radius."
            ),
        ) from exc

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "The Overpass GIS service returned "
                f"an error: {exc.response.status_code}"
            ),
        ) from exc

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to connect to the "
                "Overpass GIS service."
            ),
        ) from exc

    return {
        "success": True,
        "gis_profile": profile,
        "message": (
            "GIS extraction completed."
            if not profile["partial_results"]
            else "GIS extraction completed with partial batch results."
        ),
        "satellite_connected": False,
    }


@app.post("/analyze-infra")
def analyze_infra(data: InfraRequest):
    score = round(
        data.land_availability * 0.22
        + data.road_connectivity * 0.22
        + (
            100
            - data.environmental_sensitivity
        ) * 0.18
        + (
            100
            - data.disaster_exposure
        ) * 0.16
        + data.population_need * 0.22
    )

    score -= get_penalty(
        data.acquisition,
        {
            "Low": 0,
            "Medium": 4,
            "High": 9,
            "Very High": 14,
        },
    )

    score -= get_penalty(
        data.ownership,
        {
            "Government Land": 0,
            "Private Land": 4,
            "Mixed Ownership": 6,
            "Forest Land": 15,
            "Agricultural Land": 7,
            "Unknown / To Be Verified": 8,
        },
    )

    score -= get_penalty(
        data.sensitive_zone,
        {
            "No Major Sensitive Zone": 0,
            "Forest / Wildlife Area": 13,
            "River / Wetland": 10,
            "Heritage / Archaeological Site": 12,
            "Defence / Restricted Zone": 15,
            "High Population Settlement": 8,
        },
    )

    score -= get_penalty(
        data.approval,
        {
            "Normal Approval": 0,
            "Environmental Clearance Required": 5,
            "Forest Clearance Required": 12,
            "Land Conversion Required": 6,
            "Disaster Safety Clearance Required": 7,
            "Multiple Clearances Required": 13,
        },
    )

    score -= get_penalty(
        data.water,
        {
            "Good": 0,
            "Moderate": 3,
            "Poor": 7,
            "Needs Separate Supply Project": 10,
        },
    )

    score -= get_penalty(
        data.electricity,
        {
            "Good Grid Access": 0,
            "Moderate Grid Access": 3,
            "Poor Grid Access": 7,
            "Renewable Backup Required": 5,
        },
    )

    score -= get_penalty(
        data.drainage,
        {
            "Good Natural Drainage": 0,
            "Moderate Drainage": 3,
            "Poor Drainage": 8,
            "Flood Drainage Required": 10,
        },
    )

    score = max(
        5,
        min(98, score),
    )

    if score >= 75:
        risk_level = "LOW"
        development_priority = "HIGH"

        recommendation = (
            f"{data.location} appears highly suitable "
            f"for the proposed {data.project_type}. "
            "The submitted indicators show strong "
            "land availability, connectivity and "
            "public demand, with manageable "
            "environmental and disaster exposure. "
            "The project may proceed to detailed "
            "feasibility studies and statutory review."
        )

    elif score >= 55:
        risk_level = "MEDIUM"
        development_priority = "MODERATE"

        recommendation = (
            f"{data.location} is moderately suitable "
            f"for the proposed {data.project_type}. "
            "Additional ground verification, "
            "environmental review, land validation, "
            "disaster-risk assessment and comparison "
            "with alternate sites are recommended."
        )

    else:
        risk_level = "HIGH"
        development_priority = "LOW"

        recommendation = (
            f"{data.location} currently shows weak "
            f"suitability for the proposed "
            f"{data.project_type}. Immediate approval "
            "is not recommended. Constraints should "
            "be resolved or alternate locations "
            "should be evaluated."
        )

    terrain_risk = (
        "LOW"
        if score >= 75
        else "MEDIUM"
        if score >= 55
        else "HIGH"
    )

    environmental_risk = risk_from_value(
        data.environmental_sensitivity
    )

    climate_risk = risk_from_value(
        data.disaster_exposure
    )

    return {
        "success": True,
        "analysis_type": "infrastructure",
        "location": data.location,
        "coordinates": data.coordinates,
        "project_type": data.project_type,
        "project_scale": data.project_scale,
        "suitability_score": score,
        "risk_level": risk_level,
        "development_priority": (
            development_priority
        ),
        "recommendation": recommendation,
        "risk_matrix": {
            "terrain_risk": terrain_risk,
            "environmental_risk": (
                environmental_risk
            ),
            "climate_risk": climate_risk,
        },
        "input_summary": {
            "authority": data.authority,
            "budget": data.budget,
            "area_type": data.area_type,
            "required_land": (
                data.required_land
                or "Not specified"
            ),
            "ownership": data.ownership,
            "acquisition": data.acquisition,
            "sensitive_zone": (
                data.sensitive_zone
            ),
            "approval": data.approval,
            "water": data.water,
            "electricity": data.electricity,
            "drainage": data.drainage,
            "beneficiaries": (
                data.beneficiaries
                or "Not specified"
            ),
            "objective": (
                data.objective
                or "Not specified"
            ),
            "requirements": (
                data.requirements
                or "Not specified"
            ),
        },
        "data_source": (
            "User-submitted indicators"
        ),
        "satellite_connected": False,
    }






@app.get("/satellite/cdse-token-test")
async def cdse_token_test():
    async with httpx.AsyncClient() as client:
        token = await get_cdse_access_token(
            client
        )

    return {
        "success": True,
        "provider": (
            "Copernicus Data Space Ecosystem"
        ),
        "authentication": "CLIENT_CREDENTIALS",
        "token_received": bool(token),
        "token_preview": (
            f"{token[:6]}...{token[-4:]}"
            if len(token) >= 12
            else "RECEIVED"
        ),
        "secret_exposed": False,
    }


@app.get("/satellite/status")
def satellite_status():
    return {
        "success": True,
        "satellite_land_adapter": (
            satellite_provider_status()
        ),
        "accepted_observation_schema": (
            SatelliteLandObservation.model_json_schema()
        ),
        "important_note": (
            "No fake satellite result is generated. "
            "When CDSE credentials are configured, the "
            "backend uses Sentinel-2 L2A statistical "
            "analysis; otherwise it can accept a trusted "
            "manual observation."
        ),
    }


@app.post("/satellite-land-assessment")
async def satellite_land_assessment(
    data: SatelliteLandAssessmentRequest,
):
    async with httpx.AsyncClient() as client:
        observation = (
            await fetch_satellite_land_observation(
                data.latitude,
                data.longitude,
                data.radius_km,
                client,
            )
        )

    return {
        "success": True,
        "analysis_type": (
            "satellite_physical_land_assessment"
        ),
        "provider": satellite_provider_status(),
        "observation": observation.model_dump(),
        "legal_land_verified": False,
        "important_note": (
            "This estimates physical land suitability only. "
            "Legal ownership and availability still require "
            "official government records."
        ),
    }


@app.post("/generate-project-profile")
def generate_project_profile(
    data: ProjectProfileRequest,
):
    profile = infer_project_profile(data)

    return {
        "success": True,
        "analysis_type": (
            "project_requirement_intelligence"
        ),
        "project_profile": profile.model_dump(),
        "engine_type": (
            "validated project registry plus "
            "deterministic requirement inference"
        ),
        "trained_ml_model_connected": False,
        "external_llm_connected": False,
        "important_note": (
            "This engine dynamically interprets project "
            "requirements and generates scoring weights, "
            "but it is not yet a trained ML model or an "
            "external LLM integration."
        ),
    }


@app.post("/analyze-site")
async def analyze_site(data: AnalyzeSiteRequest):
    latitude = data.latitude
    longitude = data.longitude
    resolved_location = None
    validation = {
        "status": "COORDINATES_PROVIDED",
        "confidence": "HIGH",
    }

    if latitude is None or longitude is None:
        if not data.location_query:
            raise HTTPException(
                status_code=400,
                detail="Provide coordinates or location_query.",
            )

        async with httpx.AsyncClient(timeout=15.0) as client:
            results = await fetch_geocode_results(
                data.location_query,
                client,
                limit=5,
            )

        if not results:
            raise HTTPException(
                status_code=404,
                detail="No matching site location was found.",
            )

        formatted = [
            format_geocode_result(item)
            for item in results
        ]

        selected = None

        for item in formatted:
            item_validation = validate_geocode_match(
                item,
                data.expected_district,
                data.expected_state,
            )

            if item_validation["status"] in {
                "VERIFIED",
                "STATE_VERIFIED",
            }:
                selected = item
                validation = item_validation
                break

        if selected is None:
            first = formatted[0]

            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Resolved location does not match the expected district/state.",
                    "best_unverified_match": first,
                    "validation": validate_geocode_match(
                        first,
                        data.expected_district,
                        data.expected_state,
                    ),
                    "alternatives": formatted,
                },
            )

        resolved_location = selected
        latitude = selected["latitude"]
        longitude = selected["longitude"]

    timeout = httpx.Timeout(
        25.0,
        connect=10.0,
        read=25.0,
        write=10.0,
        pool=10.0,
    )

    async with httpx.AsyncClient(timeout=timeout) as client:
        gis_profile = await get_gis_profile(
            latitude,
            longitude,
            data.search_radius_km,
            client,
        )

    resolved_profile = resolve_project_profile(
        project_type=data.project_type,
        project_profile=data.project_profile,
    )

    satellite_observation = (
        data.satellite_observation
    )
    satellite_status = "NOT_REQUESTED"

    if (
        satellite_observation is None
        and data.auto_satellite_analysis
    ):
        if satellite_provider_configured():
            async with httpx.AsyncClient() as satellite_client:
                satellite_observation = (
                    await fetch_satellite_land_observation(
                        latitude,
                        longitude,
                        data.search_radius_km,
                        satellite_client,
                    )
                )
            satellite_status = "FETCHED"
        else:
            satellite_status = "PROVIDER_NOT_CONFIGURED"
    elif satellite_observation is not None:
        satellite_status = "SUPPLIED"

    effective_requirements = (
        apply_satellite_observation_to_requirements(
            data.requirements,
            satellite_observation,
        )
    )

    analysis = build_site_analysis(
        data.project_type,
        effective_requirements,
        gis_profile,
        resolved_profile,
    )

    analysis = apply_data_sufficiency_guard(
        analysis,
        gis_profile,
        effective_requirements,
    )

    return {
        "success": True,
        "analysis_type": "exact_site_analysis",
        "project_type": data.project_type,
        "site_name": data.site_name,
        "coordinates": {
            "latitude": latitude,
            "longitude": longitude,
        },
        "resolved_location": resolved_location,
        "location_validation": validation,
        "gis_profile": gis_profile,
        "satellite_land_status": satellite_status,
        "satellite_land_observation": (
            satellite_observation.model_dump()
            if satellite_observation
            else None
        ),
        "satellite_provider": (
            satellite_provider_status()
        ),
        "site_analysis": analysis,
        "method": (
            "Real OpenStreetMap GIS evidence plus "
            "project-specific rule and constraint analysis"
        ),
        "satellite_connected": False,
        "important_limitations": [
            "This is a preliminary decision-support result.",
            "Satellite, elevation, flood, population, legal-land, and official utility datasets are not yet connected.",
            "Field survey and competent-authority review remain mandatory.",
        ],
    }



@app.post("/discover-sites")
async def discover_sites(
    data: DiscoverSitesRequest,
):
    cleaned_regions = []

    for region in data.preferred_regions:
        cleaned = region.strip()

        if (
            cleaned
            and cleaned.lower()
            not in [
                item.lower()
                for item in cleaned_regions
            ]
        ):
            cleaned_regions.append(cleaned)

    if len(cleaned_regions) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "Provide at least two different "
                "preferred regions."
            ),
        )

    resolved_regions = []

    try:
        async with httpx.AsyncClient(
            timeout=15.0
        ) as geocode_client:
            for index, region in enumerate(
                cleaned_regions
            ):
                results = await fetch_geocode_results(
                    region,
                    geocode_client,
                    limit=3,
                )

                if not results:
                    resolved_regions.append(
                        {
                            "input_region": region,
                            "status": "NOT_FOUND",
                            "best_match": None,
                        }
                    )
                else:
                    formatted = [
                        format_geocode_result(item)
                        for item in results
                    ]

                    selected = None

                    for item in formatted:
                        validation = (
                            validate_geocode_match(
                                item,
                                None,
                                data.expected_state,
                            )
                        )

                        if validation["state_match"]:
                            selected = {
                                **item,
                                "validation": validation,
                            }
                            break

                    resolved_regions.append(
                        {
                            "input_region": region,
                            "status": (
                                "FOUND"
                                if selected
                                else "STATE_MISMATCH"
                            ),
                            "best_match": selected,
                        }
                    )

                if (
                    index
                    < len(cleaned_regions) - 1
                ):
                    await asyncio.sleep(1.1)

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="Region geocoding timed out.",
        ) from exc

    valid_regions = [
        item
        for item in resolved_regions
        if (
            item["status"] == "FOUND"
            and item["best_match"]
        )
    ]

    if len(valid_regions) < 2:
        raise HTTPException(
            status_code=400,
            detail={
                "message": (
                    "Fewer than two preferred "
                    "regions were resolved."
                ),
                "regions": resolved_regions,
            },
        )

    resolved_project_profile = resolve_project_profile(
        project_type=data.project_type,
        project_profile=data.project_profile,
        project_scale=data.project_scale,
        capacity=data.capacity,
        raw_materials=data.raw_materials,
        transport_needs=data.transport_needs,
        utility_needs=data.utility_needs,
        safety_requirements=data.safety_requirements,
        special_requirements=data.special_requirements,
        custom_notes=data.custom_notes,
    )

    region_fetch_results = []

    timeout = httpx.Timeout(
        35.0,
        connect=10.0,
        read=35.0,
        write=10.0,
        pool=10.0,
    )

    async with httpx.AsyncClient(
        timeout=timeout
    ) as gis_client:
        for region in valid_regions:
            region_name = region["input_region"]
            bounding_box = region[
                "best_match"
            ]["bounding_box"]

            elements, fetch_status = (
                await fetch_region_elements(
                    region_name,
                    bounding_box,
                    gis_client,
                )
            )

            points = generate_grid_points(
                bounding_box,
                data.grid_size,
            )

            # Stage 1: GIS-only screening. Real satellite
            # analysis is intentionally deferred until the
            # strongest zones have been shortlisted.
            analysis_request = data.model_copy(
                update={
                    "project_profile": (
                        resolved_project_profile
                    ),
                    "auto_satellite_analysis": False,
                }
            )

            zone_results = (
                await analyze_region_grid_locally(
                    region_name,
                    points,
                    elements,
                    fetch_status,
                    analysis_request,
                )
            )

            satellite_screening = (
                await enrich_shortlisted_zones_with_satellite(
                    region_name,
                    zone_results,
                    data.model_copy(
                        update={
                            "project_profile": (
                                resolved_project_profile
                            )
                        }
                    ),
                )
            )

            region_fetch_results.append(
                {
                    "region": region,
                    "fetch_status": fetch_status,
                    "elements_received": len(
                        elements
                    ),
                    "zone_results": zone_results,
                    "satellite_screening": (
                        satellite_screening
                    ),
                }
            )

            await asyncio.sleep(1.0)

    region_rankings = []
    overall_candidates = []

    for region_result in region_fetch_results:
        region = region_result["region"]
        region_name = region["input_region"]
        zones = region_result["zone_results"]

        rankable_zones = [
            zone
            for zone in zones
            if zone["site_analysis"].get(
                "data_confidence",
                0,
            ) > 0
        ]

        rankable_zones.sort(
            key=zone_sort_key,
            reverse=True,
        )

        top_zones = rankable_zones[
            : data.top_zones_per_region
        ]

        overall_candidates.extend(top_zones)

        region_rankings.append(
            {
                "region": region_name,
                "resolved_region": region[
                    "best_match"
                ],
                "region_fetch_status": (
                    region_result["fetch_status"]
                ),
                "region_elements_received": (
                    region_result[
                        "elements_received"
                    ]
                ),
                "satellite_screening": (
                    region_result.get(
                        "satellite_screening",
                        {},
                    )
                ),
                "zones_screened": len(zones),
                "top_zones": top_zones,
                "all_zone_statuses": [
                    {
                        "zone_id": zone["zone_id"],
                        "coordinates": (
                            zone["coordinates"]
                        ),
                        "zone_status": (
                            zone["zone_status"]
                        ),
                        "overall_score": (
                            zone[
                                "site_analysis"
                            ].get(
                                "overall_score",
                                0,
                            )
                        ),
                        "data_confidence": (
                            zone[
                                "site_analysis"
                            ].get(
                                "data_confidence",
                                0,
                            )
                        ),
                        "decision_status": (
                            zone[
                                "site_analysis"
                            ].get(
                                "decision_status"
                            )
                        ),
                    }
                    for zone in zones
                ],
            }
        )

    overall_candidates.sort(
        key=zone_sort_key,
        reverse=True,
    )

    for rank, candidate in enumerate(
        overall_candidates,
        start=1,
    ):
        candidate["overall_rank"] = rank

    successful_regions = sum(
        result["fetch_status"].get("status")
        == "SUCCESS"
        for result in region_fetch_results
    )

    screening_status = (
        "COMPLETE_COARSE_SCREENING"
        if successful_regions
        == len(region_fetch_results)
        else "PARTIAL_REGION_RESULTS"
    )

    final_recommendation = (
        build_final_recommendation(
            overall_candidates,
            region_fetch_results,
            len(cleaned_regions),
        )
    )

    return {
        "success": True,
        "analysis_type": (
            "broad_region_site_discovery"
        ),
        "project_type": data.project_type,
        "project_profile": (
            resolved_project_profile.model_dump()
        ),
        "satellite_provider": (
            satellite_provider_status()
        ),
        "auto_satellite_analysis_requested": (
            data.auto_satellite_analysis
        ),
        "supplied_satellite_observations": len(
            data.satellite_land_observations
        ),
        "preferred_regions": cleaned_regions,
        "resolved_regions": resolved_regions,
        "grid_size": data.grid_size,
        "zones_per_region": (
            data.grid_size ** 2
        ),
        "screening_radius_km": (
            data.screening_radius_km
        ),
        "region_fetches_attempted": len(
            region_fetch_results
        ),
        "successful_region_fetches": (
            successful_regions
        ),
        "region_rankings": region_rankings,
        "overall_ranking": overall_candidates,
        "final_recommendation": (
            final_recommendation
        ),
        "best_preliminary_zone": (
            overall_candidates[0]
            if overall_candidates
            else None
        ),
        "best_zone_interpretation": (
            {
                "region": overall_candidates[0]["region"],
                "zone_id": overall_candidates[0]["zone_id"],
                "coordinates": overall_candidates[0]["coordinates"],
                "decision_status": overall_candidates[0]["site_analysis"].get("decision_status"),
                "overall_score": overall_candidates[0]["site_analysis"].get("overall_score"),
                "decision_confidence": overall_candidates[0]["site_analysis"].get("decision_confidence"),
                "why_selected": overall_candidates[0]["site_analysis"].get("why_selected", []),
                "unverified_requirements": overall_candidates[0]["site_analysis"].get("unverified_requirements", []),
                "next_verification_steps": overall_candidates[0]["site_analysis"].get("next_verification_steps", []),
            }
            if overall_candidates
            else None
        ),
        "method": (
            "Region-level OpenStreetMap screening "
            "with retry/fallback, followed by local "
            "grid ranking and Sentinel-2 analysis "
            "only for GIS-shortlisted zones."
        ),
        "screening_status": screening_status,
        "cache_enabled": True,
        "satellite_connected": False,
        "important_limitations": [
            (
                "This remains a coarse prototype "
                "and does not inspect every parcel."
            ),
            (
                "Grid points come from district "
                "bounding boxes and may fall outside "
                "the exact administrative polygon."
            ),
            (
                "OSM ways are represented using "
                "their returned centre points, so "
                "distances are approximate."
            ),
            (
                "Sentinel-2 physical land-cover "
                "screening is connected, but its "
                "acreage remains an estimate rather "
                "than a cadastral parcel measurement."
            ),
            (
                "Legal land records, ownership, "
                "official utility capacity and other "
                "module risk outputs still require "
                "separate verification."
            ),
            (
                "Shortlisted coordinates require "
                "exact-site analysis, official data, "
                "and field verification."
            ),
        ],
    }


@app.post("/compare-infra-locations")
async def compare_infra_locations(
    data: CompareLocationsRequest,
):
    cleaned_locations: list[str] = []

    for location in data.candidate_locations:
        cleaned = location.strip()

        if (
            cleaned
            and cleaned.lower()
            not in [
                item.lower()
                for item in cleaned_locations
            ]
        ):
            cleaned_locations.append(cleaned)

    if len(cleaned_locations) < 2:
        raise HTTPException(
            status_code=400,
            detail=(
                "Please provide at least two "
                "different candidate locations."
            ),
        )

    candidates = []

    try:
        geocode_timeout = httpx.Timeout(
            20.0,
            connect=10.0,
        )

        async with httpx.AsyncClient(
            timeout=geocode_timeout
        ) as geocode_client:
            for index, location in enumerate(
                cleaned_locations
            ):
                results = await fetch_geocode_results(
                    location,
                    geocode_client,
                    limit=3,
                )

                if results:
                    candidates.append(
                        {
                            "input_location": location,
                            "geocoding_status": "FOUND",
                            "best_match": (
                                format_geocode_result(
                                    results[0]
                                )
                            ),
                        }
                    )

                else:
                    candidates.append(
                        {
                            "input_location": location,
                            "geocoding_status": (
                                "NOT_FOUND"
                            ),
                            "best_match": None,
                        }
                    )

                if (
                    index
                    < len(cleaned_locations) - 1
                ):
                    await asyncio.sleep(1.1)

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail=(
                "The geocoding service timed out "
                "while processing candidate locations."
            ),
        ) from exc

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "The geocoding service returned "
                f"an error: {exc.response.status_code}"
            ),
        ) from exc

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to connect to the "
                "geocoding service."
            ),
        ) from exc

    found_candidates = [
        item
        for item in candidates
        if item["geocoding_status"] == "FOUND"
    ]

    return {
        "success": True,
        "project_type": data.project_type,
        "requested_candidates": (
            len(cleaned_locations)
        ),
        "resolved_candidates": (
            len(found_candidates)
        ),
        "candidates": candidates,
        "comparison_status": (
            "GEOCODING_COMPLETE"
            if len(found_candidates) >= 2
            else "INSUFFICIENT_VALID_LOCATIONS"
        ),
        "data_source": (
            "OpenStreetMap Nominatim"
        ),
        "gis_analysis_connected": True,
        "gis_analysis_note": (
            "Use /gis-features with candidate "
            "coordinates to retrieve real nearby "
            "OpenStreetMap GIS features. Automatic "
            "multi-candidate GIS comparison is next."
        ),
        "satellite_connected": False,
    }
