import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from supabase import create_client

load_dotenv()

app = FastAPI(title="Butterfly AI")

# Add browser origins explicitly through CORS_ORIGINS, separated by commas.
# An empty value leaves cross-origin browser access disabled by default.
cors_origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "").split(",")
    if origin.strip()
]
if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", "Authorization"],
    )

# Connect to the existing Supabase project
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")

supabase = None

if supabase_url and supabase_key:
    supabase = create_client(supabase_url, supabase_key)


class ScenarioCreate(BaseModel):
    scenario_name: str = Field(min_length=1, max_length=150)
    description: Optional[str] = None


# These are the 36 numeric raw input columns for the congestion model.
# All fields are optional because the saved pipeline imputes missing values.
class PredictionInput(BaseModel):
    population: Optional[float] = None
    zone_area_km2: Optional[float] = None
    population_density_per_km2: Optional[float] = None
    road_length_km: Optional[float] = None
    road_density_km_per_km2: Optional[float] = None
    road_capacity_vehicles_per_hour: Optional[float] = None
    traffic_volume_vehicles_per_hour: Optional[float] = None
    avg_vehicle_speed_kmh: Optional[float] = None
    avg_travel_time_min: Optional[float] = None
    metro_station_count: Optional[float] = None
    metro_ridership_daily: Optional[float] = None
    bus_stop_count: Optional[float] = None
    bus_route_count: Optional[float] = None
    distance_to_nearest_transit_km: Optional[float] = None
    job_count: Optional[float] = None
    business_count: Optional[float] = None
    office_count: Optional[float] = None
    commercial_land_area_km2: Optional[float] = None
    office_floor_area_sqm: Optional[float] = None
    residential_building_count: Optional[float] = None
    housing_units: Optional[float] = None
    average_rent_inr_month: Optional[float] = None
    housing_demand_index_0_100: Optional[float] = None
    school_count: Optional[float] = None
    hospital_count: Optional[float] = None
    park_count: Optional[float] = None
    water_body_count: Optional[float] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    new_jobs_scenario: Optional[float] = None
    estimated_commute_share: Optional[float] = None
    estimated_private_vehicle_share: Optional[float] = None
    estimated_public_transport_share: Optional[float] = None
    estimated_new_commuters_scenario: Optional[float] = None
    estimated_private_vehicle_commuters_scenario: Optional[float] = None
    estimated_public_transport_commuters_scenario: Optional[float] = None


_prediction_model = None
_model_load_failed = False


def get_prediction_model():
    """Load the local pipeline on first use so backend routes still start without it."""
    global _prediction_model, _model_load_failed
    if _prediction_model is not None:
        return _prediction_model
    if _model_load_failed:
        return None

    model_path = Path(
        os.getenv(
            "BUTTERFLY_MODEL_PATH",
            str(Path(__file__).resolve().with_name("butterfly_model.joblib")),
        )
    )
    try:
        import joblib

        if not model_path.is_file():
            _model_load_failed = True
            return None
        _prediction_model = joblib.load(model_path)
        return _prediction_model
    except Exception:
        # Keep the API available; the endpoint will return a generic 503.
        _model_load_failed = True
        return None


@app.get("/")
def home():
    return {"message": "Butterfly AI backend is running!"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/predict")
def predict(input_data: PredictionInput):
    model = get_prediction_model()
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Prediction model is unavailable. Check model file and ML dependencies.",
        )

    try:
        import pandas as pd

        row = input_data.model_dump()
        # Preserve the fitted pipeline's exact input column order.
        feature_names = list(model.feature_names_in_)
        features = pd.DataFrame(
            [{name: row[name] for name in feature_names}],
            columns=feature_names,
        )
        value = model.predict(features)[0]
        if hasattr(value, "item"):
            value = value.item()
        return {
            "prediction": value,
            "predicted_target": "congestion_index_0_100 (0–100 scale)",
        }
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check that the model and input data are compatible.",
        )


@app.post("/scenarios")
def create_scenario(scenario: ScenarioCreate):
    if supabase is None:
        raise HTTPException(
            status_code=503,
            detail="Supabase is not configured."
        )

    try:
        response = (
            supabase.table("scenarios")
            .insert(scenario.model_dump())
            .execute()
        )
        return {"scenario": response.data[0]}

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Could not save scenario to the database."
        )


@app.get("/scenarios")
def list_scenarios():
    if supabase is None:
        raise HTTPException(
            status_code=503,
            detail="Supabase is not configured."
        )

    try:
        response = (
            supabase.table("scenarios")
            .select("*")
            .order("created_at", desc=True)
            .execute()
        )
        return {"scenarios": response.data}

    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve scenarios."
        )


@app.get("/scenarios/{scenario_id}")
def get_scenario(scenario_id: int):
    if supabase is None:
        raise HTTPException(
            status_code=503,
            detail="Supabase is not configured."
        )

    try:
        response = (
            supabase.table("scenarios")
            .select("*")
            .eq("id", scenario_id)
            .limit(1)
            .execute()
        )

        if not response.data:
            raise HTTPException(
                status_code=404,
                detail="Scenario not found."
            )

        return {"scenario": response.data[0]}

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve scenario."
        )
