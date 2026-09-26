import pandas as pd
import requests as http_requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from app.db import get_supabase
from app.services.forecast_service import (
    run_forecast_for_site,
    train_models_for_site,
    _fetch_measurements,
    _fetch_weather_forecast,
    _fetch_weather_15m,
    _fetch_site_config,
)
from app.services.optimizer_service import run_optimizer_for_site
from app.services.pypsa_optimizer import StandaloneOptimizeRequest, run_standalone_optimize
from datetime import datetime, timedelta, timezone

app = FastAPI(title="EnerSim API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SiteRequest(BaseModel):
    site_id: str
    target_date: str | None = None

@app.on_event("startup")
def prime_supabase_connection():
    try:
        client = get_supabase()
        client.table("forecast_sites").select("id").limit(1).execute()
    except Exception:
        pass

@app.get("/health")
def health():
    return {"status": "ok"}

@app.options("/{full_path:path}")
def preflight_handler(full_path: str):
    return {"ok": True}

@app.post("/train")
def train(req: SiteRequest):
    try:
        return train_models_for_site(req.site_id)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "site_id": req.site_id}
        )

@app.post("/forecast")
def forecast(req: SiteRequest):
    try:
        result = run_forecast_for_site(req.site_id)
        if "error" in result:
            return JSONResponse(status_code=422, content=result)
        return result
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "site_id": req.site_id}
        )

@app.post("/optimize")
def optimize(req: SiteRequest):
    try:
        return run_optimizer_for_site(req.site_id, target_date=req.target_date)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "site_id": req.site_id}
        )

@app.post("/forecast-and-optimize")
def forecast_and_optimize(req: SiteRequest):
    try:
        fc = run_forecast_for_site(req.site_id)
        if "error" in fc:
            return JSONResponse(status_code=422, content={"forecast": fc, "optimization": None})

        opt = run_optimizer_for_site(req.site_id, target_date=req.target_date)
        return {
            "forecast": fc,
            "optimization": opt
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "site_id": req.site_id}
        )

@app.post("/standalone-optimize")
def standalone_optimize(req: StandaloneOptimizeRequest):
    try:
        return run_standalone_optimize(req)
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "date": req.date}
        )

@app.post("/backfill-weather")
def backfill_weather(req: SiteRequest):
    """
    Backfill up to 60 days of historical 15-minute weather data from the
    Open-Meteo Archive API into flex_weather_15m. Run once per site after
    deployment to ensure the PR-ratio model has enough training data.
    Existing rows are never overwritten (ignoreDuplicates=True).
    """
    try:
        supabase = get_supabase()

        config = _fetch_site_config(req.site_id)
        flex_location_id = config.get("flex_location_id")
        lat = config.get("latitude")
        lon = config.get("longitude")

        if not flex_location_id:
            return JSONResponse(status_code=404, content={"error": "No flex_location_id for site"})
        if not lat or not lon:
            return JSONResponse(status_code=400, content={"error": "Site has no GPS coordinates"})

        now = datetime.now(timezone.utc)
        start_date = (now - timedelta(days=60)).strftime("%Y-%m-%d")
        end_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")

        archive_url = (
            f"https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={start_date}&end_date={end_date}"
            f"&minutely_15=shortwave_radiation,direct_radiation,diffuse_radiation,temperature_2m,cloud_cover"
            f"&timezone=UTC&timeformat=iso8601"
        )

        resp = http_requests.get(archive_url, timeout=60)
        if not resp.ok:
            return JSONResponse(
                status_code=502,
                content={"error": f"Open-Meteo archive returned {resp.status_code}", "detail": resp.text[:300]}
            )

        data = resp.json()
        m = data.get("minutely_15")
        if not m or not m.get("time"):
            return JSONResponse(status_code=502, content={"error": "No minutely_15 data in archive response"})

        times = m["time"]
        updated_at = now.isoformat()

        def num(key: str, i: int):
            arr = m.get(key)
            if arr is None or i >= len(arr):
                return None
            v = arr[i]
            return None if v is None else float(v)

        rows = []
        for i, t in enumerate(times):
            ts_utc = t if t.endswith("Z") else t + ":00Z"
            rows.append({
                "flex_location_id": flex_location_id,
                "ts_utc": ts_utc,
                "shortwave_radiation_wm2": num("shortwave_radiation", i),
                "direct_radiation_wm2": num("direct_radiation", i),
                "diffuse_radiation_wm2": num("diffuse_radiation", i),
                "temperature_c": num("temperature_2m", i),
                "cloud_cover_pct": num("cloud_cover", i),
                "is_forecast": False,
                "updated_at": updated_at,
            })

        inserted = 0
        batch_size = 500
        for offset in range(0, len(rows), batch_size):
            batch = rows[offset:offset + batch_size]
            result = supabase.table("flex_weather_15m").upsert(
                batch,
                on_conflict="flex_location_id,ts_utc",
                ignore_duplicates=True
            ).execute()
            inserted += len(batch)

        return {
            "site_id": req.site_id,
            "flex_location_id": flex_location_id,
            "start_date": start_date,
            "end_date": end_date,
            "rows_processed": len(rows),
            "status": "ok",
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "site_id": req.site_id}
        )

@app.get("/status/{site_id}")
def site_status(site_id: str):
    """Check what data and config is available for a site."""
    try:
        now = datetime.now(timezone.utc)
        config = _fetch_site_config(site_id)
        flex_location_id = config.get("flex_location_id")

        recent_start = now - timedelta(days=7)
        measurements = _fetch_measurements(site_id, recent_start, now)

        measurement_status = {
            "rows_last_7d": len(measurements),
            "has_load_kw": bool(not measurements.empty and "load_kw" in measurements.columns and measurements["load_kw"].notna().any()),
            "has_pv_kw": bool(not measurements.empty and "pv_kw" in measurements.columns and measurements["pv_kw"].notna().any()),
            "has_ev_kw": bool(not measurements.empty and "ev_kw" in measurements.columns and measurements["ev_kw"].notna().any()),
            "has_battery_kw": bool(not measurements.empty and "battery_kw" in measurements.columns and measurements["battery_kw"].notna().any()),
        }

        weather = _fetch_weather_forecast(site_id, now - timedelta(hours=1), now + timedelta(hours=24))
        weather_15m = pd.DataFrame()
        weather_15m_hist_rows = 0
        if flex_location_id:
            weather_15m = _fetch_weather_15m(flex_location_id, now - timedelta(hours=1), now + timedelta(hours=24))
            hist_df = _fetch_weather_15m(flex_location_id, now - timedelta(days=60), now - timedelta(hours=1))
            weather_15m_hist_rows = len(hist_df)

        weather_status = {
            "forecast_rows": len(weather),
            "available": not weather.empty,
            "weather_15m_forecast_rows": len(weather_15m),
            "weather_15m_history_rows": weather_15m_hist_rows,
            "weather_15m_available": not weather_15m.empty,
        }

        config_status = {
            "pv_configured": config.get("pv_capacity_kwp", 0) > 0,
            "battery_configured": config.get("battery_power_kw", 0) > 0 and config.get("battery_capacity_kwh", 0) > 0,
            "ev_configured": config.get("ev_charge_power_kw", 0) > 0,
            "grid_limit_configured": config.get("grid_import_limit_kw", 0) > 0,
        }

        capabilities = []
        if measurement_status["has_load_kw"] or measurement_status["rows_last_7d"] > 0:
            capabilities.append("load_forecast")
        if measurement_status["has_pv_kw"] and weather_status["weather_15m_history_rows"] >= 96 * 3:
            capabilities.append("pv_forecast_pr_weather")
        elif measurement_status["has_pv_kw"]:
            capabilities.append("pv_forecast_measured")
        elif config_status["pv_configured"] and (weather_status["available"] or weather_status["weather_15m_available"]):
            capabilities.append("pv_forecast_weather_physics")
        if measurement_status["has_ev_kw"] or config_status["ev_configured"]:
            capabilities.append("ev_forecast")
        if config_status["battery_configured"]:
            capabilities.append("battery_optimization")

        return {
            "site_id": site_id,
            "measurements": measurement_status,
            "weather": weather_status,
            "config": config_status,
            "capabilities": capabilities,
            "raw_config": config,
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "site_id": site_id}
        )
