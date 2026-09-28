import json
import math
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from app.db import get_supabase, supabase as _default_client

logger = logging.getLogger("enersim-api.forecast")

_client = _default_client

CONNECTION_ERRORS = ("Server disconnected", "RemoteProtocolError", "ConnectionError", "ConnectError", "ReadTimeout")

MEASUREMENT_COLS = "ts_utc, power_kw, load_kw, pv_kw, ev_kw, battery_kw, soc_percent"
HISTORY_DAYS_TRAIN = 60
HISTORY_DAYS_FORECAST_FALLBACK = 30
MODEL_MAX_AGE_HOURS = 48
MODEL_ID = "similar_day_weather_v2"
PR_MIN_SAMPLES_PER_SLOT = 3
PR_GHI_THRESHOLD_WM2 = 20.0
PR_MAX_RATIO = 1.1

def _get_client():
    global _client
    return _client

def _refresh_client():
    global _client
    _client = get_supabase()
    return _client

def _with_retry(fn):
    try:
        return fn(_get_client())
    except Exception as e:
        err_str = str(e)
        if any(keyword in err_str for keyword in CONNECTION_ERRORS):
            client = _refresh_client()
            return fn(client)
        raise

def _safe_float(value, default: float = 0.0) -> float:
    try:
        f = float(value)
        if math.isfinite(f):
            return round(f, 4)
        return default
    except (TypeError, ValueError):
        return default

def _sanitize_array(arr: np.ndarray, default: float = 0.0) -> np.ndarray:
    arr = np.where(np.isfinite(arr), arr, default)
    return arr

def _fetch_measurements(site_id: str, start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
    def _query(client):
        return (
            client.table("site_measurements_15m")
            .select(MEASUREMENT_COLS)
            .eq("site_id", site_id)
            .gte("ts_utc", start_ts.isoformat())
            .lte("ts_utc", end_ts.isoformat())
            .order("ts_utc")
            .execute()
        )

    result = _with_retry(_query)
    df = pd.DataFrame(result.data or [])
    if not df.empty:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        for col in ["power_kw", "load_kw", "pv_kw", "ev_kw", "battery_kw", "soc_percent"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def _fetch_weather_forecast(site_id: str, start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
    def _query(client):
        return (
            client.table("weather_forecast_hourly")
            .select("ts_utc, ghi_wm2, dni_wm2, dhi_wm2, temperature_c, cloud_cover_pct")
            .eq("site_id", site_id)
            .gte("ts_utc", start_ts.isoformat())
            .lte("ts_utc", end_ts.isoformat())
            .order("ts_utc")
            .execute()
        )

    result = _with_retry(_query)
    df = pd.DataFrame(result.data or [])
    if not df.empty:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        for col in ["ghi_wm2", "dni_wm2", "dhi_wm2", "temperature_c", "cloud_cover_pct"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def _fetch_weather_history(site_id: str, start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
    def _query(client):
        return (
            client.table("weather_history_hourly")
            .select("ts_utc, ghi_wm2, temperature_c, cloud_cover_pct")
            .eq("site_id", site_id)
            .gte("ts_utc", start_ts.isoformat())
            .lte("ts_utc", end_ts.isoformat())
            .order("ts_utc")
            .execute()
        )

    result = _with_retry(_query)
    df = pd.DataFrame(result.data or [])
    if not df.empty:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        for col in ["ghi_wm2", "temperature_c", "cloud_cover_pct"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def _fetch_weather_15m(flex_location_id: str, start_ts: datetime, end_ts: datetime) -> pd.DataFrame:
    """Fetch 15-minute weather data from flex_weather_15m (populated by fetch-flex-weather)."""
    def _query(client):
        return (
            client.table("flex_weather_15m")
            .select("ts_utc, shortwave_radiation_wm2, direct_radiation_wm2, diffuse_radiation_wm2, temperature_c, cloud_cover_pct")
            .eq("flex_location_id", flex_location_id)
            .gte("ts_utc", start_ts.isoformat())
            .lte("ts_utc", end_ts.isoformat())
            .order("ts_utc")
            .execute()
        )

    result = _with_retry(_query)
    df = pd.DataFrame(result.data or [])
    if not df.empty:
        df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)
        for col in ["shortwave_radiation_wm2", "direct_radiation_wm2", "diffuse_radiation_wm2", "temperature_c", "cloud_cover_pct"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    return df

def _fetch_site_config(site_id: str) -> dict:
    def _query_site(client):
        return (
            client.table("forecast_sites")
            .select("id, flex_location_id, latitude, longitude, metadata")
            .eq("id", site_id)
            .maybeSingle()
            .execute()
        )

    site_result = _with_retry(_query_site)
    site_data = site_result.data
    if not site_data:
        return {}

    config = {
        "latitude": _safe_float(site_data.get("latitude")),
        "longitude": _safe_float(site_data.get("longitude")),
        "flex_location_id": site_data.get("flex_location_id"),
    }

    flex_location_id = site_data.get("flex_location_id")
    if not flex_location_id:
        return config

    def _query_location(client):
        return (
            client.table("flex_locations")
            .select(
                "pv_vermogen, pv_hellingshoek, pv_azimuth, pv_azimuth_west, "
                "pv_oost_west_opstelling, "
                "batterij_vermogen, batterij_capaciteit, batterij_rte, batterij_dod, "
                "ev_laadvermogen, ev_type_gebruik, "
                "gecontracteerd_importvermogen, gecontracteerd_exportvermogen, "
                "piekvermogen"
            )
            .eq("id", flex_location_id)
            .maybeSingle()
            .execute()
        )

    loc_result = _with_retry(_query_location)
    loc_data = loc_result.data
    if not loc_data:
        return config

    config.update({
        "pv_capacity_kwp": _safe_float(loc_data.get("pv_vermogen")),
        "pv_tilt_deg": _safe_float(loc_data.get("pv_hellingshoek"), 30.0),
        "pv_azimuth_deg": _safe_float(loc_data.get("pv_azimuth"), 180.0),
        "pv_azimuth_west_deg": _safe_float(loc_data.get("pv_azimuth_west"), 270.0),
        "pv_east_west": bool(loc_data.get("pv_oost_west_opstelling")),
        "battery_power_kw": _safe_float(loc_data.get("batterij_vermogen")),
        "battery_capacity_kwh": _safe_float(loc_data.get("batterij_capaciteit")),
        "battery_rte": _safe_float(loc_data.get("batterij_rte"), 90.0) / 100.0,
        "battery_dod": _safe_float(loc_data.get("batterij_dod"), 90.0) / 100.0,
        "ev_charge_power_kw": _safe_float(loc_data.get("ev_laadvermogen")),
        "ev_usage_type": loc_data.get("ev_type_gebruik") or "Werk",
        "grid_import_limit_kw": _safe_float(loc_data.get("gecontracteerd_importvermogen")),
        "grid_export_limit_kw": _safe_float(loc_data.get("gecontracteerd_exportvermogen")),
        "peak_power_kw": _safe_float(loc_data.get("piekvermogen")),
    })

    return config

# ---------------------------------------------------------------------------
# Profile computation helpers (used by training)
# ---------------------------------------------------------------------------

def _compute_load_profile(df: pd.DataFrame) -> list[float]:
    df = df.sort_values("ts_utc").reset_index(drop=True)

    if "load_kw" in df.columns and df["load_kw"].notna().sum() > 96:
        values = df["load_kw"].dropna().values
    elif "power_kw" in df.columns and not df["power_kw"].isna().all():
        load_series = df["power_kw"].copy()
        if "pv_kw" in df.columns:
            load_series = load_series + df["pv_kw"].fillna(0)
        if "battery_kw" in df.columns:
            load_series = load_series - df["battery_kw"].fillna(0)
        values = load_series.dropna().values
    else:
        return [0.0] * 96

    if len(values) == 0:
        return [0.0] * 96

    if len(values) >= 96 * 28:
        n = 96 * 28
        result = np.nanmean(values[-n:].reshape(28, 96), axis=0)
    elif len(values) >= 96 * 7:
        n = 96 * 7
        result = np.nanmean(values[-n:].reshape(7, 96), axis=0)
    elif len(values) >= 96:
        result = values[-96:]
    else:
        result = np.full(96, _safe_float(np.nanmean(values)))

    return [_safe_float(v) for v in _sanitize_array(result)]

def _compute_load_std(df: pd.DataFrame) -> list[float]:
    df = df.sort_values("ts_utc").reset_index(drop=True)

    if "load_kw" in df.columns and df["load_kw"].notna().sum() > 96:
        values = df["load_kw"].dropna().values
    elif "power_kw" in df.columns and not df["power_kw"].isna().all():
        load_series = df["power_kw"].copy()
        if "pv_kw" in df.columns:
            load_series = load_series + df["pv_kw"].fillna(0)
        if "battery_kw" in df.columns:
            load_series = load_series - df["battery_kw"].fillna(0)
        values = load_series.dropna().values
    else:
        return [0.25] * 96

    if len(values) >= 96 * 7:
        n = 96 * 7
        daily = values[-n:].reshape(7, 96)
        spread = np.nanstd(daily, axis=0)
        spread = np.where(np.isfinite(spread) & (spread > 0.1), spread, 0.25)
    else:
        std_val = np.nanstd(values) if len(values) > 10 else 0.25
        std_val = _safe_float(std_val, 0.25)
        if std_val < 0.1:
            std_val = 0.25
        spread = np.full(96, std_val)

    return [_safe_float(v, 0.25) for v in _sanitize_array(spread, 0.25)]

def _compute_pv_profile(df: pd.DataFrame) -> list[float] | None:
    if "pv_kw" not in df.columns:
        return None

    df = df.sort_values("ts_utc").reset_index(drop=True)
    values = df["pv_kw"].dropna().values

    if len(values) < 96 * 7 or not np.any(values > 0):
        return None

    if len(values) >= 96 * 28:
        n = 96 * 28
        result = np.nanmean(values[-n:].reshape(28, 96), axis=0)
    else:
        n = 96 * 7
        result = np.nanmean(values[-n:].reshape(7, 96), axis=0)

    result = np.clip(result, 0, None)
    return [_safe_float(v) for v in _sanitize_array(result)]

def _compute_ev_profile(df: pd.DataFrame, site_config: dict) -> list[float]:
    ev_charge_power = site_config.get("ev_charge_power_kw", 0.0)

    if "ev_kw" in df.columns:
        ev_values = df["ev_kw"].dropna().values
        if len(ev_values) >= 96 * 7 and np.any(ev_values > 0):
            n = min(len(ev_values), 96 * 28)
            n = (n // 96) * 96
            trimmed = ev_values[-n:]
            result = np.nanmean(trimmed.reshape(n // 96, 96), axis=0)
            return [_safe_float(v) for v in _sanitize_array(np.clip(result, 0, None))]

    if ev_charge_power > 0:
        usage_type = site_config.get("ev_usage_type", "Werk")
        pattern = _typical_ev_pattern(ev_charge_power, usage_type)
        return [_safe_float(v) for v in pattern]

    return [0.0] * 96

def _typical_ev_pattern(charge_power_kw: float, usage_type: str) -> np.ndarray:
    pattern = np.zeros(96)

    if usage_type == "Werk":
        for i in range(88, 96):
            pattern[i] = charge_power_kw * 0.6
        for i in range(0, 24):
            pattern[i] = charge_power_kw * 0.6
        for i in range(48, 56):
            pattern[i] = charge_power_kw * 0.2
    else:
        for i in range(92, 96):
            pattern[i] = charge_power_kw * 0.5
        for i in range(0, 28):
            pattern[i] = charge_power_kw * 0.5

    return _sanitize_array(pattern * 0.4)

# ---------------------------------------------------------------------------
# PV performance ratio: learned from (measured GHI, measured PV) pairs
# ---------------------------------------------------------------------------

def _compute_pv_pr_ratio(
    df_meas: pd.DataFrame,
    df_weather: pd.DataFrame,
    pv_capacity_kwp: float,
) -> list[float] | None:
    """
    Compute a per-slot (0-95) performance ratio from historical pairs.
    """
    if df_meas.empty or df_weather.empty:
        return None
    if not math.isfinite(pv_capacity_kwp) or pv_capacity_kwp <= 0:
        return None
    if "shortwave_radiation_wm2" not in df_weather.columns:
        return None

    meas = df_meas[["ts_utc", "pv_kw"]].dropna(subset=["pv_kw"]).copy()
    if meas["ts_utc"].dt.tz is None:
        meas["ts_utc"] = meas["ts_utc"].dt.tz_localize("UTC")

    weather = df_weather[["ts_utc", "shortwave_radiation_wm2", "temperature_c"]].copy()
    if weather["ts_utc"].dt.tz is None:
        weather["ts_utc"] = weather["ts_utc"].dt.tz_localize("UTC")
    weather = weather.rename(columns={"shortwave_radiation_wm2": "ghi"})

    merged = pd.merge(meas, weather, on="ts_utc", how="inner")
    if len(merged) < 96:
        return None

    merged = merged[
        (merged["ghi"] > PR_GHI_THRESHOLD_WM2) &
        (merged["pv_kw"] >= 0)
    ].copy()

    if len(merged) < 96:
        return None

    merged["pr"] = merged["pv_kw"] / (merged["ghi"] / 1000.0 * pv_capacity_kwp)
    merged = merged[(merged["pr"] >= 0) & (merged["pr"] <= PR_MAX_RATIO)].copy()

    if len(merged) < 96:
        return None

    now = datetime.now(timezone.utc)
    merged["days_ago"] = (now - merged["ts_utc"]).dt.total_seconds() / 86400.0
    merged["weight"] = 0.4 + 0.6 * (
        1.0 - merged["days_ago"].clip(0, HISTORY_DAYS_TRAIN) / HISTORY_DAYS_TRAIN
    )

    merged["slot"] = merged["ts_utc"].dt.hour * 4 + merged["ts_utc"].dt.minute // 15

    ratio_96 = np.full(96, np.nan)
    sample_count = np.zeros(96, dtype=int)

    for slot in range(96):
        slot_data = merged[merged["slot"] == slot]
        if len(slot_data) >= PR_MIN_SAMPLES_PER_SLOT:
            w = slot_data["weight"].values
            pr = slot_data["pr"].values
            ratio_96[slot] = float(np.average(pr, weights=w))
            sample_count[slot] = len(slot_data)

    valid_mask = np.isfinite(ratio_96)
    if valid_mask.sum() < 12:
        return None

    if not np.all(valid_mask):
        indices = np.arange(96)
        filled = ratio_96.copy()
        for i in range(96):
            if not np.isfinite(filled[i]):
                dists = np.where(valid_mask, np.abs(indices - i), 9999)
                nearest = int(np.argmin(dists))
                filled[i] = ratio_96[nearest]
        ratio_96 = filled

    ratio_96 = np.where(sample_count > 0, ratio_96, 0.0)

    return [_safe_float(v, 0.0) for v in _sanitize_array(ratio_96, 0.0)]

# ---------------------------------------------------------------------------
# Model persistence (read/write forecast_model_params)
# ---------------------------------------------------------------------------

def _save_model_params(site_id: str, params: dict, training_rows: int):
    row = {
        "site_id": site_id,
        "model_id": MODEL_ID,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows_used": training_rows,
        "params": json.dumps(params),
    }

    def _upsert(client):
        return (
            client.table("forecast_model_params")
            .upsert(row, on_conflict="site_id,model_id")
            .execute()
        )

    _with_retry(_upsert)

def _load_model_params(site_id: str) -> dict | None:
    def _query(client):
        return (
            client.table("forecast_model_params")
            .select("trained_at, params")
            .eq("site_id", site_id)
            .eq("model_id", MODEL_ID)
            .maybeSingle()
            .execute()
        )

    result = _with_retry(_query)
    if not result.data:
        return None

    trained_at_str = result.data.get("trained_at")
    if trained_at_str:
        trained_at = datetime.fromisoformat(trained_at_str.replace("Z", "+00:00"))
        age_hours = (datetime.now(timezone.utc) - trained_at).total_seconds() / 3600
        if age_hours > MODEL_MAX_AGE_HOURS:
            return None

    params_raw = result.data.get("params")
    if isinstance(params_raw, str):
        return json.loads(params_raw)
    if isinstance(params_raw, dict):
        return params_raw
    return None

# ---------------------------------------------------------------------------
# Training: compute profiles and persist to DB
# ---------------------------------------------------------------------------

def train_models_for_site(site_id: str):
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=HISTORY_DAYS_TRAIN)

    df = _fetch_measurements(site_id, start, now)
    if df.empty:
        return {"error": "geen meetdata gevonden", "site_id": site_id}

    site_config = _fetch_site_config(site_id)
    flex_location_id = site_config.get("flex_location_id")
    pv_capacity_kwp = site_config.get("pv_capacity_kwp", 0.0)

    load_profile = _compute_load_profile(df)
    load_std = _compute_load_std(df)
    pv_profile = _compute_pv_profile(df)
    ev_profile = _compute_ev_profile(df, site_config)

    pv_pr_ratio = None
    pr_weather_rows = 0
    if pv_capacity_kwp > 0 and flex_location_id:
        df_weather_hist = _fetch_weather_15m(flex_location_id, start, now)
        pr_weather_rows = len(df_weather_hist)
        if not df_weather_hist.empty:
            pv_pr_ratio = _compute_pv_pr_ratio(df, df_weather_hist, pv_capacity_kwp)

    row_count = len(df)
    load_rows = int(df["load_kw"].notna().sum()) if "load_kw" in df.columns else 0
    pv_rows = int(df["pv_kw"].notna().sum()) if "pv_kw" in df.columns else 0

    params = {
        "load_profile_96": load_profile,
        "load_std_96": load_std,
        "pv_profile_96": pv_profile,
        "pv_pr_ratio_96": pv_pr_ratio,
        "ev_profile_96": ev_profile,
        "has_measured_pv": pv_profile is not None,
        "has_pr_model": pv_pr_ratio is not None,
        "load_rows": load_rows,
        "pv_rows": pv_rows,
    }

    _save_model_params(site_id, params, row_count)

    return {
        "site_id": site_id,
        "status": "trained",
        "rows_used": row_count,
        "load_rows": load_rows,
        "pv_rows": pv_rows,
        "pr_weather_rows": pr_weather_rows,
        "has_pr_model": pv_pr_ratio is not None,
        "model_id": MODEL_ID,
    }

# ---------------------------------------------------------------------------
# PV from weather physics (fallback when no learned PR model)
# ---------------------------------------------------------------------------

def _estimate_pv_from_weather(weather_df: pd.DataFrame, site_config: dict) -> np.ndarray:
    if weather_df.empty:
        return np.zeros(96)

    pv_capacity_kwp = site_config.get("pv_capacity_kwp", 0.0)
    if not math.isfinite(pv_capacity_kwp) or pv_capacity_kwp <= 0:
        return np.zeros(96)

    tilt_deg = site_config.get("pv_tilt_deg", 30.0)
    is_east_west = site_config.get("pv_east_west", False)

    ghi = weather_df["ghi_wm2"].fillna(0).values
    ghi = np.where(np.isfinite(ghi), ghi, 0)

    has_components = ("dni_wm2" in weather_df.columns and "dhi_wm2" in weather_df.columns)
    if has_components:
        dni = weather_df["dni_wm2"].fillna(0).values
        dhi = weather_df["dhi_wm2"].fillna(0).values
        dni = np.where(np.isfinite(dni), dni, 0)
        dhi = np.where(np.isfinite(dhi), dhi, 0)
    else:
        dni = None
        dhi = None

    if "temperature_c" in weather_df.columns:
        temps = weather_df["temperature_c"].fillna(25).values
        temps = np.where(np.isfinite(temps), temps, 25)
        temp_factor = 1.0 - 0.004 * np.maximum(temps - 25, 0)
        temp_factor = np.clip(temp_factor, 0.7, 1.05)
    else:
        temp_factor = np.ones(len(ghi))

    if dni is not None and dhi is not None:
        tilt_rad = math.radians(tilt_deg)
        cos_tilt_factor = 1.0 + 0.03 * math.cos(tilt_rad)
        poa_irradiance = (
            dni * cos_tilt_factor * 0.85 +
            dhi * (1 + math.cos(tilt_rad)) / 2 +
            ghi * 0.2 * (1 - math.cos(tilt_rad)) / 2
        )
    else:
        tilt_factor = 1.0 + 0.1 * math.sin(math.radians(tilt_deg))
        poa_irradiance = ghi * tilt_factor

    poa_irradiance = np.clip(poa_irradiance, 0, 1400)
    system_efficiency = 0.86

    if is_east_west:
        pv_kw = pv_capacity_kwp * (poa_irradiance / 1000.0) * temp_factor * system_efficiency * 0.92
    else:
        pv_kw = pv_capacity_kwp * (poa_irradiance / 1000.0) * temp_factor * system_efficiency

    pv_kw_q = np.repeat(pv_kw, 4)[:96]
    if len(pv_kw_q) < 96:
        pv_kw_q = np.pad(pv_kw_q, (0, 96 - len(pv_kw_q)), constant_values=0)

    pv_kw_q = np.clip(pv_kw_q, 0, pv_capacity_kwp)
    return _sanitize_array(pv_kw_q)

def _apply_temperature_correction(base_forecast: np.ndarray,
                                   weather_fc: pd.DataFrame) -> np.ndarray:
    temp_col = "temperature_c" if "temperature_c" in weather_fc.columns else None
    if temp_col is None:
        return base_forecast

    fc_temps = weather_fc[temp_col].values
    if len(fc_temps) == 0:
        return base_forecast

    avg_fc_temp = np.nanmean(fc_temps)
    if not math.isfinite(avg_fc_temp):
        return base_forecast

    baseline_temp = 15.0
    temp_delta = avg_fc_temp - baseline_temp

    if temp_delta < 0:
        correction_factor = 1.0 + abs(temp_delta) * 0.02
    elif avg_fc_temp > 25:
        correction_factor = 1.0 + (avg_fc_temp - 25) * 0.03
    else:
        correction_factor = 1.0

    correction_factor = max(0.8, min(1.3, correction_factor))
    return base_forecast * correction_factor

# ---------------------------------------------------------------------------
# Save forecast predictions to DB
# FIX: uses upsert instead of delete-then-insert (atomic-safe)
# ---------------------------------------------------------------------------

def _save_forecast(site_id: str, load_fc: np.ndarray, pv_fc: np.ndarray, ev_fc: np.ndarray,
                   lower: np.ndarray, upper: np.ndarray, model_id: str):
    now = datetime.now(timezone.utc)
    start = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)

    rows = []
    for i in range(96):
        ts = start + timedelta(minutes=15 * i)
        load_val = _safe_float(load_fc[i])
        pv_val = _safe_float(pv_fc[i])
        ev_val = _safe_float(ev_fc[i])
        net_val = _safe_float(load_val - pv_val + ev_val)
        lower_val = _safe_float(lower[i])
        upper_val = _safe_float(upper[i])

        rows.append({
            "site_id": site_id,
            "ts_utc": ts.isoformat(),
            "model_id": model_id,
            "predicted_power_kw": net_val,
            "predicted_load_kw": load_val,
            "predicted_pv_kw": pv_val,
            "predicted_ev_kw": ev_val,
            "predicted_net_kw": net_val,
            "confidence_lower": lower_val,
            "confidence_upper": upper_val,
            "forecast_type": "day_ahead",
            "metadata": {
                "source": "enersim-api",
                "version": model_id
            }
        })

    def _upsert(client):
        return client.table("forecast_predictions_15m").upsert(
            rows,
            on_conflict="site_id,ts_utc,model_id"
        ).execute()

    try:
        _with_retry(_upsert)
    except Exception as e:
        logger.error("Failed to upsert forecast predictions: %s", e)
        raise

# ---------------------------------------------------------------------------
# Lightweight inline training fallback (30 days, not 7)
# ---------------------------------------------------------------------------

def _inline_train_fallback(site_id: str) -> dict | None:
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=HISTORY_DAYS_FORECAST_FALLBACK)
    df = _fetch_measurements(site_id, start, now)
    if df.empty:
        return None

    site_config = _fetch_site_config(site_id)
    load_profile = _compute_load_profile(df)
    load_std = _compute_load_std(df)
    pv_profile = _compute_pv_profile(df)
    ev_profile = _compute_ev_profile(df, site_config)

    return {
        "load_profile_96": load_profile,
        "load_std_96": load_std,
        "pv_profile_96": pv_profile,
        "pv_pr_ratio_96": None,
        "ev_profile_96": ev_profile,
        "has_measured_pv": pv_profile is not None,
        "has_pr_model": False,
    }

# ---------------------------------------------------------------------------
# Main forecast entry point
# ---------------------------------------------------------------------------

def run_forecast_for_site(site_id: str):
    now = datetime.now(timezone.utc)
    horizon_end = now + timedelta(hours=25)

    params = _load_model_params(site_id)
    data_source = "cached_model"

    if params is None:
        params = _inline_train_fallback(site_id)
        data_source = "inline_fallback_30d"
        if params is None:
            return {"error": "geen meetdata gevonden", "site_id": site_id}

    load_fc = np.array(params["load_profile_96"], dtype=float)
    load_std = np.array(params["load_std_96"], dtype=float)
    pv_profile = params.get("pv_profile_96")
    pv_pr_ratio = params.get("pv_pr_ratio_96")
    ev_fc = np.array(params["ev_profile_96"], dtype=float)

    site_config = _fetch_site_config(site_id)
    flex_location_id = site_config.get("flex_location_id")
    pv_capacity_kwp = site_config.get("pv_capacity_kwp", 0.0)
    config_available = pv_capacity_kwp > 0

    weather_15m_fc = pd.DataFrame()
    if flex_location_id:
        raw_15m = _fetch_weather_15m(flex_location_id, now - timedelta(minutes=15), horizon_end)
        if not raw_15m.empty:
            weather_15m_fc = raw_15m[
                raw_15m["ts_utc"] >= pd.Timestamp(now, tz="UTC")
            ].reset_index(drop=True)

    weather_fc = _fetch_weather_forecast(site_id, now - timedelta(hours=1), horizon_end)

    if not weather_fc.empty:
        load_fc = _apply_temperature_correction(load_fc, weather_fc)
    elif not weather_15m_fc.empty:
        load_fc = _apply_temperature_correction(load_fc, weather_15m_fc)

    pv_source = "none"
    pv_fc = np.zeros(96)
    start = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)

    if pv_pr_ratio is not None and not weather_15m_fc.empty and config_available:
        pr_ratio = np.array(pv_pr_ratio, dtype=float)
        weather_idx = weather_15m_fc.set_index("ts_utc")
        pv_fc_pr = np.zeros(96)

        for i in range(96):
            ts = start + timedelta(minutes=15 * i)
            ts_key = pd.Timestamp(ts, tz="UTC")
            slot = ts.hour * 4 + ts.minute // 15
            pr = pr_ratio[slot]
            if pr > 0 and ts_key in weather_idx.index:
                ghi = float(weather_idx.at[ts_key, "shortwave_radiation_wm2"])
                if ghi > 0:
                    temp_factor = 1.0
                    if "temperature_c" in weather_idx.columns:
                        temp_c = float(weather_idx.at[ts_key, "temperature_c"])
                        if math.isfinite(temp_c):
                            temp_factor = max(0.7, min(1.05, 1.0 - 0.004 * max(temp_c - 25.0, 0)))
                    pv_fc_pr[i] = ghi / 1000.0 * pr * pv_capacity_kwp * temp_factor

        pv_fc = _sanitize_array(np.clip(pv_fc_pr, 0, pv_capacity_kwp))
        pv_source = "pr_weather"

    elif pv_profile is not None:
        start_slot_idx = start.hour * 4 + start.minute // 15
        pv_fc = np.roll(np.array(pv_profile, dtype=float), -start_slot_idx)
        pv_source = "measured"

    elif not weather_fc.empty and config_available:
        pv_fc = _estimate_pv_from_weather(weather_fc, site_config)
        pv_source = "weather_physics"

    elif not weather_15m_fc.empty and config_available:
        physics_input = weather_15m_fc.rename(columns={
            "shortwave_radiation_wm2": "ghi_wm2",
            "direct_radiation_wm2": "dni_wm2",
            "diffuse_radiation_wm2": "dhi_wm2",
        })
        pv_fc = _estimate_pv_from_weather(physics_input, site_config)
        pv_source = "weather_physics_15m"

    else:
        pv_source = "no_pv_config" if not config_available else "none"

    lower = _sanitize_array(load_fc - load_std)
    upper = _sanitize_array(load_fc + load_std)

    load_fc = _sanitize_array(load_fc)
    pv_fc = _sanitize_array(pv_fc)
    ev_fc = _sanitize_array(ev_fc)

    start_slot = start.hour * 4 + start.minute // 15
    load_fc = np.roll(load_fc, -start_slot)
    ev_fc = np.roll(ev_fc, -start_slot)
    lower = np.roll(lower, -start_slot)
    upper = np.roll(upper, -start_slot)

    _save_forecast(site_id, load_fc, pv_fc, ev_fc, lower, upper, MODEL_ID)

    return {
        "site_id": site_id,
        "status": "forecast_created",
        "model_id": MODEL_ID,
        "data_source": data_source,
        "rows_written": 96,
        "weather_available": not weather_fc.empty or not weather_15m_fc.empty,
        "weather_15m_rows": len(weather_15m_fc),
        "config_available": config_available,
        "pv_source": pv_source,
        "ev_source": "measured" if np.any(ev_fc > 0) else "none",
        "totals": {
            "predicted_load_kwh": _safe_float(np.sum(load_fc) * 0.25),
            "predicted_pv_kwh": _safe_float(np.sum(pv_fc) * 0.25),
            "predicted_ev_kwh": _safe_float(np.sum(ev_fc) * 0.25),
            "predicted_net_kwh": _safe_float(np.sum(load_fc - pv_fc + ev_fc) * 0.25),
        },
        "sample": [
            {
                "predicted_load_kw": _safe_float(load_fc[0]),
                "predicted_pv_kw": _safe_float(pv_fc[0]),
                "predicted_ev_kw": _safe_float(ev_fc[0]),
                "predicted_net_kw": _safe_float(load_fc[0] - pv_fc[0] + ev_fc[0]),
            }
        ]
    }
