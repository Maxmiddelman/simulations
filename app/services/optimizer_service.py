import math
import logging
import numpy as np
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from app.db import get_supabase, supabase as _default_client

_AMSTERDAM = ZoneInfo("Europe/Amsterdam")

logger = logging.getLogger("enersim-api.optimizer")

_client = _default_client

CONNECTION_ERRORS = ("Server disconnected", "RemoteProtocolError", "ConnectionError", "ConnectError", "ReadTimeout")

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

def _fetch_forecast(site_id: str) -> list[dict]:
    now = datetime.now(timezone.utc)
    start = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)
    end = start + timedelta(hours=24)

    def _query(client):
        return (
            client.table("forecast_predictions_15m")
            .select("ts_utc, predicted_load_kw, predicted_pv_kw, predicted_ev_kw, predicted_net_kw")
            .eq("site_id", site_id)
            .gte("ts_utc", start.isoformat())
            .lte("ts_utc", end.isoformat())
            .order("ts_utc")
            .execute()
        )

    result = _with_retry(_query)
    return result.data or []

def _fetch_da_prices(start: datetime, end: datetime) -> list[dict]:
    start_date = start.astimezone(_AMSTERDAM).date().isoformat()
    end_date = end.astimezone(_AMSTERDAM).date().isoformat()

    def _query(client):
        return (
            client.table("flex_trading_algo")
            .select("datum, isp_nummer, da_prijs")
            .gte("datum", start_date)
            .lte("datum", end_date)
            .order("datum")
            .order("isp_nummer")
            .execute()
        )

    try:
        result = _with_retry(_query)
        return result.data or []
    except Exception:
        return []

def _fetch_site_config(site_id: str) -> dict:
    def _query_site(client):
        return (
            client.table("forecast_sites")
            .select("id, flex_location_id")
            .eq("id", site_id)
            .maybeSingle()
            .execute()
        )

    site_result = _with_retry(_query_site)
    site_data = site_result.data
    if not site_data or not site_data.get("flex_location_id"):
        return {}

    def _query_location(client):
        return (
            client.table("flex_locations")
            .select(
                "batterij_vermogen, batterij_capaciteit, batterij_rte, batterij_dod, "
                "batterij_min_soc_pct, batterij_max_soc_pct, "
                "gecontracteerd_importvermogen, gecontracteerd_exportvermogen, piekvermogen"
            )
            .eq("id", site_data["flex_location_id"])
            .maybeSingle()
            .execute()
        )

    loc_result = _with_retry(_query_location)
    loc_data = loc_result.data
    if not loc_data:
        return {}

    return {
        "battery_power_kw": _safe_float(loc_data.get("batterij_vermogen")),
        "battery_capacity_kwh": _safe_float(loc_data.get("batterij_capaciteit")),
        "battery_rte": _safe_float(loc_data.get("batterij_rte"), 90.0) / 100.0,
        "battery_dod": _safe_float(loc_data.get("batterij_dod"), 90.0) / 100.0,
        "battery_min_soc_pct": _safe_float(loc_data.get("batterij_min_soc_pct"), 5.0),
        "battery_max_soc_pct": _safe_float(loc_data.get("batterij_max_soc_pct"), 95.0),
        "grid_import_limit_kw": _safe_float(loc_data.get("gecontracteerd_importvermogen")),
        "grid_export_limit_kw": _safe_float(loc_data.get("gecontracteerd_exportvermogen")),
        "peak_power_kw": _safe_float(loc_data.get("piekvermogen")),
    }

def _fetch_latest_soc(site_id: str, battery_capacity_kwh: float) -> float | None:
    """Fetch the latest actual battery SOC from ems_live_measurements."""
    def _query(client):
        return (
            client.table("ems_live_measurements")
            .select("soc_percent, soc_kwh")
            .eq("site_id", site_id)
            .order("ts_utc", desc=True)
            .limit(1)
            .maybeSingle()
            .execute()
        )
    try:
        result = _with_retry(_query)
        row = result.data
        if not row:
            return None
        if row.get("soc_kwh") is not None:
            return _safe_float(row["soc_kwh"])
        if row.get("soc_percent") is not None:
            pct = _safe_float(row["soc_percent"])
            return pct / 100.0 * battery_capacity_kwh
    except Exception:
        pass
    return None

def _optimize_battery_schedule(
    net_load: np.ndarray,
    prices: np.ndarray,
    battery_power_kw: float,
    battery_capacity_kwh: float,
    battery_rte: float,
    battery_dod: float,
    min_soc_pct: float,
    max_soc_pct: float,
    grid_import_limit_kw: float,
    peak_target_kw: float,
    initial_soc_kwh: float | None = None,
    grid_export_limit_kw: float = 0.0,
    pv_forecast: np.ndarray | None = None,
) -> np.ndarray:
    """
    Greedy battery scheduling that balances peak shaving, price arbitrage,
    and proactive PV curtailment prevention.
    Returns battery_kw array (positive = discharge, negative = charge).
    """
    n_slots = len(net_load)
    battery_schedule = np.zeros(n_slots)

    min_soc = battery_capacity_kwh * (min_soc_pct / 100.0)
    max_soc = battery_capacity_kwh * (max_soc_pct / 100.0)
    usable_capacity = max_soc - min_soc

    if initial_soc_kwh is not None and min_soc <= initial_soc_kwh <= max_soc:
        soc_kwh = initial_soc_kwh
    elif initial_soc_kwh is not None and initial_soc_kwh > max_soc:
        soc_kwh = max_soc
    elif initial_soc_kwh is not None and initial_soc_kwh < min_soc:
        soc_kwh = min_soc
    else:
        # No SOC reading: refuse to plan from a fictional battery level.
        return None

    dt_hours = 0.25

    # Phase 1: Peak shaving
    if peak_target_kw > 0:
        for i in range(n_slots):
            load_i = net_load[i]

            if load_i > peak_target_kw and soc_kwh > min_soc:
                needed_kw = min(load_i - peak_target_kw, battery_power_kw)
                max_discharge_kw = (soc_kwh - min_soc) / dt_hours
                discharge_kw = min(needed_kw, max_discharge_kw)
                battery_schedule[i] = discharge_kw
                soc_kwh -= discharge_kw * dt_hours

            elif load_i < peak_target_kw * 0.5 and soc_kwh < max_soc:
                headroom_kw = min(peak_target_kw * 0.5 - load_i, battery_power_kw)
                max_charge_kw = (max_soc - soc_kwh) / (dt_hours * battery_rte)
                charge_kw = min(headroom_kw, max_charge_kw)
                battery_schedule[i] = -charge_kw
                soc_kwh += charge_kw * dt_hours * battery_rte

    # Phase 2: Price arbitrage overlay
    if np.any(prices > 0):
        price_spread = np.max(prices) - np.min(prices)
        if price_spread > 20:
            median_price = np.median(prices)
            for i in range(n_slots):
                current_schedule = battery_schedule[i]
                remaining_power = battery_power_kw - abs(current_schedule)

                if remaining_power < 1.0:
                    continue

                if prices[i] > median_price * 1.3 and soc_kwh > min_soc:
                    max_discharge = min(remaining_power, (soc_kwh - min_soc) / dt_hours)
                    arb_discharge = max_discharge * 0.5
                    battery_schedule[i] += arb_discharge
                    soc_kwh -= arb_discharge * dt_hours

                elif prices[i] < median_price * 0.7 and prices[i] >= 0 and soc_kwh < max_soc:
                    max_charge = min(remaining_power, (max_soc - soc_kwh) / (dt_hours * battery_rte))
                    arb_charge = max_charge * 0.5
                    battery_schedule[i] -= arb_charge
                    soc_kwh += arb_charge * dt_hours * battery_rte

    # Phase 3: Enforce grid import limit
    if grid_import_limit_kw > 0:
        for i in range(n_slots):
            grid_after_battery = net_load[i] - battery_schedule[i]
            if grid_after_battery > grid_import_limit_kw:
                extra_discharge = grid_after_battery - grid_import_limit_kw
                # Only discharge if battery has energy above min SOC
                max_discharge_from_soc = (soc_kwh - min_soc) / dt_hours if soc_kwh > min_soc else 0
                actual_extra = min(extra_discharge, battery_power_kw - battery_schedule[i], max_discharge_from_soc)
                if actual_extra > 0:
                    battery_schedule[i] += actual_extra
                    soc_kwh -= actual_extra * dt_hours

    # Phase 4: Enforce grid export limit (negative grid = export)
    # If export exceeds the limit, charge the battery to absorb the surplus
    # instead of curtailing PV.
    if grid_export_limit_kw > 0:
        for i in range(n_slots):
            grid_after_battery = net_load[i] - battery_schedule[i]
            if grid_after_battery < -grid_export_limit_kw:
                excess_export = -grid_export_limit_kw - grid_after_battery
                charge_capacity = (max_soc - soc_kwh) / (dt_hours * battery_rte) if soc_kwh < max_soc else 0
                extra_charge = min(excess_export, battery_power_kw + battery_schedule[i], charge_capacity)
                if extra_charge > 0:
                    battery_schedule[i] -= extra_charge
                    soc_kwh += extra_charge * dt_hours * battery_rte

    # Phase 5: Final feasibility safety-net
    # Walk through the schedule one more time and clamp any battery action
    # that would violate SOC limits, ensuring all prior phases produce a
    # physically realizable schedule.
    soc_kwh_final = soc_kwh
    for i in range(n_slots):
        kw = battery_schedule[i]
        if kw > 0:  # discharge
            max_from_soc = (soc_kwh_final - min_soc) / dt_hours
            if kw > max_from_soc:
                battery_schedule[i] = max(max_from_soc, 0)
            soc_kwh_final -= battery_schedule[i] * dt_hours
        elif kw < 0:  # charge
            max_from_soc = (max_soc - soc_kwh_final) / (dt_hours * battery_rte)
            if abs(kw) > max_from_soc:
                battery_schedule[i] = -max(max_from_soc, 0)
            soc_kwh_final += abs(battery_schedule[i]) * dt_hours * battery_rte

    return battery_schedule

def _save_optimization(site_id: str, battery_schedule: np.ndarray, net_load: np.ndarray, prices: np.ndarray):
    now = datetime.now(timezone.utc)
    start = now.replace(minute=(now.minute // 15) * 15, second=0, microsecond=0)

    rows = []
    for i in range(len(battery_schedule)):
        ts = start + timedelta(minutes=15 * i)
        battery_kw = _safe_float(battery_schedule[i])
        grid_kw = _safe_float(net_load[i] - battery_schedule[i])
        price = _safe_float(prices[i]) if i < len(prices) else 0.0

        if price <= 0:
            continue

        rows.append({
            "site_id": site_id,
            "ts_utc": ts.isoformat(),
            "battery_kw": battery_kw,
            "grid_kw": grid_kw,
            "price_eur_mwh": price,
        })

    if not rows:
        print(f"optimizer_results_15m upsert skipped: no rows with valid prices for {site_id}")
        return False

    def _upsert(client):
        return (
            client.table("optimizer_results_15m")
            .upsert(rows, on_conflict="site_id,ts_utc")
            .execute()
        )

    try:
        _with_retry(_upsert)
        return True
    except Exception as e:
        print(f"optimizer_results_15m upsert skipped: {e}")
        return False

def _compute_soc_trajectory(
    battery_schedule: np.ndarray,
    battery_capacity_kwh: float,
    min_soc_pct: float,
    max_soc_pct: float,
    battery_rte: float,
    initial_soc_kwh: float | None,
) -> np.ndarray:
    """Compute the expected SOC (kWh) at each interval given the battery schedule."""
    n = len(battery_schedule)
    soc_arr = np.zeros(n)
    min_soc = battery_capacity_kwh * (min_soc_pct / 100.0)
    max_soc = battery_capacity_kwh * (max_soc_pct / 100.0)

    if initial_soc_kwh is not None and min_soc <= initial_soc_kwh <= max_soc:
        soc = initial_soc_kwh
    elif initial_soc_kwh is not None and initial_soc_kwh > max_soc:
        soc = max_soc
    elif initial_soc_kwh is not None and initial_soc_kwh < min_soc:
        soc = min_soc
    else:
        soc = (min_soc + max_soc) / 2.0

    dt_hours = 0.25
    for i in range(n):
        kw = battery_schedule[i]
        if kw > 0:  # discharge
            soc -= kw * dt_hours
        elif kw < 0:  # charge
            soc += abs(kw) * dt_hours * battery_rte
        soc = max(min_soc, min(max_soc, soc))
        soc_arr[i] = soc
    return soc_arr

def _commit_day_ahead_profile(
    site_id: str,
    target_date: str,
    battery_schedule: np.ndarray,
    net_load: np.ndarray,
    pv_forecast: np.ndarray,
    prices: np.ndarray,
    soc_trajectory: np.ndarray,
    start_ts: datetime,
    battery_capacity_kwh: float,
    min_soc_pct: float,
    max_soc_pct: float,
    battery_rte: float,
    initial_soc_kwh: float | None,
) -> dict:
    """
    Commit the finalized Day-Ahead profile to committed_day_ahead_profiles.

    This is the LOCK mechanism: once committed, the profile is immutable.
    If a committed profile already exists for this (site_id, target_date),
    the existing baseline is preserved -- it is NOT overwritten.
    A new profile_version is only created if no committed rows exist yet.
    """

    def _check_existing(client):
        return (
            client.table("committed_day_ahead_profiles")
            .select("id")
            .eq("site_id", site_id)
            .eq("target_date", target_date)
            .limit(1)
            .execute()
        )

    try:
        existing = _with_retry(_check_existing)
        if existing.data and len(existing.data) > 0:
            return {
                "committed": False,
                "reason": "already_committed",
                "message": f"Day-Ahead profile for {target_date} already committed and locked",
            }
    except Exception as e:
        print(f"committed_day_ahead_profiles check failed: {e}")
        return {"committed": False, "reason": "check_failed", "message": str(e)}

    rows = []
    for i in range(len(battery_schedule)):
        ts = start_ts + timedelta(minutes=15 * i)
        battery_kw = _safe_float(battery_schedule[i])
        grid_kw = _safe_float(net_load[i] - battery_schedule[i])
        price = _safe_float(prices[i]) if i < len(prices) else 0.0
        load_kw = _safe_float(net_load[i] + pv_forecast[i]) if i < len(pv_forecast) else 0.0
        pv_kw = _safe_float(pv_forecast[i]) if i < len(pv_forecast) else 0.0
        soc_kwh = _safe_float(soc_trajectory[i]) if i < len(soc_trajectory) else 0.0

        if price <= 0:
            continue

        rows.append({
            "site_id": site_id,
            "target_date": target_date,
            "ts_utc": ts.isoformat(),
            "committed_grid_kw": grid_kw,
            "planned_battery_kw": battery_kw,
            "expected_load_kw": load_kw,
            "expected_pv_kw": pv_kw,
            "expected_soc_kwh": soc_kwh,
            "da_price_eur_mwh": price,
            "profile_version": 1,
            "committed_at": datetime.now(timezone.utc).isoformat(),
        })

    def _insert(client):
        return (
            client.table("committed_day_ahead_profiles")
            .insert(rows)
            .execute()
        )

    try:
        _with_retry(_insert)
        return {
            "committed": True,
            "target_date": target_date,
            "slots_committed": len(rows),
            "profile_version": 1,
        }
    except Exception as e:
        print(f"committed_day_ahead_profiles insert failed: {e}")
        return {"committed": False, "reason": "insert_failed", "message": str(e)}

def run_optimizer_for_site(site_id: str, target_date: str | None = None):
    config = _fetch_site_config(site_id)

    battery_power = config.get("battery_power_kw", 0)
    battery_capacity = config.get("battery_capacity_kwh", 0)

    if battery_power <= 0 or battery_capacity <= 0:
        return {
            "status": "skipped",
            "site_id": site_id,
            "reason": "no_battery_config",
            "message": "Battery power or capacity not configured in flex_locations"
        }

    forecast_rows = _fetch_forecast(site_id)
    if not forecast_rows:
        return {
            "status": "skipped",
            "site_id": site_id,
            "reason": "no_forecast",
            "message": "No forecast predictions found for this site"
        }

    net_load = np.array([_safe_float(r.get("predicted_net_kw", 0)) for r in forecast_rows])
    pv_forecast = np.array([_safe_float(r.get("predicted_pv_kw", 0)) for r in forecast_rows])

    now = datetime.now(timezone.utc)
    start = now.replace(minute=0, second=0, microsecond=0)
    end = start + timedelta(hours=24)
    price_rows = _fetch_da_prices(start, end)

    prices = np.zeros(len(net_load))
    if price_rows:
        price_map: dict[str, float] = {}
        for pr in price_rows:
            price_map[f"{pr['datum']}_{pr['isp_nummer']}"] = _safe_float(pr.get("da_prijs", 0))
        for i in range(len(prices)):
            slot_time = start + timedelta(minutes=15 * i)
            local = slot_time.astimezone(_AMSTERDAM)
            slot_date = local.date().isoformat()
            isp_num = local.hour * 4 + local.minute // 15 + 1
            key = f"{slot_date}_{isp_num}"
            if key in price_map:
                prices[i] = price_map[key]

    battery_rte = config.get("battery_rte", 0.9)
    battery_dod = config.get("battery_dod", 0.9)
    min_soc_pct = config.get("battery_min_soc_pct", 5.0)
    max_soc_pct = config.get("battery_max_soc_pct", 95.0)
    grid_import_limit = config.get("grid_import_limit_kw", 0)
    grid_export_limit = config.get("grid_export_limit_kw", 0)
    peak_target = config.get("peak_power_kw", 0)

    if peak_target <= 0 and grid_import_limit > 0:
        peak_target = grid_import_limit * 0.8

    initial_soc = _fetch_latest_soc(site_id, battery_capacity)

    battery_schedule = _optimize_battery_schedule(
        net_load=net_load,
        prices=prices,
        battery_power_kw=battery_power,
        battery_capacity_kwh=battery_capacity,
        battery_rte=battery_rte,
        battery_dod=battery_dod,
        min_soc_pct=min_soc_pct,
        max_soc_pct=max_soc_pct,
        grid_import_limit_kw=grid_import_limit,
        peak_target_kw=peak_target,
        initial_soc_kwh=initial_soc,
        grid_export_limit_kw=grid_export_limit,
        pv_forecast=pv_forecast,
    )

    if battery_schedule is None:
        return {
            "status": "skipped",
            "site_id": site_id,
            "reason": "no_soc_reading",
            "message": "No fresh battery SOC reading available — refusing to plan from a fictional SOC",
        }

    saved = _save_optimization(site_id, battery_schedule, net_load, prices)

    commit_result = None
    if target_date is not None:
        soc_trajectory = _compute_soc_trajectory(
            battery_schedule=battery_schedule,
            battery_capacity_kwh=battery_capacity,
            min_soc_pct=min_soc_pct,
            max_soc_pct=max_soc_pct,
            battery_rte=battery_rte,
            initial_soc_kwh=initial_soc,
        )
        commit_result = _commit_day_ahead_profile(
            site_id=site_id,
            target_date=target_date,
            battery_schedule=battery_schedule,
            net_load=net_load,
            pv_forecast=pv_forecast,
            prices=prices,
            soc_trajectory=soc_trajectory,
            start_ts=start,
            battery_capacity_kwh=battery_capacity,
            min_soc_pct=min_soc_pct,
            max_soc_pct=max_soc_pct,
            battery_rte=battery_rte,
            initial_soc_kwh=initial_soc,
        )

    grid_without_battery = net_load
    grid_with_battery = net_load - battery_schedule

    peak_without = _safe_float(np.max(grid_without_battery))
    peak_with = _safe_float(np.max(grid_with_battery))
    peak_reduction = _safe_float(peak_without - peak_with)

    cost_without = _safe_float(np.sum(np.maximum(grid_without_battery, 0) * prices / 1000 * 0.25))
    cost_with = _safe_float(np.sum(np.maximum(grid_with_battery, 0) * prices / 1000 * 0.25))
    cost_saving = _safe_float(cost_without - cost_with)

    return {
        "status": "optimized",
        "site_id": site_id,
        "slots": len(battery_schedule),
        "saved_to_db": saved,
        "prices_available": bool(price_rows),
        "metrics": {
            "peak_without_kw": peak_without,
            "peak_with_kw": peak_with,
            "peak_reduction_kw": peak_reduction,
            "energy_cost_without_eur": cost_without,
            "energy_cost_with_eur": cost_with,
            "cost_saving_eur": cost_saving,
        },
        "battery_config": {
            "power_kw": battery_power,
            "capacity_kwh": battery_capacity,
            "rte": battery_rte,
            "dod": battery_dod,
            "min_soc_pct": min_soc_pct,
            "max_soc_pct": max_soc_pct,
        },
        "initial_soc_kwh": initial_soc,
        "committed_profile": commit_result,
    }
