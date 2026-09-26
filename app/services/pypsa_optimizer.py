"""
PyPSA-based battery dispatch optimizer for standalone trade calculations.

Uses linear programming via PyPSA + HiGHS to find the globally optimal
battery charge/discharge schedule that minimizes net energy cost (or
maximizes revenue) over a 96-slot (24h, 15-min) horizon.

This replaces the external Render solver and the greedy heuristic.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import Optional
from pydantic import BaseModel, Field

class StandaloneOptimizeRequest(BaseModel):
    """Input schema matching the existing DayOptimizerRequest from the edge function."""
    date: str
    pv_kwh_96: list[float]
    da_price_eur_per_mwh_96: list[float]
    id_price_eur_per_mwh_96: Optional[list[float]] = None
    initial_soc_kwh: float
    battery_power_kw: float
    battery_capacity_kwh: float
    dod: float = 0.9
    min_dod_pct: Optional[float] = None
    max_dod_pct: Optional[float] = None
    round_trip_efficiency: float = 0.9
    degradation_cost_eur_per_kwh: float = 0.02
    energy_tax_eur_per_kwh: float = 0.0
    supplier_margin_eur_per_kwh: float = 0.0
    include_kw_max: bool = False
    kw_max_free_threshold_kw: float = 50.0
    kw_max_mid_threshold_kw: float = 2000.0
    kw_max_mid_tariff_eur_per_kw: float = 3.57
    kw_max_high_tariff_eur_per_kw: float = 6.24
    max_import_kw: float = 9999.0
    max_export_kw: float = 9999.0
    min_revenue_eur_per_mwh: float = 0.0
    sde_rate_per_kwh_eur: float = 0.0
    load_kwh_96: Optional[list[float]] = None
    import_transport_eur_per_mwh: float = 0.0
    export_transport_eur_per_mwh: float = 0.0
    handel_cost_eur_per_mwh: float = 0.0
    clearing_cost_eur_per_mwh: float = 0.0

def _build_pypsa_network(req: StandaloneOptimizeRequest) -> "pypsa.Network":
    import pypsa

    n_slots = 96
    dt_hours = 0.25

    min_pct = max(0, min(100, req.min_dod_pct or 0))
    max_pct = max(min_pct, min(100, req.max_dod_pct or (req.dod * 100)))
    min_soc = req.battery_capacity_kwh * (min_pct / 100)
    max_soc = req.battery_capacity_kwh * (max_pct / 100)
    usable_band = max(0.001, max_soc - min_soc)

    # Build time index (15-min snapshots)
    # PyPSA requires timezone-naive timestamps; timezone-aware datetime64[ns]
    # with tz raises "objects with timezones are not supported in snapshots".
    snapshots = pd.date_range(
        start=f"{req.date}T00:00:00", periods=n_slots, freq="15min"
    )

    network = pypsa.Network()
    network.set_snapshots(snapshots)

    # Single AC bus
    network.add("Bus", "grid", v_nom=400.0)

    # --- Import generator (buying from grid) ---
    import_marginal_cost = (
        np.array(req.da_price_eur_per_mwh_96) / 1000.0  # EUR/kWh
        + req.energy_tax_eur_per_kwh
        + req.supplier_margin_eur_per_kwh
        + req.import_transport_eur_per_mwh / 1000.0
        + req.handel_cost_eur_per_mwh / 1000.0
        + req.clearing_cost_eur_per_mwh / 1000.0
    )
    network.add(
        "Generator",
        "grid_import",
        bus="grid",
        p_nom=req.max_import_kw,
        p_max_pu=[1.0] * n_slots,
        marginal_cost=import_marginal_cost.tolist(),
    )

    # --- Export generator (selling to grid, negative cost = revenue) ---
    # SDE subsidy adds to export revenue
    export_revenue = (
        -(np.array(req.da_price_eur_per_mwh_96) / 1000.0)
        - req.export_transport_eur_per_mwh / 1000.0
        - req.handel_cost_eur_per_mwh / 1000.0
        - req.clearing_cost_eur_per_mwh / 1000.0
        + req.sde_rate_per_kwh_eur
    )
    # Min revenue threshold: if DA price < min_revenue, don't export
    export_marginal = export_revenue.copy()
    for i in range(n_slots):
        if req.da_price_eur_per_mwh_96[i] < req.min_revenue_eur_per_mwh:
            export_marginal[i] = 999.0  # effectively prevent export

    network.add(
        "Generator",
        "grid_export",
        bus="grid",
        p_nom=req.max_export_kw,
        p_max_pu=[1.0] * n_slots,
        marginal_cost=export_marginal.tolist(),
        sign=-1,  # negative sign = export
    )

    # --- PV generator (zero cost, fixed profile) ---
    pv_kwh = np.array(req.pv_kwh_96)
    if pv_kwh.sum() > 0:
        pv_kw = pv_kwh / dt_hours
        network.add(
            "Generator",
            "pv",
            bus="grid",
            p_nom=max(pv_kw.max(), 0.001),
            p_max_pu=(pv_kw / max(pv_kw.max(), 0.001)).tolist(),
            marginal_cost=0.0,
        )

    # --- Load (fixed profile) ---
    load_kwh = np.array(req.load_kwh_96 or [0.0] * n_slots)
    if load_kwh.sum() > 0:
        load_kw = load_kwh / dt_hours
        network.add(
            "Load",
            "demand",
            bus="grid",
            p_set=load_kw.tolist(),
        )

    # --- Battery storage ---
    rte = req.round_trip_efficiency
    sqrt_rte = math.sqrt(rte)

    # Shift initial SOC into usable band coordinates
    shifted_initial = max(0, req.initial_soc_kwh - min_soc)

    network.add(
        "StorageUnit",
        "battery",
        bus="grid",
        p_nom=req.battery_power_kw,
        max_hours=usable_band / req.battery_power_kw,
        efficiency_store=sqrt_rte,
        efficiency_dispatch=sqrt_rte,
        standing_loss=0.0,
        cyclic=False,
        state_of_charge_initial=shifted_initial,
        state_of_charge_min=0.0,
        state_of_charge_max=usable_band,
    )

    return network, min_soc, max_soc, usable_band

def run_standalone_optimize(req: StandaloneOptimizeRequest) -> dict:
    """
    Run PyPSA LP optimization for a single day (96 slots).

    Returns a dict matching the existing edge function response format:
    {
        success, date, status, revenue_eur, charge_kwh, discharge_kwh,
        import_kwh, export_kwh, pv_total_kwh, avg_da_price_eur_mwh,
        final_soc_kwh, net_total_cost_eur, energy_cost_eur,
        degradation_cost_eur, schedule_96
    }
    """
    import pypsa

    n_slots = 96
    dt_hours = 0.25

    network, min_soc, max_soc, usable_band = _build_pypsa_network(req)

    # Solve with HiGHS LP
    network.optimize(solver_name="highs")

    if network.objective is None:
        return {
            "success": False,
            "error": "Optimization failed — no solution found",
            "status": "infeasible",
        }

    # Extract results
    snapshots = network.snapshots

    # Grid power at each snapshot (positive = import, negative = export)
    grid_import = network.generators_t.p["grid_import"].values  # kW
    grid_export_vals = network.generators_t.p.get("grid_export")
    if grid_export_vals is not None:
        grid_export_kw = grid_export_vals.values
    else:
        grid_export_kw = np.zeros(n_slots)

    # PV production
    pv_power = np.zeros(n_slots)
    if "pv" in network.generators_t.p.columns:
        pv_power = network.generators_t.p["pv"].values

    # Battery dispatch (positive = discharge, negative = charge)
    battery_dispatch = network.storage_units_t.p["battery"].values  # kW
    battery_soc = network.storage_units_t.state_of_charge["battery"].values  # kWh

    # Build per-slot schedule
    schedule_96 = []
    total_charge_kwh = 0.0
    total_discharge_kwh = 0.0
    total_import_kwh = 0.0
    total_export_kwh = 0.0
    total_pv_kwh = 0.0
    total_degradation_cost = 0.0

    da_prices = np.array(req.da_price_eur_per_mwh_96)

    for i in range(n_slots):
        charge_kw = max(0, -battery_dispatch[i]) if battery_dispatch[i] < 0 else 0.0
        discharge_kw = max(0, battery_dispatch[i]) if battery_dispatch[i] > 0 else 0.0
        charge_kwh = charge_kw * dt_hours
        discharge_kwh = discharge_kw * dt_hours
        import_kwh = max(0, grid_import[i]) * dt_hours
        export_kwh = max(0, grid_export_kw[i]) * dt_hours
        pv_kwh = pv_power[i] * dt_hours

        # SOC in real coordinates (shift back)
        soc_real = battery_soc[i] + min_soc

        # Net cost per slot (import cost - export revenue)
        import_cost = import_kwh * da_prices[i] / 1000.0
        export_revenue = export_kwh * da_prices[i] / 1000.0
        net_cost = import_cost - export_revenue

        # Degradation cost (per kWh throughput)
        degradation = (charge_kwh + discharge_kwh) * req.degradation_cost_eur_per_kwh / 2

        total_charge_kwh += charge_kwh
        total_discharge_kwh += discharge_kwh
        total_import_kwh += import_kwh
        total_export_kwh += export_kwh
        total_pv_kwh += pv_kwh
        total_degradation_cost += degradation

        schedule_96.append({
            "charge_kwh": round(charge_kwh, 4),
            "discharge_kwh": round(discharge_kwh, 4),
            "import_kwh": round(import_kwh, 4),
            "export_kwh": round(export_kwh, 4),
            "soc_end_kwh": round(soc_real, 4),
            "net_cost_eur": round(net_cost - degradation, 4),
        })

    # Final SOC in real coordinates
    final_soc_real = battery_soc[-1] + min_soc if len(battery_soc) > 0 else req.initial_soc_kwh

    # Total net cost from objective (LP minimizes cost)
    net_total_cost = float(network.objective)

    # Revenue = negative net cost
    revenue_eur = -net_total_cost

    # Energy cost (import cost - export revenue, before degradation)
    energy_cost = sum(s["net_cost_eur"] for s in schedule_96)

    avg_da_price = float(np.mean(da_prices)) if len(da_prices) > 0 else 0.0

    return {
        "success": True,
        "date": req.date,
        "status": "optimal",
        "revenue_eur": round(revenue_eur, 2),
        "charge_kwh": round(total_charge_kwh, 2),
        "discharge_kwh": round(total_discharge_kwh, 2),
        "import_kwh": round(total_import_kwh, 2),
        "export_kwh": round(total_export_kwh, 2),
        "pv_total_kwh": round(total_pv_kwh, 2),
        "avg_da_price_eur_mwh": round(avg_da_price, 2),
        "final_soc_kwh": round(final_soc_real, 2),
        "net_total_cost_eur": round(net_total_cost, 2),
        "energy_cost_eur": round(energy_cost, 2),
        "degradation_cost_eur": round(total_degradation_cost, 2),
        "schedule_96": schedule_96,
    }
