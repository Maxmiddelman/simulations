"""
PyPSA-based battery dispatch optimizer for standalone trade calculations.

Uses linear programming via PyPSA + HiGHS to find the globally optimal
battery charge/discharge schedule that minimizes net energy cost (or
maximizes revenue) over a 96-slot (24h, 15-min) horizon.

Optimizes the full day in a single LP solve — no rolling 4h windows.
Degradation and SOC target are integrated into the LP objective.
"""
from __future__ import annotations

import math
import numpy as np
import pandas as pd
from typing import Optional
from pydantic import BaseModel, Field, field_validator

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
    target_soc_kwh: Optional[float] = None

    @field_validator("pv_kwh_96", "da_price_eur_per_mwh_96")
    @classmethod
    def validate_96_elements(cls, v):
        if len(v) != 96:
            raise ValueError(f"Array must have exactly 96 elements, got {len(v)}")
        return v

    @field_validator("id_price_eur_per_mwh_96", "load_kwh_96")
    @classmethod
    def validate_96_optional(cls, v):
        if v is not None and len(v) != 96:
            raise ValueError(f"Array must have exactly 96 elements, got {len(v)}")
        return v

    @field_validator("battery_power_kw")
    @classmethod
    def validate_power_positive(cls, v):
        if v <= 0:
            raise ValueError("battery_power_kw must be > 0")
        return v

    @field_validator("battery_capacity_kwh")
    @classmethod
    def validate_capacity_positive(cls, v):
        if v <= 0:
            raise ValueError("battery_capacity_kwh must be > 0")
        return v

def _build_pypsa_network(req: StandaloneOptimizeRequest):
    import pypsa

    n_slots = 96
    dt_hours = 0.25

    min_pct = max(0, min(100, req.min_dod_pct or 0))
    max_pct = max(min_pct, min(100, req.max_dod_pct or (req.dod * 100)))
    min_soc = req.battery_capacity_kwh * (min_pct / 100)
    max_soc = req.battery_capacity_kwh * (max_pct / 100)
    usable_band = max(0.001, max_soc - min_soc)

    snapshots = pd.date_range(
        start=f"{req.date}T00:00:00", periods=n_slots, freq="15min"
    )

    network = pypsa.Network()
    network.set_snapshots(snapshots)

    # Single AC bus
    network.add("Bus", "grid", v_nom=400.0)

    # --- Import generator (buying from grid) ---
    import_marginal_cost = (
        np.array(req.da_price_eur_per_mwh_96) / 1000.0
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
    export_revenue = (
        -(np.array(req.da_price_eur_per_mwh_96) / 1000.0)
        - req.export_transport_eur_per_mwh / 1000.0
        - req.handel_cost_eur_per_mwh / 1000.0
        - req.clearing_cost_eur_per_mwh / 1000.0
        + req.sde_rate_per_kwh_eur
    )
    export_marginal = export_revenue.copy()
    for i in range(n_slots):
        if req.da_price_eur_per_mwh_96[i] < req.min_revenue_eur_per_mwh:
            export_marginal[i] = 999.0

    network.add(
        "Generator",
        "grid_export",
        bus="grid",
        p_nom=req.max_export_kw,
        p_max_pu=[1.0] * n_slots,
        marginal_cost=export_marginal.tolist(),
        sign=-1,
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

    shifted_initial = max(0, req.initial_soc_kwh - min_soc)

    degradation_marginal = req.degradation_cost_eur_per_kwh / dt_hours

    use_cyclic = req.target_soc_kwh is not None

    network.add(
        "StorageUnit",
        "battery",
        bus="grid",
        p_nom=req.battery_power_kw,
        max_hours=usable_band / req.battery_power_kw,
        efficiency_store=sqrt_rte,
        efficiency_dispatch=sqrt_rte,
        standing_loss=0.0,
        cyclic_state_of_charge=use_cyclic,
        state_of_charge_initial=shifted_initial,
        marginal_cost=degradation_marginal,
    )

    # --- SOC target constraint ---
    if req.target_soc_kwh is not None:
        shifted_target = max(0, req.target_soc_kwh - min_soc)
        if use_cyclic:
            network.storage_units.loc["battery", "state_of_charge_initial"] = shifted_target
        else:
            try:
                model = network.optimize.create_model(solver_name="highs")
                soc = model.variables["StorageUnit-state_of_charge"]
                last_snapshot = network.snapshots[-1]
                model.add_constraints(
                    soc.loc[last_snapshot, "battery"] >= shifted_target,
                    name="soc_target_final",
                )
            except Exception:
                pass

    return network, min_soc, max_soc, usable_band

def run_standalone_optimize(req: StandaloneOptimizeRequest) -> dict:
    """
    Run PyPSA LP optimization for a single day (96 slots).
    Returns a dict matching the existing edge function response format.
    """
    n_slots = 96
    dt_hours = 0.25

    # Early validation — prevents cryptic PyPSA errors
    if req.battery_power_kw <= 0:
        return {
            "success": False,
            "error": "battery_power_kw must be > 0",
            "status": "invalid_input",
        }
    if req.battery_capacity_kwh <= 0:
        return {
            "success": False,
            "error": "battery_capacity_kwh must be > 0",
            "status": "invalid_input",
        }

    network, min_soc, max_soc, usable_band = _build_pypsa_network(req)

    try:
        network.optimize(solver_name="highs")
    except Exception as exc:
        import linopy
        return {
            "success": False,
            "error": f"Optimalisatie faalde: {exc}",
            "status": "solver_error",
            "linopy_version": getattr(linopy, "__version__", "onbekend"),
        }

    if network.objective is None:
        return {
            "success": False,
            "error": "Optimization failed — no solution found",
            "status": "infeasible",
        }

    # Extract results
    grid_import = network.generators_t.p["grid_import"].values
    grid_export_vals = network.generators_t.p.get("grid_export")
    if grid_export_vals is not None:
        grid_export_kw = grid_export_vals.values
    else:
        grid_export_kw = np.zeros(n_slots)

    pv_power = np.zeros(n_slots)
    if "pv" in network.generators_t.p.columns:
        pv_power = network.generators_t.p["pv"].values

    battery_dispatch = network.storage_units_t.p["battery"].values
    battery_soc = network.storage_units_t.state_of_charge["battery"].values

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

        soc_real = battery_soc[i] + min_soc

        import_cost = import_kwh * da_prices[i] / 1000.0
        export_revenue = export_kwh * da_prices[i] / 1000.0
        net_cost = import_cost - export_revenue

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

    final_soc_real = battery_soc[-1] + min_soc if len(battery_soc) > 0 else req.initial_soc_kwh

    net_total_cost = float(network.objective)
    revenue_eur = -net_total_cost
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
