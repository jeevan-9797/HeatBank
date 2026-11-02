"""
Performance and savings metrics calculation for HeatBank simulation.

Computes KPI metrics, comfort violation statistics, and percentage savings with zero-division safety.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any, Union
import pandas as pd
import numpy as np


@dataclass
class ModeMetrics:
    """Aggregated performance metrics for a single operational mode."""
    total_grid_energy_kwh: float
    peak_period_grid_energy_kwh: float
    peak_demand_kw: float
    peak_period_demand_kw: float
    total_cost: float
    comfort_violation_hours: float
    max_indoor_temp: float
    min_indoor_temp: float
    mean_indoor_temp: float


@dataclass
class StorageMetrics:
    """Metrics specific to thermal storage utilization."""
    initial_soc_pct: float
    final_soc_pct: float
    max_soc_pct: float
    min_soc_pct: float
    total_energy_charged_kwh_th: float
    total_energy_discharged_kwh_th: float
    total_losses_kwh_th: float
    storage_utilization_pct: float  # (total_discharged / capacity) * 100
    peak_cooling_met_by_storage_pct: float  # % of cooling demand during peak met by storage


@dataclass
class SavingsMetrics:
    """Comparative savings of HeatBank versus Conventional AC baseline."""
    energy_savings_kwh: float
    energy_savings_pct: Optional[float]  # None if baseline is zero
    energy_savings_pct_str: str

    peak_energy_savings_kwh: float
    peak_energy_savings_pct: Optional[float]
    peak_energy_savings_pct_str: str

    cost_savings_dollars: float
    cost_savings_pct: Optional[float]
    cost_savings_pct_str: str

    peak_demand_reduction_kw: float
    peak_demand_reduction_pct: Optional[float]
    peak_demand_reduction_pct_str: str


@dataclass
class SimulationMetrics:
    """Consolidated KPI report containing Conventional AC, HeatBank, Storage, and Savings metrics."""
    conventional: ModeMetrics
    heatbank: ModeMetrics
    storage: StorageMetrics
    savings: SavingsMetrics


def _safe_pct_savings(baseline: float, actual: float) -> tuple[Optional[float], str]:
    """
    Safely compute percentage savings: (baseline - actual) / baseline * 100.
    Returns (float_value_or_None, formatted_string).
    Handles zero baseline cleanly without throwing ZeroDivisionError.
    Shows negative savings honestly when actual > baseline.
    """
    if abs(baseline) < 1e-6:
        if abs(actual) < 1e-6:
            return 0.0, "0.0%"
        return None, "N/A (zero baseline)"

    diff = baseline - actual
    pct = (diff / baseline) * 100.0

    sign = "+" if pct > 0 else ""
    return pct, f"{sign}{pct:.1f}%"


def calculate_metrics(df: pd.DataFrame, storage_capacity_kwh: float, dt_hours: float = 0.25) -> SimulationMetrics:
    """
    Compute comprehensive simulation metrics from step-level DataFrame.
    """
    if df.empty:
        raise ValueError("DataFrame is empty, cannot calculate metrics.")

    # 1. Conventional AC Metrics
    conv_total_energy = float(df["conv_grid_energy_kwh"].sum())
    conv_peak_mask = df["is_peak_period"]
    conv_peak_energy = float(df.loc[conv_peak_mask, "conv_grid_energy_kwh"].sum())
    conv_peak_demand = float(df["conv_grid_power_kw"].max())
    conv_peak_period_demand = float(df.loc[conv_peak_mask, "conv_grid_power_kw"].max()) if conv_peak_mask.any() else 0.0
    conv_cost = float(df["conv_cost"].sum())
    conv_violation_hours = float((~df["conv_in_comfort"]).sum() * dt_hours)
    conv_max_temp = float(df["conv_indoor_temp"].max())
    conv_min_temp = float(df["conv_indoor_temp"].min())
    conv_mean_temp = float(df["conv_indoor_temp"].mean())

    conventional_metrics = ModeMetrics(
        total_grid_energy_kwh=round(conv_total_energy, 2),
        peak_period_grid_energy_kwh=round(conv_peak_energy, 2),
        peak_demand_kw=round(conv_peak_demand, 2),
        peak_period_demand_kw=round(conv_peak_period_demand, 2),
        total_cost=round(conv_cost, 2),
        comfort_violation_hours=round(conv_violation_hours, 2),
        max_indoor_temp=round(conv_max_temp, 2),
        min_indoor_temp=round(conv_min_temp, 2),
        mean_indoor_temp=round(conv_mean_temp, 2),
    )

    # 2. HeatBank Metrics
    hb_total_energy = float(df["hb_grid_energy_kwh"].sum())
    hb_peak_energy = float(df.loc[conv_peak_mask, "hb_grid_energy_kwh"].sum())
    hb_peak_demand = float(df["hb_grid_power_kw"].max())
    hb_peak_period_demand = float(df.loc[conv_peak_mask, "hb_grid_power_kw"].max()) if conv_peak_mask.any() else 0.0
    hb_cost = float(df["hb_cost"].sum())
    hb_violation_hours = float((~df["hb_in_comfort"]).sum() * dt_hours)
    hb_max_temp = float(df["hb_indoor_temp"].max())
    hb_min_temp = float(df["hb_indoor_temp"].min())
    hb_mean_temp = float(df["hb_indoor_temp"].mean())

    heatbank_metrics = ModeMetrics(
        total_grid_energy_kwh=round(hb_total_energy, 2),
        peak_period_grid_energy_kwh=round(hb_peak_energy, 2),
        peak_demand_kw=round(hb_peak_demand, 2),
        peak_period_demand_kw=round(hb_peak_period_demand, 2),
        total_cost=round(hb_cost, 2),
        comfort_violation_hours=round(hb_violation_hours, 2),
        max_indoor_temp=round(hb_max_temp, 2),
        min_indoor_temp=round(hb_min_temp, 2),
        mean_indoor_temp=round(hb_mean_temp, 2),
    )

    # 3. Storage Metrics
    init_soc = float(df["hb_storage_soc_pct"].iloc[0])
    final_soc = float(df["hb_storage_soc_pct"].iloc[-1])
    max_soc = float(df["hb_storage_soc_pct"].max())
    min_soc = float(df["hb_storage_soc_pct"].min())
    total_charged_th = float((df["hb_charge_rate_kw_th"] * dt_hours).sum())
    total_discharged_th = float(df["hb_storage_drawn_kwh"].sum())
    total_losses_th = float(df["hb_storage_loss_kwh"].sum())

    storage_util = 0.0
    if storage_capacity_kwh > 0:
        storage_util = round((total_discharged_th / storage_capacity_kwh) * 100.0, 1)

    peak_cooling_demand_total = float((df.loc[conv_peak_mask, "hb_cooling_demand_kw"] * dt_hours).sum())
    peak_cooling_delivered_by_storage = float((df.loc[conv_peak_mask, "hb_storage_discharge_cooling_kw"] * dt_hours).sum())
    peak_met_by_storage_pct = 0.0
    if peak_cooling_demand_total > 1e-4:
        peak_met_by_storage_pct = round((peak_cooling_delivered_by_storage / peak_cooling_demand_total) * 100.0, 1)

    storage_metrics = StorageMetrics(
        initial_soc_pct=round(init_soc, 1),
        final_soc_pct=round(final_soc, 1),
        max_soc_pct=round(max_soc, 1),
        min_soc_pct=round(min_soc, 1),
        total_energy_charged_kwh_th=round(total_charged_th, 2),
        total_energy_discharged_kwh_th=round(total_discharged_th, 2),
        total_losses_kwh_th=round(total_losses_th, 2),
        storage_utilization_pct=storage_util,
        peak_cooling_met_by_storage_pct=peak_met_by_storage_pct,
    )

    # 4. Comparative Savings
    energy_sav_val, energy_sav_str = _safe_pct_savings(conv_total_energy, hb_total_energy)
    peak_energy_sav_val, peak_energy_sav_str = _safe_pct_savings(conv_peak_energy, hb_peak_energy)
    cost_sav_val, cost_sav_str = _safe_pct_savings(conv_cost, hb_cost)
    peak_demand_red_val, peak_demand_red_str = _safe_pct_savings(conv_peak_demand, hb_peak_demand)

    savings_metrics = SavingsMetrics(
        energy_savings_kwh=round(conv_total_energy - hb_total_energy, 2),
        energy_savings_pct=energy_sav_val,
        energy_savings_pct_str=energy_sav_str,
        peak_energy_savings_kwh=round(conv_peak_energy - hb_peak_energy, 2),
        peak_energy_savings_pct=peak_energy_sav_val,
        peak_energy_savings_pct_str=peak_energy_sav_str,
        cost_savings_dollars=round(conv_cost - hb_cost, 2),
        cost_savings_pct=cost_sav_val,
        cost_savings_pct_str=cost_sav_str,
        peak_demand_reduction_kw=round(conv_peak_demand - hb_peak_demand, 2),
        peak_demand_reduction_pct=peak_demand_red_val,
        peak_demand_reduction_pct_str=peak_demand_red_str,
    )

    return SimulationMetrics(
        conventional=conventional_metrics,
        heatbank=heatbank_metrics,
        storage=storage_metrics,
        savings=savings_metrics,
    )


def calculate_savings(conv_val: float, hb_val: float) -> tuple[float, Optional[float], str]:
    """Helper for ad-hoc savings calculation."""
    diff = conv_val - hb_val
    pct_val, pct_str = _safe_pct_savings(conv_val, hb_val)
    return diff, pct_val, pct_str
