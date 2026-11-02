"""
Tests for metrics calculation and comparative savings.
"""

import pytest
import pandas as pd
import numpy as np

from simulation.metrics import _safe_pct_savings, calculate_metrics, calculate_savings
from simulation.building import BuildingConfig
from simulation.storage import ThermalStorageConfig
from simulation.scenarios import get_scenario_preset
from simulation.controller import SimulationEngine


def test_safe_pct_savings_division_by_zero():
    # Both zero
    val, str_rep = _safe_pct_savings(0.0, 0.0)
    assert val == 0.0
    assert "0.0%" in str_rep

    # Baseline zero, actual > 0
    val, str_rep = _safe_pct_savings(0.0, 15.0)
    assert val is None
    assert "N/A" in str_rep

    # Normal savings
    val, str_rep = _safe_pct_savings(100.0, 70.0)
    assert val == 30.0
    assert "+30.0%" in str_rep

    # Honest negative savings (HeatBank performed worse)
    val, str_rep = _safe_pct_savings(100.0, 115.0)
    assert val == -15.0
    assert "-15.0%" in str_rep


def test_metrics_calculation_consistency():
    engine = SimulationEngine(
        BuildingConfig(),
        ThermalStorageConfig(capacity_kwh=50.0),
        get_scenario_preset("hot_sunny"),
    )
    result = engine.run()
    metrics = calculate_metrics(result.df, storage_capacity_kwh=50.0, dt_hours=0.25)

    # Check conservation & sanity
    assert metrics.conventional.total_grid_energy_kwh >= 0.0
    assert metrics.heatbank.total_grid_energy_kwh >= 0.0
    assert metrics.conventional.total_cost >= 0.0
    assert metrics.heatbank.total_cost >= 0.0

    # Comfort violation hours within [0, 24]
    assert 0.0 <= metrics.conventional.comfort_violation_hours <= 24.0
    assert 0.0 <= metrics.heatbank.comfort_violation_hours <= 24.0

    # Storage utilization sanity
    assert 0.0 <= metrics.storage.max_soc_pct <= 100.0
    assert 0.0 <= metrics.storage.min_soc_pct <= 100.0
    assert metrics.storage.total_energy_charged_kwh_th >= 0.0

    # Savings checks
    assert metrics.savings.energy_savings_kwh == round(
        metrics.conventional.total_grid_energy_kwh - metrics.heatbank.total_grid_energy_kwh, 2
    )
    assert metrics.savings.cost_savings_dollars == round(
        metrics.conventional.total_cost - metrics.heatbank.total_cost, 2
    )


def test_metrics_empty_dataframe_raises():
    with pytest.raises(ValueError):
        calculate_metrics(pd.DataFrame(), storage_capacity_kwh=50.0)
