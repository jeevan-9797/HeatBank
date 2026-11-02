"""
Additional edge-case and energy consistency tests for HeatBank simulation.
"""

import pytest
import math
import numpy as np

from simulation.building import BuildingConfig, BuildingModel
from simulation.storage import ThermalStorageConfig, ThermalStorage
from simulation.scenarios import get_scenario_preset
from simulation.controller import SimulationEngine
from simulation.metrics import calculate_metrics, _safe_pct_savings


def test_energy_accounting_consistency_over_24h():
    """
    Verify that energy accounting is strictly preserved:
    1. Grid energy at each step equals grid power * dt.
    2. Cost at each step equals grid energy * price.
    3. Total cooling delivered equals storage discharge + backup AC.
    4. For HeatBank, total electric power equals backup AC power + charging electric power.
    """
    engine = SimulationEngine(
        BuildingConfig(),
        ThermalStorageConfig(),
        get_scenario_preset("hot_sunny"),
        dt_hours=0.25,
    )
    result = engine.run()
    df = result.df

    # 1. Grid energy consistency
    np.testing.assert_allclose(df["conv_grid_energy_kwh"], df["conv_grid_power_kw"] * 0.25, rtol=1e-5)
    np.testing.assert_allclose(df["hb_grid_energy_kwh"], df["hb_grid_power_kw"] * 0.25, rtol=1e-5)

    # 2. Step cost consistency
    np.testing.assert_allclose(df["conv_cost"], df["conv_grid_energy_kwh"] * df["electricity_price"], rtol=1e-5)
    np.testing.assert_allclose(df["hb_cost"], df["hb_grid_energy_kwh"] * df["electricity_price"], rtol=1e-5)

    # 3. HeatBank cooling balance
    np.testing.assert_allclose(
        df["hb_cooling_delivered_kw"],
        df["hb_storage_discharge_cooling_kw"] + df["hb_backup_ac_cooling_kw"],
        rtol=1e-5,
    )

    # 4. Total electric power balance before solar net metering
    np.testing.assert_allclose(
        df["hb_total_electric_power_kw"],
        df["hb_backup_ac_electric_kw"] + df["hb_charge_electric_kw"],
        rtol=1e-5,
    )


def test_high_loss_scenario_honest_negative_savings():
    """
    When thermal storage has extreme standing losses (e.g. 25%/hr) and low efficiency,
    HeatBank consumes MORE total energy than conventional AC.
    The simulation and metrics must honestly report negative savings without crashing.
    """
    inefficient_storage = ThermalStorageConfig(
        capacity_kwh=80.0,
        initial_soc_pct=0.0,
        charge_efficiency=0.50,
        discharge_efficiency=0.50,
        loss_rate_per_hour=0.25,  # Very leaky
    )
    # Scenario with flat electricity price so there's no TOU tariff advantage
    flat_price_scenario = get_scenario_preset("hot_sunny")
    flat_price_scenario.off_peak_price = 0.20
    flat_price_scenario.mid_peak_price = 0.20
    flat_price_scenario.peak_price = 0.20
    flat_price_scenario.solar_peak_kw = 0.0

    engine = SimulationEngine(
        BuildingConfig(),
        inefficient_storage,
        flat_price_scenario,
    )
    result = engine.run()
    metrics = calculate_metrics(result.df, storage_capacity_kwh=80.0)

    # HeatBank should have negative energy savings because of high round-trip losses
    assert metrics.savings.energy_savings_kwh < 0.0
    assert metrics.savings.energy_savings_pct < 0.0
    assert "-" in metrics.savings.energy_savings_pct_str


def test_custom_temperature_offset():
    """Verify custom temperature offset elevates the entire temperature curve."""
    sc_base = get_scenario_preset("hot_sunny")
    sc_base.custom_temp_offset = 0.0

    sc_offset = get_scenario_preset("hot_sunny")
    sc_offset.custom_temp_offset = 5.0

    eng_base = SimulationEngine(BuildingConfig(), ThermalStorageConfig(), sc_base)
    eng_offset = SimulationEngine(BuildingConfig(), ThermalStorageConfig(), sc_offset)

    res_base = eng_base.run()
    res_offset = eng_offset.run()

    np.testing.assert_allclose(
        res_offset.df["outdoor_temp"],
        res_base.df["outdoor_temp"] + 5.0,
        rtol=1e-5,
    )
