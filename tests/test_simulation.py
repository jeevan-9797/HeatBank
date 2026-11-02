"""
Tests for full 24-hour simulation engine and building thermal dynamics.
"""

import pytest
import numpy as np
import math

from simulation.building import BuildingConfig, BuildingModel
from simulation.storage import ThermalStorageConfig
from simulation.scenarios import get_scenario_preset, ScenarioPreset
from simulation.controller import SimulationEngine


def test_building_config_validation():
    with pytest.raises(ValueError):
        BuildingConfig(thermal_mass=-5.0).validate()
    with pytest.raises(ValueError):
        BuildingConfig(heat_loss_coeff=-1.0).validate()
    with pytest.raises(ValueError):
        BuildingConfig(target_temp=50.0).validate()
    with pytest.raises(ValueError):
        BuildingConfig(ac_cop=0.0).validate()


def test_building_thermal_step_finite():
    cfg = BuildingConfig(thermal_mass=20.0, heat_loss_coeff=1.2, initial_temp=24.0, target_temp=22.0)
    model_uncooled = BuildingModel(cfg)
    model_cooled = BuildingModel(cfg)

    # Free floating step without cooling
    t_uncooled = model_uncooled.step(
        cooling_delivered_kw=0.0,
        outdoor_temp=35.0,
        internal_gain_kw=3.0,
        solar_irradiance_kw=4.0,
        dt_hours=0.25,
    )
    assert math.isfinite(t_uncooled)
    assert t_uncooled > 24.0  # Heat entered building

    # Step with cooling from identical initial conditions
    t_cooled = model_cooled.step(
        cooling_delivered_kw=10.0,
        outdoor_temp=35.0,
        internal_gain_kw=3.0,
        solar_irradiance_kw=4.0,
        dt_hours=0.25,
    )
    assert math.isfinite(t_cooled)
    assert t_cooled < t_uncooled


def test_simulation_runs_full_24_hours():
    building_cfg = BuildingConfig()
    storage_cfg = ThermalStorageConfig()
    scenario = get_scenario_preset("hot_sunny")

    engine = SimulationEngine(
        building_config=building_cfg,
        storage_config=storage_cfg,
        scenario=scenario,
        duration_hours=24.0,
        dt_hours=0.25,
    )

    result = engine.run()
    assert len(result.steps) == 96
    assert len(result.df) == 96
    assert result.duration_hours == 24.0

    # Ensure all telemetry fields are finite numbers
    for col in [
        "conv_indoor_temp",
        "hb_indoor_temp",
        "conv_grid_power_kw",
        "hb_grid_power_kw",
        "conv_cost",
        "hb_cost",
        "hb_storage_soc_pct",
        "hb_storage_kwh",
    ]:
        vals = result.df[col].to_numpy()
        assert np.all(np.isfinite(vals)), f"Column {col} contains non-finite values"

    # State of charge bounds
    soc_vals = result.df["hb_storage_soc_pct"].to_numpy()
    assert np.all(soc_vals >= -1e-5)
    assert np.all(soc_vals <= 100.0 + 1e-5)

    storage_kwh = result.df["hb_storage_kwh"].to_numpy()
    assert np.all(storage_kwh >= -1e-5)
    assert np.all(storage_kwh <= storage_cfg.capacity_kwh + 1e-5)


def test_simulation_with_zero_solar():
    """Ensure simulation works properly on a zero-solar scenario (e.g. night or 100% blackout)."""
    scenario = get_scenario_preset("hot_sunny")
    scenario.solar_peak_kw = 0.0  # Zero solar

    engine = SimulationEngine(
        building_config=BuildingConfig(),
        storage_config=ThermalStorageConfig(),
        scenario=scenario,
        duration_hours=24.0,
        dt_hours=0.25,
    )

    result = engine.run()
    assert len(result.steps) == 96
    solar_vals = result.df["solar_generation_kw"].to_numpy()
    assert np.all(solar_vals == 0.0)
    # Energy and cost calculations remain finite and non-negative
    assert np.all(result.df["conv_grid_power_kw"] >= 0.0)
    assert np.all(result.df["hb_grid_power_kw"] >= 0.0)


def test_simulation_with_zero_storage_capacity():
    """Ensure simulation functions cleanly when storage capacity is zero."""
    storage_cfg = ThermalStorageConfig(capacity_kwh=0.0, initial_soc_pct=0.0)
    scenario = get_scenario_preset("hot_sunny")

    engine = SimulationEngine(
        building_config=BuildingConfig(),
        storage_config=storage_cfg,
        scenario=scenario,
        duration_hours=24.0,
        dt_hours=0.25,
    )

    result = engine.run()
    assert len(result.steps) == 96
    assert np.all(result.df["hb_storage_kwh"] == 0.0)
    assert np.all(result.df["hb_storage_discharge_cooling_kw"] == 0.0)


def test_scenarios_change_actual_inputs():
    """Verify presets change actual simulation outputs, not just labels."""
    engine_sunny = SimulationEngine(
        BuildingConfig(),
        ThermalStorageConfig(),
        get_scenario_preset("hot_sunny"),
    )
    res_sunny = engine_sunny.run()

    engine_cloudy = SimulationEngine(
        BuildingConfig(),
        ThermalStorageConfig(),
        get_scenario_preset("hot_cloudy"),
    )
    res_cloudy = engine_cloudy.run()

    # Sunny should have significantly higher solar generation than cloudy
    assert res_sunny.df["solar_generation_kw"].max() > res_cloudy.df["solar_generation_kw"].max()
