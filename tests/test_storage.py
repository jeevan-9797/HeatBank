"""
Tests for ThermalStorage class.
"""

import pytest
import math
from simulation.storage import ThermalStorageConfig, ThermalStorage


def test_storage_initialization():
    cfg = ThermalStorageConfig(
        capacity_kwh=50.0,
        initial_soc_pct=40.0,
        max_charge_rate_kw=10.0,
        max_discharge_rate_kw=15.0,
    )
    storage = ThermalStorage(cfg)
    assert storage.capacity_kwh == 50.0
    assert math.isclose(storage.current_kwh, 20.0, rel_tol=1e-5)
    assert math.isclose(storage.soc_pct, 40.0, rel_tol=1e-5)


def test_storage_bounds_never_exceeded():
    cfg = ThermalStorageConfig(
        capacity_kwh=50.0,
        initial_soc_pct=95.0,
        max_charge_rate_kw=20.0,
        charge_efficiency=0.9,
    )
    storage = ThermalStorage(cfg)

    # Attempt to overcharge with a massive charge request
    p_ch, e_elec, delta_s = storage.charge(100.0, dt_hours=1.0)
    assert storage.current_kwh <= 50.0
    assert storage.soc_pct <= 100.0
    assert delta_s <= 2.5 + 1e-6  # Room was 2.5 kWh

    # Attempt to over-discharge
    storage_empty_test = ThermalStorage(ThermalStorageConfig(capacity_kwh=50.0, initial_soc_pct=5.0))
    p_dis, drawn = storage_empty_test.discharge(100.0, dt_hours=1.0)
    assert storage_empty_test.current_kwh >= 0.0
    assert storage_empty_test.soc_pct >= 0.0


def test_charge_discharge_rate_limits():
    max_ch = 8.0
    max_dis = 12.0
    cfg = ThermalStorageConfig(
        capacity_kwh=100.0,
        initial_soc_pct=50.0,
        max_charge_rate_kw=max_ch,
        max_discharge_rate_kw=max_dis,
    )
    storage = ThermalStorage(cfg)

    # Request higher than limit
    actual_ch, _, _ = storage.charge(50.0, dt_hours=0.25)
    assert actual_ch <= max_ch + 1e-6

    actual_dis, _ = storage.discharge(50.0, dt_hours=0.25)
    assert actual_dis <= max_dis + 1e-6


def test_energy_accounting_in_storage():
    cop_ch = 3.0
    eff_ch = 0.90
    cfg = ThermalStorageConfig(
        capacity_kwh=100.0,
        initial_soc_pct=0.0,
        max_charge_rate_kw=10.0,
        charge_efficiency=eff_ch,
        chiller_cop_charge=cop_ch,
    )
    storage = ThermalStorage(cfg)

    power_th, elec_kwh, stored_delta = storage.charge(10.0, dt_hours=0.5)
    # Expected power = 10 kW
    assert math.isclose(power_th, 10.0, rel_tol=1e-5)
    # Expected electric consumption: (10 kW * 0.5 h) / 3.0 = 1.6667 kWh
    expected_elec = (10.0 * 0.5) / cop_ch
    assert math.isclose(elec_kwh, expected_elec, rel_tol=1e-5)
    # Stored thermal delta: 10 kW * 0.5 h * 0.9 = 4.5 kWh_th
    assert math.isclose(stored_delta, 10.0 * 0.5 * eff_ch, rel_tol=1e-5)
    assert math.isclose(storage.current_kwh, 4.5, rel_tol=1e-5)


def test_zero_capacity_storage():
    """Ensure zero capacity does not trigger exceptions and returns zero."""
    cfg = ThermalStorageConfig(capacity_kwh=0.0, initial_soc_pct=0.0)
    storage = ThermalStorage(cfg)

    assert storage.available_energy_kwh == 0.0
    assert storage.soc_pct == 0.0

    p_ch, e_elec, delta_s = storage.charge(10.0, dt_hours=0.25)
    assert p_ch == 0.0
    assert e_elec == 0.0
    assert delta_s == 0.0

    p_dis, drawn = storage.discharge(10.0, dt_hours=0.25)
    assert p_dis == 0.0
    assert drawn == 0.0

    loss = storage.apply_standing_loss(0.25)
    assert loss == 0.0


def test_standing_losses():
    loss_rate = 0.02  # 2% per hour
    cfg = ThermalStorageConfig(
        capacity_kwh=100.0,
        initial_soc_pct=50.0,
        loss_rate_per_hour=loss_rate,
    )
    storage = ThermalStorage(cfg)
    initial_kwh = storage.current_kwh  # 50 kWh

    loss = storage.apply_standing_loss(dt_hours=1.0)
    assert math.isclose(loss, initial_kwh * loss_rate, rel_tol=1e-5)
    assert math.isclose(storage.current_kwh, initial_kwh * (1.0 - loss_rate), rel_tol=1e-5)


def test_invalid_storage_config_rejected():
    with pytest.raises(ValueError):
        ThermalStorageConfig(capacity_kwh=-10.0).validate()
    with pytest.raises(ValueError):
        ThermalStorageConfig(initial_soc_pct=105.0).validate()
    with pytest.raises(ValueError):
        ThermalStorageConfig(charge_efficiency=0.0).validate()
    with pytest.raises(ValueError):
        ThermalStorageConfig(discharge_efficiency=1.5).validate()
    with pytest.raises(ValueError):
        ThermalStorageConfig(loss_rate_per_hour=-0.01).validate()
    with pytest.raises(ValueError):
        ThermalStorageConfig(chiller_cop_charge=-1.0).validate()
