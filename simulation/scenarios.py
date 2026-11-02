"""
Scenario presets and environmental profile generators for HeatBank simulation.

Generates 24-hour profiles (96 time steps at 15-min intervals) for:
- Outdoor temperature (°C)
- Solar generation (kW electric)
- Electricity tariff / pricing ($/kWh)
- Building internal occupancy heat gain (kW thermal)
"""

import copy
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np


@dataclass
class ScenarioPreset:
    """Complete specification of a simulation scenario."""
    id: str
    name: str
    description: str
    # Building
    building_type: str = "Small Office"
    thermal_mass: float = 20.0  # kWh/°C
    heat_loss_coeff: float = 1.2  # kW/°C
    initial_temp: float = 24.0  # °C
    target_temp: float = 22.0  # °C
    comfort_tolerance: float = 1.5  # °C
    ac_cop: float = 3.2
    max_ac_cooling_power: float = 25.0  # kW
    # Weather
    base_temp: float = 28.0  # Mean temperature (°C)
    temp_variation: float = 10.0  # Daily swing (°C)
    peak_temp_hour: float = 15.0  # Hour when temperature peaks
    min_temp_hour: float = 6.0   # Hour when temperature is lowest
    custom_temp_offset: float = 0.0  # Offset to shift entire curve (°C)
    # Solar
    solar_peak_kw: float = 6.0  # Max solar generation at midday (kW)
    cloud_cover_factor: float = 1.0  # 1.0 = full sun, 0.2 = cloudy
    solar_start_hour: float = 6.5
    solar_end_hour: float = 18.5
    # Electricity pricing
    off_peak_price: float = 0.12  # $/kWh
    mid_peak_price: float = 0.20  # $/kWh
    peak_price: float = 0.45      # $/kWh
    peak_start_hour: float = 14.0 # Hour peak pricing begins
    peak_end_hour: float = 20.0   # Hour peak pricing ends
    # Thermal Storage
    storage_capacity_kwh: float = 50.0  # kWh thermal
    initial_soc_pct: float = 20.0       # %
    max_charge_rate_kw: float = 12.0    # kW thermal
    max_discharge_rate_kw: float = 15.0 # kW thermal
    charge_efficiency: float = 0.90
    discharge_efficiency: float = 0.95
    loss_rate_per_hour: float = 0.01
    chiller_cop_charge: float = 3.0


def generate_diurnal_temperature_profile(
    hours: np.ndarray,
    base_temp: float,
    temp_variation: float,
    peak_hour: float = 15.0,
    offset: float = 0.0,
) -> np.ndarray:
    """
    Generate realistic diurnal outdoor temperature profile using asymmetric sine model.
    Minimum typically around 06:00, maximum around 15:00.
    """
    # Normalized phase angle so peak is at peak_hour (typically 15:00)
    phase = (hours - peak_hour) * (2.0 * np.pi / 24.0)
    # Cosine has peak at 0, so cos(phase) peaks at peak_hour
    t_profile = base_temp + (temp_variation / 2.0) * np.cos(phase) + offset
    return np.round(t_profile, 2)


def generate_solar_generation_profile(
    hours: np.ndarray,
    peak_kw: float,
    cloud_factor: float = 1.0,
    start_hour: float = 6.5,
    end_hour: float = 18.5,
) -> np.ndarray:
    """
    Generate solar PV generation profile (kW electric).
    Zero at night, bell curve between sunrise and sunset scaled by cloudiness.
    """
    solar = np.zeros_like(hours, dtype=float)
    daylight_mask = (hours >= start_hour) & (hours <= end_hour)
    
    if np.any(daylight_mask) and peak_kw > 0:
        day_length = end_hour - start_hour
        normalized_time = (hours[daylight_mask] - start_hour) / day_length
        # Sine shaped profile for solar irradiance
        irradiance = np.sin(np.pi * normalized_time) ** 1.3
        effective_solar = peak_kw * cloud_factor * irradiance
        solar[daylight_mask] = np.maximum(0.0, effective_solar)

    return np.round(solar, 2)


def generate_electricity_price_profile(
    hours: np.ndarray,
    off_peak_price: float,
    mid_peak_price: float,
    peak_price: float,
    peak_start: float = 14.0,
    peak_end: float = 20.0,
) -> np.ndarray:
    """
    Generate Time-Of-Use (TOU) electricity price schedule ($/kWh).
    - Off-peak: night (22:00 - 07:00)
    - Peak: designated afternoon/evening hours (e.g. 14:00 - 20:00)
    - Mid-peak: morning & shoulder hours (07:00 - 14:00 and 20:00 - 22:00)
    """
    prices = np.full_like(hours, fill_value=off_peak_price, dtype=float)
    
    for i, h in enumerate(hours):
        h_mod = h % 24.0
        # Check peak window
        if peak_start <= peak_end:
            is_peak = peak_start <= h_mod < peak_end
        else:
            is_peak = (h_mod >= peak_start) or (h_mod < peak_end)
            
        if is_peak:
            prices[i] = peak_price
        elif 7.0 <= h_mod < 22.0:
            prices[i] = mid_peak_price
        else:
            prices[i] = off_peak_price

    return np.round(prices, 3)


# Standard Presets
SCENARIOS: Dict[str, ScenarioPreset] = {
    "hot_sunny": ScenarioPreset(
        id="hot_sunny",
        name="Hot Sunny Day",
        description="High afternoon temperatures (up to 37°C) with strong midday solar availability (6.5 kW). Ideal for pre-cooling & solar-driven charging.",
        base_temp=30.0,
        temp_variation=12.0,
        solar_peak_kw=6.5,
        cloud_cover_factor=1.0,
        off_peak_price=0.12,
        mid_peak_price=0.20,
        peak_price=0.48,
        peak_start_hour=14.0,
        peak_end_hour=20.0,
        storage_capacity_kwh=50.0,
        initial_soc_pct=25.0,
    ),
    "hot_cloudy": ScenarioPreset(
        id="hot_cloudy",
        name="Hot Cloudy Day",
        description="High temperatures (up to 34°C) but heavily overcast with minimal solar generation. Tests grid-only off-peak charging economics.",
        base_temp=29.0,
        temp_variation=8.0,
        solar_peak_kw=6.0,
        cloud_cover_factor=0.20,  # 80% solar reduction
        off_peak_price=0.12,
        mid_peak_price=0.20,
        peak_price=0.45,
        peak_start_hour=14.0,
        peak_end_hour=20.0,
        storage_capacity_kwh=50.0,
        initial_soc_pct=20.0,
    ),
    "high_price_peak": ScenarioPreset(
        id="high_price_peak",
        name="High Electricity Price Peak",
        description="Extreme TOU pricing delta ($0.65/kWh peak vs $0.10/kWh off-peak). Demonstrates massive peak-shaving cost reduction.",
        base_temp=28.0,
        temp_variation=10.0,
        solar_peak_kw=5.0,
        cloud_cover_factor=0.85,
        off_peak_price=0.10,
        mid_peak_price=0.18,
        peak_price=0.65,  # Extreme peak
        peak_start_hour=13.0,
        peak_end_hour=19.0,
        storage_capacity_kwh=60.0,
        initial_soc_pct=30.0,
    ),
    "low_storage": ScenarioPreset(
        id="low_storage",
        name="Low Storage Capacity",
        description="Undersized storage capacity (12 kWh thermal). Highlights storage depletion during peak hours and backup AC activation.",
        base_temp=30.0,
        temp_variation=10.0,
        solar_peak_kw=5.5,
        cloud_cover_factor=0.90,
        off_peak_price=0.12,
        mid_peak_price=0.20,
        peak_price=0.45,
        peak_start_hour=14.0,
        peak_end_hour=20.0,
        storage_capacity_kwh=12.0,  # Small storage
        initial_soc_pct=20.0,
        max_charge_rate_kw=4.0,
        max_discharge_rate_kw=6.0,
    ),
    "custom": ScenarioPreset(
        id="custom",
        name="Custom Settings",
        description="Fully user-configurable parameters for building, weather, tariffs, and thermal storage.",
        base_temp=28.0,
        temp_variation=10.0,
        solar_peak_kw=5.0,
        cloud_cover_factor=1.0,
        off_peak_price=0.12,
        mid_peak_price=0.20,
        peak_price=0.45,
        peak_start_hour=14.0,
        peak_end_hour=20.0,
        storage_capacity_kwh=50.0,
        initial_soc_pct=20.0,
    ),
}


def get_scenario_preset(scenario_id: str) -> ScenarioPreset:
    """Retrieve an independent copy of scenario preset by ID or default to hot_sunny."""
    preset = SCENARIOS.get(scenario_id, SCENARIOS["hot_sunny"])
    return copy.deepcopy(preset)


def list_scenarios() -> List[Dict[str, str]]:
    """Return list of available scenario presets for UI dropdowns."""
    return [{"id": k, "name": v.name, "description": v.description} for k, v in SCENARIOS.items()]
