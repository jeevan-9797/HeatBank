"""
Building thermal dynamics model.

Implements a physically interpretable lumped-capacitance thermal balance model.
Units:
- Temperature: °C
- Thermal energy: kWh (thermal)
- Power: kW (thermal)
- Time step: hours
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any
import numpy as np


@dataclass
class BuildingConfig:
    """Configuration parameters for building thermal model."""
    building_type: str = "Small Office"  # "Small Office", "Classroom", "Custom"
    thermal_mass: float = 20.0  # Heat capacity C_bldg in kWh/°C
    heat_loss_coeff: float = 1.2  # Envelope UA in kW/°C
    initial_temp: float = 24.0  # Initial indoor temperature in °C
    target_temp: float = 22.0  # Target indoor comfort temperature in °C
    comfort_tolerance: float = 1.5  # Allowed deviation for comfort band (e.g., target ± 1.5°C)
    ac_cop: float = 3.2  # Coefficient of performance for conventional/backup AC
    max_ac_cooling_power: float = 25.0  # Max AC cooling power in kW thermal
    solar_heat_gain_coeff: float = 0.5  # Window/solar transmission factor into space (kW / kW_solar_ref)

    def validate(self) -> None:
        """Validate configuration values to prevent impossible or negative values."""
        if self.thermal_mass <= 0:
            raise ValueError(f"Thermal mass must be positive, got {self.thermal_mass} kWh/°C")
        if self.heat_loss_coeff < 0:
            raise ValueError(f"Heat loss coefficient cannot be negative, got {self.heat_loss_coeff} kW/°C")
        if not (-20.0 <= self.initial_temp <= 60.0):
            raise ValueError(f"Initial indoor temperature {self.initial_temp}°C is out of realistic range")
        if not (10.0 <= self.target_temp <= 35.0):
            raise ValueError(f"Target temperature {self.target_temp}°C is out of realistic range")
        if self.comfort_tolerance <= 0:
            raise ValueError(f"Comfort tolerance must be positive, got {self.comfort_tolerance}°C")
        if self.ac_cop <= 0:
            raise ValueError(f"AC COP must be positive, got {self.ac_cop}")
        if self.max_ac_cooling_power < 0:
            raise ValueError(f"Max AC cooling power cannot be negative, got {self.max_ac_cooling_power} kW")


class BuildingModel:
    """
    Simulates building thermal response over discrete time intervals.
    
    Thermal balance:
    C_bldg * dT_in/dt = UA * (T_out - T_in) + Q_internal + Q_solar_gain - Q_cooling
    """

    def __init__(self, config: BuildingConfig):
        config.validate()
        self.config = config
        self.current_temp = float(config.initial_temp)

    def reset(self, initial_temp: Optional[float] = None) -> None:
        """Reset building state to initial condition."""
        if initial_temp is not None:
            self.current_temp = float(initial_temp)
        else:
            self.current_temp = float(self.config.initial_temp)

    def get_internal_gain(self, hour_of_day: float) -> float:
        """
        Compute internal heat gain from occupants, lighting, and plug loads (kW).
        Profile depends on building_type.
        """
        hour = hour_of_day % 24.0

        if self.config.building_type == "Classroom":
            # High occupancy during school hours (08:00 to 15:30)
            if 8.0 <= hour <= 15.5:
                # 30 students + instructor + computers/lighting ~ 4.5 - 5.5 kW
                return 5.0
            elif 7.0 <= hour < 8.0 or 15.5 < hour <= 17.0:
                # Prep / after-school cleaning / staff
                return 1.5
            else:
                # Idle baseline (emergency lights, standby equipment)
                return 0.3

        elif self.config.building_type == "Small Office":
            # Standard office hours (08:00 to 18:00) with slight midday peak
            if 8.0 <= hour < 12.0:
                return 3.5
            elif 12.0 <= hour < 14.0:
                return 4.2  # Cafeteria/lunch appliances, full occupancy
            elif 14.0 <= hour <= 18.0:
                return 3.8
            elif 6.5 <= hour < 8.0 or 18.0 < hour <= 20.0:
                return 1.2
            else:
                return 0.5  # Servers, networking, standby equipment

        else:
            # Custom profile: baseline 0.6 kW, standard daytime 3.0 kW
            if 8.0 <= hour <= 18.0:
                return 3.5
            return 0.6

    def calculate_free_floating_temp(
        self,
        outdoor_temp: float,
        internal_gain_kw: float,
        solar_irradiance_kw: float,
        dt_hours: float,
    ) -> float:
        """
        Calculate the predicted temperature if NO cooling is provided.
        """
        envelope_gain_kw = self.config.heat_loss_coeff * (outdoor_temp - self.current_temp)
        solar_gain_kw = self.config.solar_heat_gain_coeff * max(0.0, solar_irradiance_kw)
        total_heat_gain_kw = envelope_gain_kw + internal_gain_kw + solar_gain_kw

        # Temperature increment over dt
        delta_temp = (total_heat_gain_kw * dt_hours) / self.config.thermal_mass
        return self.current_temp + delta_temp

    def calculate_cooling_demand(
        self,
        outdoor_temp: float,
        internal_gain_kw: float,
        solar_irradiance_kw: float,
        dt_hours: float,
    ) -> float:
        """
        Calculate thermal cooling power (kW thermal) required to achieve target comfort temperature.
        Returns 0.0 if indoor temperature is already below or at target.
        """
        t_predicted = self.calculate_free_floating_temp(
            outdoor_temp=outdoor_temp,
            internal_gain_kw=internal_gain_kw,
            solar_irradiance_kw=solar_irradiance_kw,
            dt_hours=dt_hours,
        )

        if t_predicted <= self.config.target_temp:
            return 0.0

        # Required cooling energy in kWh_thermal
        needed_energy_kwh = self.config.thermal_mass * (t_predicted - self.config.target_temp)
        # Power required in kW_thermal
        needed_power_kw = needed_energy_kwh / dt_hours

        # Respect maximum AC capacity if configured
        if self.config.max_ac_cooling_power > 0:
            return float(min(needed_power_kw, self.config.max_ac_cooling_power))
        return float(needed_power_kw)

    def step(
        self,
        cooling_delivered_kw: float,
        outdoor_temp: float,
        internal_gain_kw: float,
        solar_irradiance_kw: float,
        dt_hours: float,
    ) -> float:
        """
        Advance building thermal state by one time step with actual cooling delivered.
        Updates self.current_temp and returns the new temperature.
        """
        t_float = self.calculate_free_floating_temp(
            outdoor_temp=outdoor_temp,
            internal_gain_kw=internal_gain_kw,
            solar_irradiance_kw=solar_irradiance_kw,
            dt_hours=dt_hours,
        )

        cooling_temp_reduction = (cooling_delivered_kw * dt_hours) / self.config.thermal_mass
        self.current_temp = t_float - cooling_temp_reduction

        # Safety numerical bounds
        self.current_temp = float(np.clip(self.current_temp, -10.0, 60.0))
        return self.current_temp

    def is_in_comfort_range(self) -> bool:
        """Check if current temperature is within comfortable bounds."""
        lower_bound = self.config.target_temp - self.config.comfort_tolerance
        upper_bound = self.config.target_temp + self.config.comfort_tolerance
        return lower_bound <= self.current_temp <= upper_bound
