"""
Thermal Energy Storage (HeatBank) model.

Models a cold thermal energy storage (CTES) system, such as chilled water,
ice storage, or phase change material (PCM).

Units:
- Capacity & stored energy: kWh (thermal)
- Power: kW (thermal)
- Electricity: kWh (electric)
- Time step: hours
"""

from dataclasses import dataclass
from typing import Tuple, Dict, Any
import numpy as np


@dataclass
class ThermalStorageConfig:
    """Configuration parameters for thermal energy storage system."""
    capacity_kwh: float = 50.0  # Storage capacity in kWh thermal
    initial_soc_pct: float = 20.0  # Initial state of charge percentage (0 - 100%)
    max_charge_rate_kw: float = 12.0  # Max thermal charging rate in kW thermal
    max_discharge_rate_kw: float = 15.0  # Max thermal discharge rate in kW thermal
    charge_efficiency: float = 0.90  # Efficiency of charging process (0.01 - 1.0)
    discharge_efficiency: float = 0.95  # Efficiency of discharging process (0.01 - 1.0)
    loss_rate_per_hour: float = 0.01  # Standing loss rate per hour (e.g. 0.01 = 1%/hour)
    chiller_cop_charge: float = 3.0  # Chiller COP when operating to charge cold storage

    def validate(self) -> None:
        """Validate storage configuration to prevent impossible physical parameters."""
        if self.capacity_kwh < 0:
            raise ValueError(f"Storage capacity cannot be negative, got {self.capacity_kwh} kWh")
        if not (0.0 <= self.initial_soc_pct <= 100.0):
            raise ValueError(f"Initial SOC percentage must be between 0 and 100%, got {self.initial_soc_pct}%")
        if self.max_charge_rate_kw < 0:
            raise ValueError(f"Max charge rate cannot be negative, got {self.max_charge_rate_kw} kW")
        if self.max_discharge_rate_kw < 0:
            raise ValueError(f"Max discharge rate cannot be negative, got {self.max_discharge_rate_kw} kW")
        if not (0.0 < self.charge_efficiency <= 1.0):
            raise ValueError(f"Charge efficiency must be in (0, 1], got {self.charge_efficiency}")
        if not (0.0 < self.discharge_efficiency <= 1.0):
            raise ValueError(f"Discharge efficiency must be in (0, 1], got {self.discharge_efficiency}")
        if not (0.0 <= self.loss_rate_per_hour < 1.0):
            raise ValueError(f"Loss rate per hour must be in [0, 1), got {self.loss_rate_per_hour}")
        if self.chiller_cop_charge <= 0:
            raise ValueError(f"Chiller charging COP must be positive, got {self.chiller_cop_charge}")


class ThermalStorage:
    """
    Simulates thermal energy storage operations: standby losses, charging, discharging.
    """

    def __init__(self, config: ThermalStorageConfig):
        config.validate()
        self.config = config
        self.capacity_kwh = float(config.capacity_kwh)
        self.current_kwh = (config.initial_soc_pct / 100.0) * self.capacity_kwh
        self._ensure_bounds()

    def reset(self, initial_soc_pct: float = None) -> None:
        """Reset storage to configured or specified initial state."""
        pct = self.config.initial_soc_pct if initial_soc_pct is None else initial_soc_pct
        pct = float(np.clip(pct, 0.0, 100.0))
        self.current_kwh = (pct / 100.0) * self.capacity_kwh
        self._ensure_bounds()

    def _ensure_bounds(self) -> None:
        """Clamp current storage within [0, capacity]."""
        if self.capacity_kwh <= 0.0:
            self.current_kwh = 0.0
        else:
            self.current_kwh = float(np.clip(self.current_kwh, 0.0, self.capacity_kwh))

    @property
    def soc_pct(self) -> float:
        """Current state of charge in percent (0 - 100%)."""
        if self.capacity_kwh <= 0:
            return 0.0
        return float((self.current_kwh / self.capacity_kwh) * 100.0)

    @property
    def available_energy_kwh(self) -> float:
        """Thermal energy currently stored (kWh thermal)."""
        return float(self.current_kwh)

    def apply_standing_loss(self, dt_hours: float) -> float:
        """
        Apply standing thermal losses over time step dt.
        Returns the thermal kWh lost to ambient.
        """
        if self.capacity_kwh <= 0.0 or self.current_kwh <= 0.0:
            return 0.0

        loss_fraction = min(1.0, max(0.0, self.config.loss_rate_per_hour * dt_hours))
        lost_kwh = self.current_kwh * loss_fraction
        self.current_kwh = max(0.0, self.current_kwh - lost_kwh)
        return float(lost_kwh)

    def charge(self, requested_power_kw_th: float, dt_hours: float) -> Tuple[float, float, float]:
        """
        Charge thermal storage with requested thermal cooling power.
        
        Args:
            requested_power_kw_th: Desired charging rate in kW thermal.
            dt_hours: Time step duration in hours.
            
        Returns:
            Tuple of:
            - actual_charge_power_kw_th: Actual thermal cooling rate into storage (kW_th)
            - electricity_used_kwh: Electricity consumed to run chiller for charging (kWh_e)
            - stored_delta_kwh: Net thermal energy added to storage medium (kWh_th)
        """
        if self.capacity_kwh <= 0.0 or requested_power_kw_th <= 0.0 or dt_hours <= 0.0:
            return 0.0, 0.0, 0.0

        # Max allowed rate
        allowed_power_kw = min(requested_power_kw_th, self.config.max_charge_rate_kw)

        # Room remaining in storage
        room_kwh = max(0.0, self.capacity_kwh - self.current_kwh)
        if room_kwh <= 1e-9:
            return 0.0, 0.0, 0.0

        # Maximum thermal power that fits before reaching capacity
        # stored_delta = power * dt * charge_efficiency <= room_kwh
        max_power_by_room = room_kwh / (dt_hours * self.config.charge_efficiency)

        actual_power_kw = min(allowed_power_kw, max_power_by_room)
        stored_delta_kwh = actual_power_kw * dt_hours * self.config.charge_efficiency

        # Electricity required by the chiller to generate actual_power_kw:
        # P_elec = actual_power_kw / chiller_cop_charge
        electricity_used_kwh = (actual_power_kw * dt_hours) / self.config.chiller_cop_charge

        self.current_kwh += stored_delta_kwh
        self._ensure_bounds()

        return float(actual_power_kw), float(electricity_used_kwh), float(stored_delta_kwh)

    def discharge(self, requested_cooling_kw_th: float, dt_hours: float) -> Tuple[float, float]:
        """
        Discharge thermal storage to meet building cooling demand.
        
        Args:
            requested_cooling_kw_th: Thermal cooling power demanded by the building (kW_th).
            dt_hours: Time step duration in hours.
            
        Returns:
            Tuple of:
            - cooling_delivered_kw_th: Cooling thermal power delivered to building (kW_th)
            - storage_drawn_kwh: Thermal energy extracted from storage medium (kWh_th)
        """
        if self.capacity_kwh <= 0.0 or requested_cooling_kw_th <= 0.0 or dt_hours <= 0.0:
            return 0.0, 0.0

        if self.current_kwh <= 1e-9:
            return 0.0, 0.0

        # Power limit
        allowed_power_kw = min(requested_cooling_kw_th, self.config.max_discharge_rate_kw)

        # Storage availability limit:
        # cooling_delivered = storage_drawn * discharge_efficiency
        # storage_drawn = cooling_delivered * dt / discharge_efficiency <= current_kwh
        max_power_by_storage = (self.current_kwh * self.config.discharge_efficiency) / dt_hours

        cooling_delivered_kw = min(allowed_power_kw, max_power_by_storage)
        storage_drawn_kwh = (cooling_delivered_kw * dt_hours) / self.config.discharge_efficiency

        self.current_kwh = max(0.0, self.current_kwh - storage_drawn_kwh)
        self._ensure_bounds()

        return float(cooling_delivered_kw), float(storage_drawn_kwh)
