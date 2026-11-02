"""
Simulation Controller and Engine for Conventional AC vs HeatBank.

Coordinates time-stepping, operating decisions, energy accounting, and state tracking.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd

from .building import BuildingConfig, BuildingModel
from .storage import ThermalStorageConfig, ThermalStorage
from .scenarios import ScenarioPreset, generate_diurnal_temperature_profile, generate_solar_generation_profile, generate_electricity_price_profile


@dataclass
class SimulationStepResult:
    """Detailed telemetry record for a single simulation time step."""
    step_index: int
    hour: float  # e.g., 14.25 = 14:15
    time_label: str  # e.g., "14:15"
    outdoor_temp: float  # °C
    solar_generation_kw: float  # kW electric
    electricity_price: float  # $/kWh
    is_peak_period: bool
    internal_gain_kw: float  # kW thermal

    # Conventional AC
    conv_indoor_temp: float  # °C
    conv_cooling_demand_kw: float  # kW thermal
    conv_cooling_delivered_kw: float  # kW thermal
    conv_ac_electric_power_kw: float  # kW electric
    conv_grid_power_kw: float  # kW electric
    conv_grid_energy_kwh: float  # kWh electric
    conv_cost: float  # $
    conv_in_comfort: bool

    # HeatBank
    hb_indoor_temp: float  # °C
    hb_cooling_demand_kw: float  # kW thermal
    hb_cooling_delivered_kw: float  # kW thermal
    hb_storage_discharge_cooling_kw: float  # kW thermal
    hb_storage_drawn_kwh: float  # kWh thermal
    hb_backup_ac_cooling_kw: float  # kW thermal
    hb_backup_ac_electric_kw: float  # kW electric
    hb_charge_rate_kw_th: float  # kW thermal into storage
    hb_charge_electric_kw: float  # kW electric
    hb_storage_kwh: float  # kWh thermal
    hb_storage_soc_pct: float  # %
    hb_storage_loss_kwh: float  # kWh thermal
    hb_total_electric_power_kw: float  # kW electric
    hb_grid_power_kw: float  # kW electric
    hb_grid_energy_kwh: float  # kWh electric
    hb_cost: float  # $
    hb_in_comfort: bool
    hb_decision_label: str  # Explanation of controller action


@dataclass
class SimulationResult:
    """Aggregated simulation output containing step-by-step telemetry and DataFrame."""
    steps: List[SimulationStepResult] = field(default_factory=list)
    df: pd.DataFrame = field(default_factory=pd.DataFrame)
    duration_hours: float = 24.0
    dt_hours: float = 0.25
    scenario_name: str = ""


class SimulationEngine:
    """
    Executes a 24-hour comparative simulation between Conventional AC and HeatBank.
    """

    def __init__(
        self,
        building_config: BuildingConfig,
        storage_config: ThermalStorageConfig,
        scenario: ScenarioPreset,
        start_hour: float = 0.0,
        dt_hours: float = 0.25,
        duration_hours: float = 24.0,
    ):
        self.building_config = building_config
        self.storage_config = storage_config
        self.scenario = scenario
        self.start_hour = float(start_hour)
        self.dt_hours = float(dt_hours)
        self.duration_hours = float(duration_hours)

    def run(self) -> SimulationResult:
        """Run the full 24-hour simulation and return structured results."""
        # Validate configs
        self.building_config.validate()
        self.storage_config.validate()

        num_steps = int(round(self.duration_hours / self.dt_hours))
        hours = np.array([self.start_hour + i * self.dt_hours for i in range(num_steps)])

        # Generate weather & environmental profiles
        outdoor_temps = generate_diurnal_temperature_profile(
            hours=hours,
            base_temp=self.scenario.base_temp,
            temp_variation=self.scenario.temp_variation,
            peak_hour=self.scenario.peak_temp_hour,
            offset=self.scenario.custom_temp_offset,
        )

        solar_profile = generate_solar_generation_profile(
            hours=hours,
            peak_kw=self.scenario.solar_peak_kw,
            cloud_factor=self.scenario.cloud_cover_factor,
            start_hour=self.scenario.solar_start_hour,
            end_hour=self.scenario.solar_end_hour,
        )

        price_profile = generate_electricity_price_profile(
            hours=hours,
            off_peak_price=self.scenario.off_peak_price,
            mid_peak_price=self.scenario.mid_peak_price,
            peak_price=self.scenario.peak_price,
            peak_start=self.scenario.peak_start_hour,
            peak_end=self.scenario.peak_end_hour,
        )

        # Initialize models
        conv_building = BuildingModel(self.building_config)
        hb_building = BuildingModel(self.building_config)
        storage = ThermalStorage(self.storage_config)

        step_records: List[SimulationStepResult] = []

        for i in range(num_steps):
            h_raw = hours[i]
            h_mod = h_raw % 24.0
            h_int = int(h_mod)
            m_int = int(round((h_mod - h_int) * 60))
            time_label = f"{h_int:02d}:{m_int:02d}"

            t_out = float(outdoor_temps[i])
            p_solar = float(solar_profile[i])
            c_price = float(price_profile[i])

            # Determine peak period
            p_start = self.scenario.peak_start_hour
            p_end = self.scenario.peak_end_hour
            if p_start <= p_end:
                is_peak = p_start <= h_mod < p_end
            else:
                is_peak = (h_mod >= p_start) or (h_mod < p_end)

            # Internal gains (occupancy, plug loads)
            q_int = conv_building.get_internal_gain(h_mod)

            # -----------------------------------------------------------------
            # 1. Mode 1: Conventional AC Simulation Step
            # -----------------------------------------------------------------
            conv_q_req = conv_building.calculate_cooling_demand(
                outdoor_temp=t_out,
                internal_gain_kw=q_int,
                solar_irradiance_kw=p_solar,
                dt_hours=self.dt_hours,
            )
            # Conventional AC delivers cooling to match demand (within max AC capacity)
            conv_q_delivered = conv_q_req
            conv_t_next = conv_building.step(
                cooling_delivered_kw=conv_q_delivered,
                outdoor_temp=t_out,
                internal_gain_kw=q_int,
                solar_irradiance_kw=p_solar,
                dt_hours=self.dt_hours,
            )
            conv_in_comfort = conv_building.is_in_comfort_range()

            # Electricity consumed by conventional AC:
            conv_p_ac_elec = conv_q_delivered / self.building_config.ac_cop
            # Net grid power after consuming available on-site solar:
            conv_p_grid = max(0.0, conv_p_ac_elec - p_solar)
            conv_e_grid = conv_p_grid * self.dt_hours
            conv_cost = conv_e_grid * c_price

            # -----------------------------------------------------------------
            # 2. Mode 2: HeatBank Simulation Step
            # -----------------------------------------------------------------
            # Apply standing thermal storage losses
            hb_loss_kwh = storage.apply_standing_loss(self.dt_hours)

            hb_q_req = hb_building.calculate_cooling_demand(
                outdoor_temp=t_out,
                internal_gain_kw=q_int,
                solar_irradiance_kw=p_solar,
                dt_hours=self.dt_hours,
            )

            hb_q_dis_delivered = 0.0
            hb_storage_drawn_kwh = 0.0
            hb_q_backup_ac = 0.0
            hb_p_backup_ac_elec = 0.0
            hb_p_charge_th = 0.0
            hb_e_charge_elec = 0.0
            decision_label = ""

            if is_peak:
                # PEAK PERIOD: Maximize discharge from thermal storage to eliminate AC draw
                if hb_q_req > 0.0 and storage.current_kwh > 1e-6:
                    hb_q_dis_delivered, hb_storage_drawn_kwh = storage.discharge(
                        requested_cooling_kw_th=hb_q_req,
                        dt_hours=self.dt_hours,
                    )
                    hb_q_rem = max(0.0, hb_q_req - hb_q_dis_delivered)
                    if hb_q_rem > 0.0:
                        # Storage exhausted or rate-limited; backup AC provides remainder
                        hb_q_backup_ac = hb_q_rem
                        hb_p_backup_ac_elec = hb_q_backup_ac / self.building_config.ac_cop
                        decision_label = "Peak Discharge + Backup AC"
                    else:
                        decision_label = "Peak Discharge (100% Thermal Storage)"
                elif hb_q_req > 0.0:
                    # Storage completely depleted
                    hb_q_backup_ac = hb_q_req
                    hb_p_backup_ac_elec = hb_q_backup_ac / self.building_config.ac_cop
                    decision_label = "Peak Backup AC (Storage Empty)"
                else:
                    decision_label = "Peak Idle (No Cooling Demand)"

                # Never charge storage during peak price period
                hb_p_charge_th = 0.0
                hb_e_charge_elec = 0.0

            else:
                # OFF-PEAK / SHOULDER PERIOD
                # Building cooling demand is served directly by AC
                hb_q_backup_ac = hb_q_req
                hb_p_backup_ac_elec = hb_q_backup_ac / self.building_config.ac_cop

                # Determine storage charging strategy:
                # 1) Solar surplus charging: if solar exceeds AC load, prioritize storing free solar energy
                surplus_solar_kw = max(0.0, p_solar - hb_p_backup_ac_elec)

                # Is this an off-peak price window? (e.g. night/early morning or price <= off_peak_price)
                is_off_peak_price = (c_price <= self.scenario.off_peak_price + 1e-4) or (h_mod < 8.0) or (h_mod >= 22.0)

                if surplus_solar_kw > 0.1:
                    # Drive chiller charging using free surplus solar electricity
                    # P_th = surplus_solar_kw * COP_charge
                    solar_ch_power_th = surplus_solar_kw * self.storage_config.chiller_cop_charge
                    # Can also charge up to max_charge_rate_kw
                    desired_charge_kw = min(solar_ch_power_th, self.storage_config.max_charge_rate_kw)
                    hb_p_charge_th, hb_e_charge_elec, _ = storage.charge(desired_charge_kw, self.dt_hours)
                    decision_label = "Solar Surplus Storage Charging"

                elif is_off_peak_price and storage.soc_pct < 99.5:
                    # Charge using cheap off-peak grid electricity before peak arrives
                    desired_charge_kw = self.storage_config.max_charge_rate_kw
                    hb_p_charge_th, hb_e_charge_elec, _ = storage.charge(desired_charge_kw, self.dt_hours)
                    decision_label = "Off-Peak Grid Storage Charging"

                else:
                    # Shoulder period / mid-peak: float storage (no charging from expensive grid)
                    hb_p_charge_th = 0.0
                    hb_e_charge_elec = 0.0
                    if hb_q_req > 0.0:
                        decision_label = "Direct AC (Shoulder Hours)"
                    else:
                        decision_label = "Idle (Shoulder Hours)"

            # Advance HeatBank building thermal state
            hb_q_total_cooling_delivered = hb_q_dis_delivered + hb_q_backup_ac
            hb_t_next = hb_building.step(
                cooling_delivered_kw=hb_q_total_cooling_delivered,
                outdoor_temp=t_out,
                internal_gain_kw=q_int,
                solar_irradiance_kw=p_solar,
                dt_hours=self.dt_hours,
            )
            hb_in_comfort = hb_building.is_in_comfort_range()

            # HeatBank electricity accounting
            hb_p_charge_elec = hb_e_charge_elec / self.dt_hours
            hb_total_electric_power_kw = hb_p_backup_ac_elec + hb_p_charge_elec
            hb_p_grid = max(0.0, hb_total_electric_power_kw - p_solar)
            hb_e_grid = hb_p_grid * self.dt_hours
            hb_cost = hb_e_grid * c_price

            # Record step telemetry
            step_record = SimulationStepResult(
                step_index=i,
                hour=h_raw,
                time_label=time_label,
                outdoor_temp=t_out,
                solar_generation_kw=p_solar,
                electricity_price=c_price,
                is_peak_period=is_peak,
                internal_gain_kw=q_int,
                # Conv
                conv_indoor_temp=conv_t_next,
                conv_cooling_demand_kw=conv_q_req,
                conv_cooling_delivered_kw=conv_q_delivered,
                conv_ac_electric_power_kw=conv_p_ac_elec,
                conv_grid_power_kw=conv_p_grid,
                conv_grid_energy_kwh=conv_e_grid,
                conv_cost=conv_cost,
                conv_in_comfort=conv_in_comfort,
                # HeatBank
                hb_indoor_temp=hb_t_next,
                hb_cooling_demand_kw=hb_q_req,
                hb_cooling_delivered_kw=hb_q_total_cooling_delivered,
                hb_storage_discharge_cooling_kw=hb_q_dis_delivered,
                hb_storage_drawn_kwh=hb_storage_drawn_kwh,
                hb_backup_ac_cooling_kw=hb_q_backup_ac,
                hb_backup_ac_electric_kw=hb_p_backup_ac_elec,
                hb_charge_rate_kw_th=hb_p_charge_th,
                hb_charge_electric_kw=hb_p_charge_elec,
                hb_storage_kwh=storage.available_energy_kwh,
                hb_storage_soc_pct=storage.soc_pct,
                hb_storage_loss_kwh=hb_loss_kwh,
                hb_total_electric_power_kw=hb_total_electric_power_kw,
                hb_grid_power_kw=hb_p_grid,
                hb_grid_energy_kwh=hb_e_grid,
                hb_cost=hb_cost,
                hb_in_comfort=hb_in_comfort,
                hb_decision_label=decision_label,
            )
            step_records.append(step_record)

        # Convert to pandas DataFrame for seamless plotting and metrics
        df = pd.DataFrame([vars(s) for s in step_records])
        # Cumulative cost columns
        df["conv_cumulative_cost"] = df["conv_cost"].cumsum()
        df["hb_cumulative_cost"] = df["hb_cost"].cumsum()
        df["conv_cumulative_energy_kwh"] = df["conv_grid_energy_kwh"].cumsum()
        df["hb_cumulative_energy_kwh"] = df["hb_grid_energy_kwh"].cumsum()

        return SimulationResult(
            steps=step_records,
            df=df,
            duration_hours=self.duration_hours,
            dt_hours=self.dt_hours,
            scenario_name=self.scenario.name,
        )
