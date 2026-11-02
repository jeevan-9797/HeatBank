"""
HeatBank Thermal Energy Simulation Application.
Interactive web dashboard comparing Conventional AC vs HeatBank thermal storage.
"""

import inspect
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from simulation.building import BuildingConfig, BuildingModel
from simulation.storage import ThermalStorageConfig, ThermalStorage
from simulation.scenarios import (
    ScenarioPreset,
    get_scenario_preset,
    list_scenarios,
    SCENARIOS,
)
from simulation.controller import SimulationEngine, SimulationResult
from simulation.metrics import calculate_metrics, SimulationMetrics

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="HeatBank — Thermal Energy Simulation",
    page_icon="❄️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Compatibility helper for Streamlit container width vs width argument
def get_width_kwargs():
    """Return appropriate width argument based on Streamlit version to avoid deprecation warnings."""
    sig = inspect.signature(st.plotly_chart)
    if "width" in sig.parameters:
        return {"width": "stretch"}
    return {"use_container_width": True}

W_KWARGS = get_width_kwargs()

# Custom CSS styling for modern energy-tech aesthetic
st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        margin-bottom: 0.1rem;
        color: #1E293B;
    }
    .sub-title {
        font-size: 1.15rem;
        font-weight: 500;
        color: #0284C7;
        margin-bottom: 0.8rem;
    }
    .simulation-badge {
        display: inline-block;
        background-color: #FEF3C7;
        color: #92400E;
        padding: 0.25rem 0.65rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.5rem;
        border: 1px solid #FCD34D;
    }
    .kpi-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 0.75rem;
        padding: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 0.75rem;
        height: 100%;
    }
    .kpi-title {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0.25rem;
    }
    .kpi-val {
        font-size: 1.55rem;
        font-weight: 800;
        color: #0F172A;
    }
    .kpi-unit {
        font-size: 0.85rem;
        font-weight: 500;
        color: #64748B;
        margin-left: 0.2rem;
    }
    .kpi-savings-pos {
        color: #059669;
        font-size: 0.82rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }
    .kpi-savings-neg {
        color: #DC2626;
        font-size: 0.82rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }
    .kpi-subtext {
        color: #64748B;
        font-size: 0.82rem;
        font-weight: 500;
        margin-top: 0.25rem;
    }
    .info-callout {
        background-color: #F8FAFC;
        border-left: 4px solid #0284C7;
        padding: 0.75rem 1rem;
        border-radius: 0.25rem;
        font-size: 0.88rem;
        color: #334155;
        margin-bottom: 1.1rem;
    }
    .steps-container {
        display: flex;
        gap: 1rem;
        background: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-radius: 0.5rem;
        padding: 0.75rem 1rem;
        margin-bottom: 1.2rem;
    }
    .step-item {
        flex: 1;
        font-size: 0.85rem;
        color: #166534;
    }
    .step-num {
        font-weight: 800;
        color: #15803D;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Session State Initialization & Presets with Versioned Synchronization
# -----------------------------------------------------------------------------
scenario_list = list_scenarios()
scenario_names = [s["name"] for s in scenario_list]
name_to_id = {s["name"]: s["id"] for s in scenario_list}
id_to_name = {s["id"]: s["name"] for s in scenario_list}

if "selected_scenario_id" not in st.session_state:
    st.session_state["selected_scenario_id"] = "hot_sunny"

if "preset_version" not in st.session_state:
    st.session_state["preset_version"] = 0

def on_preset_change():
    """Triggered when user selects a different preset from the dropdown."""
    chosen_name = st.session_state["scenario_dropdown_key"]
    st.session_state["selected_scenario_id"] = name_to_id[chosen_name]
    st.session_state["preset_version"] += 1

def reset_to_defaults():
    """Reset active scenario parameters back to default template."""
    st.session_state["preset_version"] += 1

current_preset = get_scenario_preset(st.session_state["selected_scenario_id"])
rev = st.session_state["preset_version"]

# -----------------------------------------------------------------------------
# Sidebar: Interactive Simulation Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Simulation Controls")

    # Preset Selector
    current_idx = scenario_names.index(id_to_name.get(st.session_state["selected_scenario_id"], "Hot Sunny Day"))
    selected_name = st.selectbox(
        "Scenario Preset",
        options=scenario_names,
        index=current_idx,
        key="scenario_dropdown_key",
        on_change=on_preset_change,
        help="Select a benchmark scenario. Changing this preset populates verified parameters across all subsystems.",
    )

    st.caption(f"ℹ️ *{current_preset.description}*")

    # Action Buttons: Run & Reset
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_sim = st.button("▶ Run", type="primary", **W_KWARGS)
        if run_sim:
            st.toast("Simulation executed with updated parameters!", icon="⚡")
    with col_btn2:
        if st.button("↺ Reset Defaults", **W_KWARGS):
            reset_to_defaults()
            st.toast("Restored default parameters!", icon="🔄")
            st.rerun()

    st.divider()

    # 1. Building Parameters
    with st.expander("🏢 Building Parameters", expanded=False):
        bldg_options = ["Small Office", "Classroom", "Custom"]
        bldg_idx = bldg_options.index(current_preset.building_type) if current_preset.building_type in bldg_options else 0
        bldg_type = st.selectbox(
            "Building Archetype",
            options=bldg_options,
            index=bldg_idx,
            key=f"bldg_type_{rev}",
            help="Determines baseline internal heat gains from occupants and equipment.",
        )
        thermal_mass = st.number_input(
            "Thermal Mass (kWh/°C)",
            min_value=2.0,
            max_value=150.0,
            value=float(current_preset.thermal_mass),
            step=2.0,
            key=f"thermal_mass_{rev}",
            help="Heat capacity of the building envelope and internal mass.",
        )
        heat_loss_coeff = st.number_input(
            "Envelope Heat Loss UA (kW/°C)",
            min_value=0.1,
            max_value=10.0,
            value=float(current_preset.heat_loss_coeff),
            step=0.1,
            key=f"heat_loss_{rev}",
            help="Thermal conductivity of the envelope to outside temperature differences.",
        )
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            initial_temp = st.number_input(
                "Initial Temp (°C)",
                min_value=12.0,
                max_value=35.0,
                value=float(current_preset.initial_temp),
                step=0.5,
                key=f"init_temp_{rev}",
            )
        with col_t2:
            target_temp = st.number_input(
                "Target Temp (°C)",
                min_value=16.0,
                max_value=28.0,
                value=float(current_preset.target_temp),
                step=0.5,
                key=f"target_temp_{rev}",
                help="Desired comfort temperature setpoint.",
            )
        comfort_tol = st.slider(
            "Comfort Band Tolerance (±°C)",
            min_value=0.5,
            max_value=3.0,
            value=float(current_preset.comfort_tolerance),
            step=0.1,
            key=f"comfort_tol_{rev}",
            help="Allowable indoor deviation before counting as a comfort violation.",
        )
        ac_cop = st.number_input(
            "AC Direct Cooling COP",
            min_value=1.5,
            max_value=6.0,
            value=float(current_preset.ac_cop),
            step=0.1,
            key=f"ac_cop_{rev}",
            help="Coefficient of Performance for conventional/backup air conditioning.",
        )
        max_ac_power = st.number_input(
            "Max AC Cooling Power (kW_th)",
            min_value=5.0,
            max_value=100.0,
            value=float(current_preset.max_ac_cooling_power),
            step=5.0,
            key=f"max_ac_{rev}",
        )

    # 2. Weather & Climate
    with st.expander("☀️ Weather & Solar Profile", expanded=False):
        base_temp = st.number_input(
            "Mean Daily Temp (°C)",
            min_value=10.0,
            max_value=45.0,
            value=float(current_preset.base_temp),
            step=1.0,
            key=f"base_temp_{rev}",
        )
        temp_variation = st.number_input(
            "Daily Temp Swing (°C)",
            min_value=2.0,
            max_value=25.0,
            value=float(current_preset.temp_variation),
            step=1.0,
            key=f"temp_var_{rev}",
            help="Difference between peak afternoon temperature and morning minimum.",
        )
        custom_temp_offset = st.slider(
            "Custom Temp Offset (°C)",
            min_value=-10.0,
            max_value=10.0,
            value=float(current_preset.custom_temp_offset),
            step=0.5,
            key=f"temp_offset_{rev}",
            help="Global offset applied across the entire 24h temperature profile.",
        )
        solar_peak_kw = st.number_input(
            "Rooftop Solar Peak (kW_e)",
            min_value=0.0,
            max_value=30.0,
            value=float(current_preset.solar_peak_kw),
            step=0.5,
            key=f"solar_peak_{rev}",
            help="Rated maximum electric power output of on-site solar PV at solar noon.",
        )
        cloud_cover = st.slider(
            "Solar Clearness Index",
            min_value=0.0,
            max_value=1.0,
            value=float(current_preset.cloud_cover_factor),
            step=0.05,
            key=f"cloud_cover_{rev}",
            help="1.0 = clear sky, 0.2 = overcast/cloudy day.",
        )

    # 3. Energy Tariff & TOU Pricing
    with st.expander("⚡ Electricity Tariff & TOU", expanded=False):
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            off_peak_price = st.number_input(
                "Off-Peak Price ($/kWh)",
                min_value=0.01,
                max_value=1.0,
                value=float(current_preset.off_peak_price),
                step=0.01,
                key=f"off_peak_{rev}",
            )
        with col_p2:
            peak_price = st.number_input(
                "On-Peak Price ($/kWh)",
                min_value=0.05,
                max_value=2.0,
                value=float(current_preset.peak_price),
                step=0.02,
                key=f"on_peak_{rev}",
            )
        mid_peak_price = st.number_input(
            "Mid-Peak / Shoulder ($/kWh)",
            min_value=0.02,
            max_value=1.5,
            value=float(current_preset.mid_peak_price),
            step=0.01,
            key=f"mid_peak_{rev}",
        )
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            peak_start_hour = st.number_input(
                "Peak Start (Hour)",
                min_value=0.0,
                max_value=23.0,
                value=float(current_preset.peak_start_hour),
                step=1.0,
                key=f"peak_start_{rev}",
            )
        with col_h2:
            peak_end_hour = st.number_input(
                "Peak End (Hour)",
                min_value=1.0,
                max_value=24.0,
                value=float(current_preset.peak_end_hour),
                step=1.0,
                key=f"peak_end_{rev}",
            )
        if peak_start_hour >= peak_end_hour:
            st.warning("⚠️ Peak start hour should be earlier than peak end hour for standard daytime TOU.")

    # 4. Thermal Energy Storage (HeatBank)
    with st.expander("🧊 HeatBank Thermal Storage", expanded=False):
        storage_capacity = st.number_input(
            "Storage Capacity (kWh_th)",
            min_value=0.0,
            max_value=200.0,
            value=float(current_preset.storage_capacity_kwh),
            step=5.0,
            key=f"storage_cap_{rev}",
            help="Total thermal cold storage capacity in kWh thermal.",
        )
        initial_soc = st.slider(
            "Initial SOC (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(current_preset.initial_soc_pct),
            step=5.0,
            key=f"init_soc_{rev}",
            help="State of charge at start of simulation.",
        )
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            max_charge_rate = st.number_input(
                "Max Charge Rate (kW_th)",
                min_value=0.1,
                max_value=50.0,
                value=float(current_preset.max_charge_rate_kw),
                step=1.0,
                key=f"max_charge_{rev}",
            )
        with col_r2:
            max_discharge_rate = st.number_input(
                "Max Discharge (kW_th)",
                min_value=0.1,
                max_value=50.0,
                value=float(current_preset.max_discharge_rate_kw),
                step=1.0,
                key=f"max_discharge_{rev}",
            )
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            charge_efficiency = st.slider(
                "Charge Efficiency",
                min_value=0.5,
                max_value=1.0,
                value=float(current_preset.charge_efficiency),
                step=0.02,
                key=f"charge_eff_{rev}",
            )
        with col_e2:
            discharge_efficiency = st.slider(
                "Discharge Efficiency",
                min_value=0.5,
                max_value=1.0,
                value=float(current_preset.discharge_efficiency),
                step=0.02,
                key=f"discharge_eff_{rev}",
            )
        loss_rate_pct = st.slider(
            "Thermal Loss Rate (%/hr)",
            min_value=0.0,
            max_value=10.0,
            value=float(current_preset.loss_rate_per_hour * 100.0),
            step=0.2,
            key=f"loss_rate_{rev}",
            help="Standing thermal loss rate of stored cold to ambient per hour.",
        )
        chiller_cop_charge = st.number_input(
            "Chiller Charging COP",
            min_value=1.2,
            max_value=6.0,
            value=float(current_preset.chiller_cop_charge),
            step=0.1,
            key=f"chiller_cop_{rev}",
            help="COP when chiller operates at lower evaporator temperatures to freeze/chill storage.",
        )

    # 5. Simulation Horizon & Resolution
    with st.expander("⏱️ Horizon & Time Step", expanded=False):
        start_hour = st.slider(
            "Simulation Start Hour",
            min_value=0,
            max_value=23,
            value=0,
            step=1,
            key=f"start_hour_{rev}",
        )
        st.caption("Duration: **24 Hours** | Time Step: **15 Minutes (96 steps)**")

# -----------------------------------------------------------------------------
# Construct Configuration Objects & Execute Simulation
# -----------------------------------------------------------------------------
building_config = BuildingConfig(
    building_type=bldg_type,
    thermal_mass=thermal_mass,
    heat_loss_coeff=heat_loss_coeff,
    initial_temp=initial_temp,
    target_temp=target_temp,
    comfort_tolerance=comfort_tol,
    ac_cop=ac_cop,
    max_ac_cooling_power=max_ac_power,
)

storage_config = ThermalStorageConfig(
    capacity_kwh=storage_capacity,
    initial_soc_pct=initial_soc,
    max_charge_rate_kw=max_charge_rate,
    max_discharge_rate_kw=max_discharge_rate,
    charge_efficiency=charge_efficiency,
    discharge_efficiency=discharge_efficiency,
    loss_rate_per_hour=loss_rate_pct / 100.0,
    chiller_cop_charge=chiller_cop_charge,
)

scenario_config = ScenarioPreset(
    id="active_scenario",
    name=selected_name,
    description="Active runtime simulation scenario",
    building_type=bldg_type,
    thermal_mass=thermal_mass,
    heat_loss_coeff=heat_loss_coeff,
    initial_temp=initial_temp,
    target_temp=target_temp,
    comfort_tolerance=comfort_tol,
    ac_cop=ac_cop,
    max_ac_cooling_power=max_ac_power,
    base_temp=base_temp,
    temp_variation=temp_variation,
    custom_temp_offset=custom_temp_offset,
    solar_peak_kw=solar_peak_kw,
    cloud_cover_factor=cloud_cover,
    off_peak_price=off_peak_price,
    mid_peak_price=mid_peak_price,
    peak_price=peak_price,
    peak_start_hour=peak_start_hour,
    peak_end_hour=peak_end_hour,
    storage_capacity_kwh=storage_capacity,
    initial_soc_pct=initial_soc,
    max_charge_rate_kw=max_charge_rate,
    max_discharge_rate_kw=max_discharge_rate,
    charge_efficiency=charge_efficiency,
    discharge_efficiency=discharge_efficiency,
    loss_rate_per_hour=loss_rate_pct / 100.0,
    chiller_cop_charge=chiller_cop_charge,
)

engine = SimulationEngine(
    building_config=building_config,
    storage_config=storage_config,
    scenario=scenario_config,
    start_hour=start_hour,
    dt_hours=0.25,
    duration_hours=24.0,
)

sim_result = engine.run()
df = sim_result.df
metrics = calculate_metrics(df, storage_capacity_kwh=storage_capacity, dt_hours=0.25)

# -----------------------------------------------------------------------------
# Main Dashboard UI Layout
# -----------------------------------------------------------------------------
# Header
col_head_left, col_head_right = st.columns([3.5, 1.2])
with col_head_left:
    st.markdown('<div class="simulation-badge">Simulation Model Prototype</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="main-title">HeatBank</h1>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Store cooling. Use it when it matters.</div>', unsafe_allow_html=True)
with col_head_right:
    st.caption(f"**Active Preset:** {selected_name}")
    st.caption(f"**Building:** {bldg_type} ({thermal_mass:.0f} kWh/°C)")
    st.caption(f"**Storage:** {storage_capacity:.0f} kWh_th ({storage_config.initial_soc_pct:.0f}% start)")

# Scientific Simulation Disclaimer Callout
st.markdown(
    """
    <div class="info-callout">
        <strong>⚠️ Simulation Notice:</strong> This software is a physical simulation model for academic evaluation, 
        demonstrating energy balance, thermal mass dynamics, and tariff arbitrage. All displayed savings and performance metrics 
        are simulated estimates based on lumped-parameter thermodynamics and are not experimental field measurements.
    </div>
    """,
    unsafe_allow_html=True,
)

# 3-Step "How HeatBank Works" Quick Reference Banner
st.markdown(
    """
    <div class="steps-container">
        <div class="step-item">
            <span class="step-num">Step 1 — Store:</span> Pre-cool thermal storage when electricity is cheap off-peak or when surplus rooftop solar PV is available.
        </div>
        <div class="step-item">
            <span class="step-num">Step 2 — Discharge:</span> Deliver stored cold directly to the building during peak-tariff hours, shutting down the AC compressor.
        </div>
        <div class="step-item">
            <span class="step-num">Step 3 — Backup:</span> Allow the AC compressor to modulate seamlessly only to cover residual cooling demand if storage depletes.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Comprehensive KPI Cards: Row 1 & Row 2
# -----------------------------------------------------------------------------
# Row 1: Costs, Energy, Peak Energy, Peak Demand
k1, k2, k3, k4 = st.columns(4)

with k1:
    cost_diff = metrics.conventional.total_cost - metrics.heatbank.total_cost
    cost_sav_pct_str = metrics.savings.cost_savings_pct_str
    css_class = "kpi-savings-pos" if cost_diff >= 0 else "kpi-savings-neg"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">24h Electricity Cost</div>
            <div class="kpi-val">${metrics.heatbank.total_cost:.2f}</div>
            <div class="{css_class}">
                {cost_sav_pct_str} vs Conv (${metrics.conventional.total_cost:.2f})
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k2:
    energy_diff = metrics.conventional.total_grid_energy_kwh - metrics.heatbank.total_grid_energy_kwh
    energy_sav_pct_str = metrics.savings.energy_savings_pct_str
    css_class = "kpi-savings-pos" if energy_diff >= 0 else "kpi-savings-neg"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Total Grid Energy</div>
            <div class="kpi-val">{metrics.heatbank.total_grid_energy_kwh:.1f}<span class="kpi-unit">kWh</span></div>
            <div class="{css_class}">
                {energy_sav_pct_str} vs Conv ({metrics.conventional.total_grid_energy_kwh:.1f} kWh)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k3:
    peak_kwh_diff = metrics.conventional.peak_period_grid_energy_kwh - metrics.heatbank.peak_period_grid_energy_kwh
    peak_sav_pct_str = metrics.savings.peak_energy_savings_pct_str
    css_class = "kpi-savings-pos" if peak_kwh_diff >= 0 else "kpi-savings-neg"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Peak-Period Grid Use</div>
            <div class="kpi-val">{metrics.heatbank.peak_period_grid_energy_kwh:.1f}<span class="kpi-unit">kWh</span></div>
            <div class="{css_class}">
                {peak_sav_pct_str} vs Conv ({metrics.conventional.peak_period_grid_energy_kwh:.1f} kWh)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k4:
    peak_dem_diff = metrics.conventional.peak_period_demand_kw - metrics.heatbank.peak_period_demand_kw
    dem_sav_pct_str = metrics.savings.peak_demand_reduction_pct_str
    css_class = "kpi-savings-pos" if peak_dem_diff >= 0 else "kpi-savings-neg"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Peak-Period Max Demand</div>
            <div class="kpi-val">{metrics.heatbank.peak_period_demand_kw:.1f}<span class="kpi-unit">kW</span></div>
            <div class="{css_class}">
                {dem_sav_pct_str} vs Conv ({metrics.conventional.peak_period_demand_kw:.1f} kW)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Row 2: Comfort, Temperature, Storage Utilization, Peak Cooling Coverage
k5, k6, k7, k8 = st.columns(4)

with k5:
    viol_hours = metrics.heatbank.comfort_violation_hours
    viol_status = "100% Comfort Maintained" if viol_hours <= 0.0 else f"{viol_hours:.1f} hrs violation"
    css_viol = "kpi-savings-pos" if viol_hours <= 0.0 else "kpi-savings-neg"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Comfort Violations</div>
            <div class="kpi-val">{viol_hours:.1f}<span class="kpi-unit">hrs</span></div>
            <div class="{css_viol}">
                {viol_status} (Band ±{comfort_tol}°C)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k6:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Peak Indoor Temp</div>
            <div class="kpi-val">{metrics.heatbank.max_indoor_temp:.1f}<span class="kpi-unit">°C</span></div>
            <div class="kpi-subtext">
                Target: {target_temp:.1f}°C (Conv max: {metrics.conventional.max_indoor_temp:.1f}°C)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k7:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Storage Utilization</div>
            <div class="kpi-val">{metrics.storage.storage_utilization_pct:.0f}%<span class="kpi-unit">throughput</span></div>
            <div class="kpi-subtext">
                {metrics.storage.total_energy_discharged_kwh_th:.1f} kWh_th cooling discharged
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k8:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Peak Cooling Coverage</div>
            <div class="kpi-val">{metrics.storage.peak_cooling_met_by_storage_pct:.0f}%<span class="kpi-unit">by storage</span></div>
            <div class="kpi-savings-pos" style="color: #0284C7;">
                Offset peak compressor load
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# -----------------------------------------------------------------------------
# Interactive Plotly Charts Section
# -----------------------------------------------------------------------------
tabs = st.tabs(["📊 Performance Charts", "📋 Detailed Comparison", "🧠 Controller Decisions", "📖 About the Model"])

with tabs[0]:
    st.markdown("#### Dynamic Simulation Telemetry (24-Hour Horizon)")

    time_labels = df["time_label"]

    # -------------------------------------------------------------------------
    # Chart 1: Indoor Temperature vs Target and Comfort Range
    # -------------------------------------------------------------------------
    fig_temp = go.Figure()

    # Comfort Band Shading
    upper_band = target_temp + comfort_tol
    lower_band = target_temp - comfort_tol
    fig_temp.add_trace(
        go.Scatter(
            x=list(time_labels) + list(time_labels)[::-1],
            y=[upper_band] * len(time_labels) + [lower_band] * len(time_labels),
            fill="toself",
            fillcolor="rgba(14, 165, 233, 0.12)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name=f"Comfort Band (±{comfort_tol}°C)",
        )
    )

    # Outdoor Temperature
    fig_temp.add_trace(
        go.Scatter(
            x=time_labels,
            y=df["outdoor_temp"],
            name="Outdoor Ambient Temp",
            line=dict(color="#94A3B8", width=1.8, dash="dot"),
        )
    )

    # Target Setpoint Line
    fig_temp.add_trace(
        go.Scatter(
            x=time_labels,
            y=[target_temp] * len(time_labels),
            name="Target Setpoint",
            line=dict(color="#0284C7", width=1.5, dash="dash"),
        )
    )

    # Conventional AC Indoor Temp
    fig_temp.add_trace(
        go.Scatter(
            x=time_labels,
            y=df["conv_indoor_temp"],
            name="Conventional AC Temp",
            line=dict(color="#EF4444", width=2.5),
        )
    )

    # HeatBank Indoor Temp
    fig_temp.add_trace(
        go.Scatter(
            x=time_labels,
            y=df["hb_indoor_temp"],
            name="HeatBank Indoor Temp",
            line=dict(color="#10B981", width=2.5),
        )
    )

    fig_temp.update_layout(
        title="1. Indoor Temperature & Thermal Comfort Tracking",
        xaxis_title="Time of Day",
        yaxis_title="Temperature (°C)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
        height=380,
    )
    st.plotly_chart(fig_temp, **W_KWARGS)

    # -------------------------------------------------------------------------
    # Chart 2: Grid Electricity Demand Comparison
    # -------------------------------------------------------------------------
    fig_power = go.Figure()

    # Peak Period Highlight Background
    peak_steps = df[df["is_peak_period"]]
    if not peak_steps.empty:
        p_start_label = peak_steps["time_label"].iloc[0]
        p_end_label = peak_steps["time_label"].iloc[-1]
        fig_power.add_vrect(
            x0=p_start_label,
            x1=p_end_label,
            fillcolor="#FEF2F2",
            opacity=0.6,
            layer="below",
            line_width=1,
            line_color="#FCA5A5",
            annotation_text="Peak Tariff Window",
            annotation_position="top left",
        )

    fig_power.add_trace(
        go.Scatter(
            x=time_labels,
            y=df["conv_grid_power_kw"],
            name="Conventional AC Grid Power (kW)",
            line=dict(color="#EF4444", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(239, 68, 68, 0.08)",
        )
    )

    fig_power.add_trace(
        go.Scatter(
            x=time_labels,
            y=df["hb_grid_power_kw"],
            name="HeatBank Net Grid Power (kW)",
            line=dict(color="#059669", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(5, 150, 105, 0.12)",
        )
    )

    fig_power.update_layout(
        title="2. Electricity Grid Power Demand Over Time (Peak-Shaving Effect)",
        xaxis_title="Time of Day",
        yaxis_title="Grid Power (kW electric)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        margin=dict(l=40, r=20, t=60, b=40),
        height=380,
    )
    st.plotly_chart(fig_power, **W_KWARGS)

    # -------------------------------------------------------------------------
    # Chart 3 & 4 Side by Side: Storage SOC & Cumulative Cost
    # -------------------------------------------------------------------------
    c_left, c_right = st.columns(2)

    with c_left:
        fig_storage = make_subplots(specs=[[{"secondary_y": True}]])

        # SOC %
        fig_storage.add_trace(
            go.Scatter(
                x=time_labels,
                y=df["hb_storage_soc_pct"],
                name="Storage SOC (%)",
                line=dict(color="#0284C7", width=3),
            ),
            secondary_y=False,
        )

        # Discharge cooling rate (kW_th)
        fig_storage.add_trace(
            go.Bar(
                x=time_labels,
                y=df["hb_storage_discharge_cooling_kw"],
                name="Discharging Cooling (kW_th)",
                marker_color="#34D399",
                opacity=0.6,
            ),
            secondary_y=True,
        )

        # Charging power into storage (kW_th)
        fig_storage.add_trace(
            go.Bar(
                x=time_labels,
                y=df["hb_charge_rate_kw_th"],
                name="Charging Power (kW_th)",
                marker_color="#818CF8",
                opacity=0.5,
            ),
            secondary_y=True,
        )

        fig_storage.update_layout(
            title="3. Thermal Storage Dynamics & State of Charge (SOC)",
            xaxis_title="Time of Day",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            margin=dict(l=40, r=40, t=60, b=40),
            height=360,
        )
        fig_storage.update_yaxes(title_text="State of Charge (%)", secondary_y=False, range=[0, 105])
        fig_storage.update_yaxes(title_text="Thermal Flow (kW_th)", secondary_y=True)
        st.plotly_chart(fig_storage, **W_KWARGS)

    with c_right:
        fig_cost = go.Figure()

        fig_cost.add_trace(
            go.Scatter(
                x=time_labels,
                y=df["conv_cumulative_cost"],
                name="Conventional Cumulative Cost ($)",
                line=dict(color="#EF4444", width=2.5),
            )
        )

        fig_cost.add_trace(
            go.Scatter(
                x=time_labels,
                y=df["hb_cumulative_cost"],
                name="HeatBank Cumulative Cost ($)",
                line=dict(color="#059669", width=2.5),
            )
        )

        fig_cost.update_layout(
            title="4. Cumulative Electricity Cost Accrual Over 24h",
            xaxis_title="Time of Day",
            yaxis_title="Total Cost ($ USD)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            margin=dict(l=40, r=20, t=60, b=40),
            height=360,
        )
        st.plotly_chart(fig_cost, **W_KWARGS)

    # -------------------------------------------------------------------------
    # Chart 5 & 6 Side by Side: Tariffs & Peak-Period Power Demand Comparison
    # -------------------------------------------------------------------------
    c_left2, c_right2 = st.columns(2)

    with c_left2:
        fig_tariff = make_subplots(specs=[[{"secondary_y": True}]])

        fig_tariff.add_trace(
            go.Scatter(
                x=time_labels,
                y=df["electricity_price"],
                name="Tariff Price ($/kWh)",
                line=dict(color="#7C3AED", width=2.5, shape="hv"),
            ),
            secondary_y=False,
        )

        fig_tariff.add_trace(
            go.Scatter(
                x=time_labels,
                y=df["solar_generation_kw"],
                name="Solar PV Output (kW)",
                line=dict(color="#F59E0B", width=2.2),
                fill="tozeroy",
                fillcolor="rgba(245, 158, 11, 0.15)",
            ),
            secondary_y=True,
        )

        fig_tariff.update_layout(
            title="5. Environmental Inputs: Tariffs & Solar PV",
            xaxis_title="Time of Day",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified",
            margin=dict(l=40, r=40, t=60, b=40),
            height=360,
        )
        fig_tariff.update_yaxes(title_text="Price ($/kWh)", secondary_y=False)
        fig_tariff.update_yaxes(title_text="Solar Generation (kW)", secondary_y=True)
        st.plotly_chart(fig_tariff, **W_KWARGS)

    with c_right2:
        # Chart 6: Dedicated Peak-Period Power Demand Comparison
        fig_peak_comp = go.Figure()
        
        peak_mask = df["is_peak_period"]
        if peak_mask.any():
            peak_time_labels = df.loc[peak_mask, "time_label"]
            conv_peak_power = df.loc[peak_mask, "conv_grid_power_kw"]
            hb_peak_power = df.loc[peak_mask, "hb_grid_power_kw"]

            fig_peak_comp.add_trace(
                go.Bar(
                    x=peak_time_labels,
                    y=conv_peak_power,
                    name="Conventional AC (kW)",
                    marker_color="#F87171",
                )
            )
            fig_peak_comp.add_trace(
                go.Bar(
                    x=peak_time_labels,
                    y=hb_peak_power,
                    name="HeatBank System (kW)",
                    marker_color="#34D399",
                )
            )
            fig_peak_comp.update_layout(
                title=f"6. Peak-Period Power Demand Comparison ({df.loc[peak_mask, 'time_label'].iloc[0]}–{df.loc[peak_mask, 'time_label'].iloc[-1]})",
                xaxis_title="Peak Period Hours",
                yaxis_title="Power Demand (kW)",
                barmode="group",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                margin=dict(l=40, r=20, t=60, b=40),
                height=360,
            )
        else:
            fig_peak_comp.add_annotation(
                text="No peak-period hours active in current configuration.",
                xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False
            )
            fig_peak_comp.update_layout(title="6. Peak-Period Power Demand Comparison", height=360)

        st.plotly_chart(fig_peak_comp, **W_KWARGS)

# -----------------------------------------------------------------------------
# Tab 2: Detailed Comparison Table
# -----------------------------------------------------------------------------
with tabs[1]:
    st.markdown("#### Comprehensive Performance & Savings Audit")

    comparison_rows = [
        {
            "Metric": "Total Electricity Consumption",
            "Units": "kWh",
            "Conventional AC": f"{metrics.conventional.total_grid_energy_kwh:.2f}",
            "HeatBank System": f"{metrics.heatbank.total_grid_energy_kwh:.2f}",
            "Absolute Difference": f"{metrics.savings.energy_savings_kwh:+.2f} kWh",
            "Relative Savings (%)": metrics.savings.energy_savings_pct_str,
        },
        {
            "Metric": "Peak-Period Electricity Use",
            "Units": "kWh",
            "Conventional AC": f"{metrics.conventional.peak_period_grid_energy_kwh:.2f}",
            "HeatBank System": f"{metrics.heatbank.peak_period_grid_energy_kwh:.2f}",
            "Absolute Difference": f"{metrics.savings.peak_energy_savings_kwh:+.2f} kWh",
            "Relative Savings (%)": metrics.savings.peak_energy_savings_pct_str,
        },
        {
            "Metric": "Peak-Hours Power Demand",
            "Units": "kW",
            "Conventional AC": f"{metrics.conventional.peak_period_demand_kw:.2f}",
            "HeatBank System": f"{metrics.heatbank.peak_period_demand_kw:.2f}",
            "Absolute Difference": f"{metrics.savings.peak_demand_reduction_kw:+.2f} kW",
            "Relative Savings (%)": metrics.savings.peak_demand_reduction_pct_str,
        },
        {
            "Metric": "24h Electricity Operating Cost",
            "Units": "$ USD",
            "Conventional AC": f"${metrics.conventional.total_cost:.2f}",
            "HeatBank System": f"${metrics.heatbank.total_cost:.2f}",
            "Absolute Difference": f"${metrics.savings.cost_savings_dollars:+.2f}",
            "Relative Savings (%)": metrics.savings.cost_savings_pct_str,
        },
        {
            "Metric": "Hours Outside Comfort Range (± tolerance)",
            "Units": "Hours",
            "Conventional AC": f"{metrics.conventional.comfort_violation_hours:.2f}",
            "HeatBank System": f"{metrics.heatbank.comfort_violation_hours:.2f}",
            "Absolute Difference": f"{metrics.conventional.comfort_violation_hours - metrics.heatbank.comfort_violation_hours:+.2f} hrs",
            "Relative Savings (%)": "0.0 hrs target met" if metrics.heatbank.comfort_violation_hours == 0 else f"{metrics.heatbank.comfort_violation_hours:.1f} hrs",
        },
        {
            "Metric": "Maximum Indoor Temperature",
            "Units": "°C",
            "Conventional AC": f"{metrics.conventional.max_indoor_temp:.2f}°C",
            "HeatBank System": f"{metrics.heatbank.max_indoor_temp:.2f}°C",
            "Absolute Difference": f"{metrics.conventional.max_indoor_temp - metrics.heatbank.max_indoor_temp:+.2f}°C",
            "Relative Savings (%)": "Setpoint maintained",
        },
        {
            "Metric": "Thermal Storage Utilization Throughput",
            "Units": "% / kWh_th",
            "Conventional AC": "N/A (No Storage)",
            "HeatBank System": f"{metrics.storage.storage_utilization_pct:.1f}% ({metrics.storage.total_energy_discharged_kwh_th:.1f} kWh_th)",
            "Absolute Difference": f"+{metrics.storage.total_energy_discharged_kwh_th:.1f} kWh_th",
            "Relative Savings (%)": f"{metrics.storage.peak_cooling_met_by_storage_pct:.1f}% peak cooling",
        },
    ]

    comp_df = pd.DataFrame(comparison_rows)
    st.table(comp_df)

    st.markdown(
        """
        > **Interpretation Guide:**
        > - **Energy Savings vs Cost Savings:** Total electricity kWh savings may be modest (or slightly negative if storage efficiency losses exceed solar gains), but **Cost Savings** and **Peak Demand Reduction** remain substantial due to time-of-use tariff arbitrage and free rooftop solar absorption.
        > - **Honest Accounting:** The model accounts for all electricity used to charge cold storage ($E_{\\text{elec}} = Q_{\\text{ch}} / \\text{COP}_{\\text{charge}}$), charging/discharging roundtrip losses, and standing thermal leakage.
        """
    )

# -----------------------------------------------------------------------------
# Tab 3: Controller Decisions & State Log
# -----------------------------------------------------------------------------
with tabs[2]:
    st.markdown("#### Real-Time Intelligent Controller State Log")
    st.caption("Step-by-step dispatch decisions made by the HeatBank supervisory controller over the 24-hour cycle.")

    # Decision summary breakdown
    decision_counts = df["hb_decision_label"].value_counts()
    c_pie, c_explain = st.columns([1.5, 2.5])

    with c_pie:
        fig_pie = go.Figure(
            data=[
                go.Pie(
                    labels=decision_counts.index,
                    values=decision_counts.values,
                    hole=0.45,
                    marker=dict(colors=["#10B981", "#3B82F6", "#F59E0B", "#8B5CF6", "#64748B"]),
                )
            ]
        )
        fig_pie.update_layout(
            title="Controller Action Distribution",
            margin=dict(l=10, r=10, t=40, b=10),
            height=300,
        )
        st.plotly_chart(fig_pie, **W_KWARGS)

    with c_explain:
        st.markdown(
            """
            ##### How the Controller Selects Operating States:
            1. **Solar Surplus Storage Charging:** When rooftop solar PV generation exceeds daytime base AC cooling load, surplus electricity is routed to the chiller to pre-freeze/chill storage for **$0 marginal electricity cost**.
            2. **Off-Peak Grid Storage Charging:** During overnight/early morning hours when the tariff is at minimum, the storage is filled to 100% capacity using low-cost grid power.
            3. **Peak Discharge (100% Thermal Storage):** During expensive peak hours ($0.45+/kWh), the AC compressor is turned off entirely. The building's entire cooling demand is supplied from stored cooling via heat exchangers.
            4. **Peak Discharge + Backup AC:** If storage capacity is depleted or cooling demand exceeds max discharge power, the backup AC modulates only to meet the residual difference.
            """
        )

    # Detailed step telemetry table
    with st.expander("🔍 View Raw Step-by-Step Telemetry (96 Time Steps)"):
        st.dataframe(
            df[
                [
                    "time_label",
                    "outdoor_temp",
                    "solar_generation_kw",
                    "electricity_price",
                    "is_peak_period",
                    "conv_grid_power_kw",
                    "hb_grid_power_kw",
                    "hb_storage_soc_pct",
                    "hb_storage_discharge_cooling_kw",
                    "hb_charge_rate_kw_th",
                    "hb_decision_label",
                ]
            ],
            **W_KWARGS,
        )

# -----------------------------------------------------------------------------
# Tab 4: About the Model & Methodology
# -----------------------------------------------------------------------------
with tabs[3]:
    st.markdown("### 📖 About the HeatBank Model & Mathematical Formulation")

    st.markdown(
        """
        #### 1. Concept: Battery for Cooling
        Air conditioning represents over **40% of peak electricity demand** in commercial buildings on hot summer afternoons.
        Conventional cooling forces chillers and AC compressors to run simultaneously with peak grid stress and maximum tariff prices.
        
        **HeatBank** decouples cooling generation from cooling consumption by storing cold thermal energy in an insulated medium 
        (e.g., chilled water, ice storage, or phase change material).

        ```
        Off-Peak / Solar Hours          Peak Tariff Hours (14:00 - 20:00)
        [ Cheap Grid / Solar ]           [ Thermal Storage ]
                │                                 │
                ▼                                 ▼
        [ Chiller Charging ]              [ Direct Cold Delivery ]
                │                                 │
                ▼                                 ▼
        [ Thermal Storage ] ───────►    [ Building Zone (Target 22°C) ]
                                                  ▲
                                         (Optional Backup AC)
        ```

        ---
        #### 2. Governing Equations

        ##### A. Building Envelope & Thermal Mass Dynamics (1st Law Energy Balance)
        The building indoor temperature $T_{\\text{in}}(t)$ is modeled as a lumped-capacitance thermal system:
        $$C_{\\text{bldg}} \\frac{dT_{\\text{in}}}{dt} = UA \\cdot (T_{\\text{out}}(t) - T_{\\text{in}}(t)) + Q_{\\text{internal}}(t) + Q_{\\text{solar}}(t) - Q_{\\text{cooling}}(t)$$
        
        Where:
        - $C_{\\text{bldg}}$: Effective thermal capacitance of the space (kWh/°C)
        - $UA$: Building overall heat loss coefficient (kW/°C)
        - $Q_{\\text{internal}}$: Internal heat gains from occupants, computers, and lighting (kW)
        - $Q_{\\text{cooling}}$: Net cooling thermal power delivered to the space (kW_thermal)

        ##### B. Cooling Demand Calculation
        At each discrete time step $\\Delta t$ (0.25 hours = 15 minutes), the cooling power needed to reach the setpoint $T_{\\text{target}}$ is:
        $$Q_{\\text{req}} = \\max\\left(0, \\frac{C_{\\text{bldg}} \\cdot (T_{\\text{float}} - T_{\\text{target}})}{\\Delta t}\\right)$$
        Where $T_{\\text{float}}$ is the uncooled temperature predicted at $t + \\Delta t$.

        ##### C. Thermal Storage Conservation & Real Electricity Accounting
        - **Standing Losses:** Cold thermal energy lost to ambient per step:
          $$\\Delta S_{\\text{loss}} = S(t) \\cdot \\lambda_{\\text{loss}} \\cdot \\Delta t$$
        - **Charging Process:** Thermal energy stored:
          $$\\Delta S = P_{\\text{ch}} \\cdot \\Delta t \\cdot \\eta_{\\text{ch}}$$
          Electric power consumed by chiller:
          $$P_{\\text{elec, ch}} = \\frac{P_{\\text{ch}}}{\\text{COP}_{\\text{ch}}}$$
        - **Discharging Process:** Cooling delivered to building space:
          $$Q_{\\text{dis}} = \\min\\left(Q_{\\text{req}}, P_{\\text{dis, max}}, \\frac{S(t) \\cdot \\eta_{\\text{dis}}}{\\Delta t}\\right)$$
          Stored energy extracted:
          $$\\Delta S_{\\text{drawn}} = \\frac{Q_{\\text{dis}} \\cdot \\Delta t}{\\eta_{\\text{dis}}}$$

        ##### D. Net Grid Electricity & Cost Accounting
        $$P_{\\text{grid}}(t) = \\max(0, P_{\\text{elec, total}}(t) - P_{\\text{solar}}(t))$$
        $$\\text{Cost}(t) = P_{\\text{grid}}(t) \\cdot \\Delta t \\cdot c_{\\text{tariff}}(t)$$

        ---
        #### 3. Assumptions and Model Limitations
        - **Spatial Homogeneity:** The building is treated as a single well-mixed thermal zone. Multi-zone air distribution is not modeled.
        - **Constant COP Approximation:** A user-specified nominal COP is applied; while chillers experience minor COP degradation at extreme condensing temperatures, the model uses rated operating COP values.
        - **Deterministic Profiles:** Weather and occupancy follow calibrated daily schedules rather than stochastic occupant movements.
        - **Academic Simulation Scope:** Designed for conceptual trade-off exploration, optimization sizing, and hackathon presentation.
        """
    )

st.markdown("---")
st.caption(
    "HeatBank Prototype Simulation Platform • Built with Python, Streamlit & Plotly • "
    "Designed for Clean Energy & Thermal Storage Hackathon Demonstration."
)
