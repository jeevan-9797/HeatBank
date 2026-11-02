"""
HeatBank Thermal Energy Simulation Application.
Interactive web dashboard comparing Conventional AC vs HeatBank thermal storage.
"""

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
        padding: 1.1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 0.8rem;
    }
    .kpi-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 0.25rem;
    }
    .kpi-val {
        font-size: 1.65rem;
        font-weight: 800;
        color: #0F172A;
    }
    .kpi-unit {
        font-size: 0.9rem;
        font-weight: 500;
        color: #64748B;
        margin-left: 0.2rem;
    }
    .kpi-savings-pos {
        color: #059669;
        font-size: 0.88rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }
    .kpi-savings-neg {
        color: #DC2626;
        font-size: 0.88rem;
        font-weight: 700;
        margin-top: 0.3rem;
    }
    .info-callout {
        background-color: #F8FAFC;
        border-left: 4px solid #0284C7;
        padding: 0.75rem 1rem;
        border-radius: 0.25rem;
        font-size: 0.88rem;
        color: #334155;
        margin-bottom: 1.2rem;
    }
    .mode-tag-conv {
        background-color: #FEE2E2;
        color: #991B1B;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
    }
    .mode-tag-hb {
        background-color: #DCFCE7;
        color: #166534;
        padding: 0.15rem 0.45rem;
        border-radius: 4px;
        font-weight: 600;
        font-size: 0.78rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Session State Initialization & Presets
# -----------------------------------------------------------------------------
scenario_list = list_scenarios()
scenario_ids = [s["id"] for s in scenario_list]
scenario_names = [s["name"] for s in scenario_list]
name_to_id = {s["name"]: s["id"] for s in scenario_list}
id_to_name = {s["id"]: s["name"] for s in scenario_list}

if "selected_scenario_id" not in st.session_state:
    st.session_state["selected_scenario_id"] = "hot_sunny"

if "reset_trigger" not in st.session_state:
    st.session_state["reset_trigger"] = 0


def apply_preset(preset_id: str):
    """Load preset values into session state keys."""
    preset = get_scenario_preset(preset_id)
    st.session_state["bldg_type"] = preset.building_type
    st.session_state["thermal_mass"] = float(preset.thermal_mass)
    st.session_state["heat_loss_coeff"] = float(preset.heat_loss_coeff)
    st.session_state["initial_temp"] = float(preset.initial_temp)
    st.session_state["target_temp"] = float(preset.target_temp)
    st.session_state["comfort_tol"] = float(preset.comfort_tolerance)
    st.session_state["ac_cop"] = float(preset.ac_cop)
    st.session_state["max_ac_power"] = float(preset.max_ac_cooling_power)

    st.session_state["base_temp"] = float(preset.base_temp)
    st.session_state["temp_variation"] = float(preset.temp_variation)
    st.session_state["custom_temp_offset"] = float(preset.custom_temp_offset)

    st.session_state["solar_peak_kw"] = float(preset.solar_peak_kw)
    st.session_state["cloud_cover"] = float(preset.cloud_cover_factor)

    st.session_state["off_peak_price"] = float(preset.off_peak_price)
    st.session_state["mid_peak_price"] = float(preset.mid_peak_price)
    st.session_state["peak_price"] = float(preset.peak_price)
    st.session_state["peak_start_hour"] = float(preset.peak_start_hour)
    st.session_state["peak_end_hour"] = float(preset.peak_end_hour)

    st.session_state["storage_capacity"] = float(preset.storage_capacity_kwh)
    st.session_state["initial_soc"] = float(preset.initial_soc_pct)
    st.session_state["max_charge_rate"] = float(preset.max_charge_rate_kw)
    st.session_state["max_discharge_rate"] = float(preset.max_discharge_rate_kw)
    st.session_state["charge_efficiency"] = float(preset.charge_efficiency)
    st.session_state["discharge_efficiency"] = float(preset.discharge_efficiency)
    st.session_state["loss_rate_pct"] = float(preset.loss_rate_per_hour * 100.0)
    st.session_state["chiller_cop_charge"] = float(preset.chiller_cop_charge)


# If session keys are not yet set, initialize from current preset
if "bldg_type" not in st.session_state:
    apply_preset(st.session_state["selected_scenario_id"])

# -----------------------------------------------------------------------------
# Sidebar: Controls & Configuration
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚙️ Simulation Controls")

    # Preset Selector
    current_preset_name = id_to_name.get(st.session_state["selected_scenario_id"], "Hot Sunny Day")
    selected_name = st.selectbox(
        "Scenario Preset",
        options=scenario_names,
        index=scenario_names.index(current_preset_name),
        help="Select a benchmark scenario. Changing this preset populates verified parameters across all subsystems.",
    )

    new_preset_id = name_to_id[selected_name]
    if new_preset_id != st.session_state["selected_scenario_id"]:
        st.session_state["selected_scenario_id"] = new_preset_id
        apply_preset(new_preset_id)
        st.rerun()

    active_preset = get_scenario_preset(st.session_state["selected_scenario_id"])
    st.caption(f"ℹ️ *{active_preset.description}*")

    # Action Buttons: Run & Reset
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        run_sim = st.button("▶ Run", type="primary", use_container_width=True)
    with col_btn2:
        if st.button("↺ Reset Defaults", use_container_width=True):
            apply_preset(st.session_state["selected_scenario_id"])
            st.rerun()

    st.divider()

    # 1. Building Parameters
    with st.expander("🏢 Building Parameters", expanded=False):
        bldg_type = st.selectbox(
            "Building Archetype",
            options=["Small Office", "Classroom", "Custom"],
            index=["Small Office", "Classroom", "Custom"].index(st.session_state.get("bldg_type", "Small Office")),
            help="Determines baseline internal heat gains from occupants and equipment.",
        )
        thermal_mass = st.number_input(
            "Thermal Mass (kWh/°C)",
            min_value=2.0,
            max_value=150.0,
            value=float(st.session_state.get("thermal_mass", 20.0)),
            step=2.0,
            help="Heat capacity of the building envelope and internal mass.",
        )
        heat_loss_coeff = st.number_input(
            "Envelope Heat Loss UA (kW/°C)",
            min_value=0.1,
            max_value=10.0,
            value=float(st.session_state.get("heat_loss_coeff", 1.2)),
            step=0.1,
            help="Thermal conductivity of the envelope to outside temperature differences.",
        )
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            initial_temp = st.number_input(
                "Initial Temp (°C)",
                min_value=12.0,
                max_value=35.0,
                value=float(st.session_state.get("initial_temp", 24.0)),
                step=0.5,
            )
        with col_t2:
            target_temp = st.number_input(
                "Target Temp (°C)",
                min_value=16.0,
                max_value=28.0,
                value=float(st.session_state.get("target_temp", 22.0)),
                step=0.5,
                help="Desired comfort temperature setpoint.",
            )
        comfort_tol = st.slider(
            "Comfort Band Tolerance (±°C)",
            min_value=0.5,
            max_value=3.0,
            value=float(st.session_state.get("comfort_tol", 1.5)),
            step=0.1,
            help="Allowable indoor deviation before counting as a comfort violation.",
        )
        ac_cop = st.number_input(
            "AC Direct Cooling COP",
            min_value=1.5,
            max_value=6.0,
            value=float(st.session_state.get("ac_cop", 3.2)),
            step=0.1,
            help="Coefficient of Performance for the air conditioning compressor.",
        )
        max_ac_power = st.number_input(
            "Max AC Cooling Power (kW_th)",
            min_value=5.0,
            max_value=100.0,
            value=float(st.session_state.get("max_ac_power", 25.0)),
            step=5.0,
        )

    # 2. Weather & Climate
    with st.expander("☀️ Weather & Solar Profile", expanded=False):
        base_temp = st.number_input(
            "Mean Daily Temp (°C)",
            min_value=10.0,
            max_value=45.0,
            value=float(st.session_state.get("base_temp", 28.0)),
            step=1.0,
        )
        temp_variation = st.number_input(
            "Daily Temp Swing (°C)",
            min_value=2.0,
            max_value=25.0,
            value=float(st.session_state.get("temp_variation", 10.0)),
            step=1.0,
            help="Difference between peak afternoon temperature and morning minimum.",
        )
        custom_temp_offset = st.slider(
            "Custom Temp Offset (°C)",
            min_value=-10.0,
            max_value=10.0,
            value=float(st.session_state.get("custom_temp_offset", 0.0)),
            step=0.5,
            help="Global offset applied across the entire 24h temperature profile.",
        )
        solar_peak_kw = st.number_input(
            "Rooftop Solar Peak (kW_e)",
            min_value=0.0,
            max_value=30.0,
            value=float(st.session_state.get("solar_peak_kw", 6.0)),
            step=0.5,
            help="Rated maximum electric power output of on-site solar PV at solar noon.",
        )
        cloud_cover = st.slider(
            "Solar Clearness Index",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.get("cloud_cover", 1.0)),
            step=0.05,
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
                value=float(st.session_state.get("off_peak_price", 0.12)),
                step=0.01,
            )
        with col_p2:
            peak_price = st.number_input(
                "On-Peak Price ($/kWh)",
                min_value=0.05,
                max_value=2.0,
                value=float(st.session_state.get("peak_price", 0.45)),
                step=0.02,
            )
        mid_peak_price = st.number_input(
            "Mid-Peak / Shoulder ($/kWh)",
            min_value=0.02,
            max_value=1.5,
            value=float(st.session_state.get("mid_peak_price", 0.20)),
            step=0.01,
        )
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            peak_start_hour = st.number_input(
                "Peak Start (Hour)",
                min_value=0.0,
                max_value=23.0,
                value=float(st.session_state.get("peak_start_hour", 14.0)),
                step=1.0,
            )
        with col_h2:
            peak_end_hour = st.number_input(
                "Peak End (Hour)",
                min_value=1.0,
                max_value=24.0,
                value=float(st.session_state.get("peak_end_hour", 20.0)),
                step=1.0,
            )

    # 4. Thermal Energy Storage (HeatBank)
    with st.expander("🧊 HeatBank Thermal Storage", expanded=False):
        storage_capacity = st.number_input(
            "Storage Capacity (kWh_th)",
            min_value=0.0,
            max_value=200.0,
            value=float(st.session_state.get("storage_capacity", 50.0)),
            step=5.0,
            help="Total thermal cold storage capacity in kWh thermal.",
        )
        initial_soc = st.slider(
            "Initial SOC (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(st.session_state.get("initial_soc", 20.0)),
            step=5.0,
            help="State of charge at start of simulation.",
        )
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            max_charge_rate = st.number_input(
                "Max Charge Rate (kW_th)",
                min_value=0.1,
                max_value=50.0,
                value=float(st.session_state.get("max_charge_rate", 12.0)),
                step=1.0,
            )
        with col_r2:
            max_discharge_rate = st.number_input(
                "Max Discharge (kW_th)",
                min_value=0.1,
                max_value=50.0,
                value=float(st.session_state.get("max_discharge_rate", 15.0)),
                step=1.0,
            )
        col_e1, col_e2 = st.columns(2)
        with col_e1:
            charge_efficiency = st.slider(
                "Charge Efficiency",
                min_value=0.5,
                max_value=1.0,
                value=float(st.session_state.get("charge_efficiency", 0.90)),
                step=0.02,
            )
        with col_e2:
            discharge_efficiency = st.slider(
                "Discharge Efficiency",
                min_value=0.5,
                max_value=1.0,
                value=float(st.session_state.get("discharge_efficiency", 0.95)),
                step=0.02,
            )
        loss_rate_pct = st.slider(
            "Thermal Loss Rate (%/hr)",
            min_value=0.0,
            max_value=10.0,
            value=float(st.session_state.get("loss_rate_pct", 1.0)),
            step=0.2,
            help="Standing thermal loss rate of stored cold to ambient per hour.",
        )
        chiller_cop_charge = st.number_input(
            "Chiller Charging COP",
            min_value=1.2,
            max_value=6.0,
            value=float(st.session_state.get("chiller_cop_charge", 3.0)),
            step=0.1,
            help="COP when chiller operates at lower evaporator temperatures to freeze/chill storage.",
        )

    # 5. Simulation Horizon & Resolution
    with st.expander("⏱️ Horizon & Time Step", expanded=False):
        start_hour = st.slider("Simulation Start Hour", min_value=0, max_value=23, value=0, step=1)
        st.caption("Duration: **24 Hours** | Resolution: **15 Minutes (96 steps)**")

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
    id="runtime_custom",
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
    st.caption(f"**Active Scenario:** {selected_name}")
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

# -----------------------------------------------------------------------------
# KPI Cards: Top-Level Metrics
# -----------------------------------------------------------------------------
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

with k3:
    peak_dem_diff = metrics.conventional.peak_period_demand_kw - metrics.heatbank.peak_period_demand_kw
    dem_sav_pct_str = metrics.savings.peak_demand_reduction_pct_str
    css_class = "kpi-savings-pos" if peak_dem_diff >= 0 else "kpi-savings-neg"
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Peak-Hours Max Demand</div>
            <div class="kpi-val">{metrics.heatbank.peak_period_demand_kw:.1f}<span class="kpi-unit">kW</span></div>
            <div class="{css_class}">
                {dem_sav_pct_str} vs Conv ({metrics.conventional.peak_period_demand_kw:.1f} kW)
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with k4:
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-title">Thermal Storage Utilization</div>
            <div class="kpi-val">{metrics.storage.storage_utilization_pct:.0f}%<span class="kpi-unit">throughput</span></div>
            <div class="kpi-savings-pos" style="color: #0284C7;">
                {metrics.storage.peak_cooling_met_by_storage_pct:.0f}% of peak cooling met by HeatBank
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
    st.plotly_chart(fig_temp, use_container_width=True)

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
            annotation_text="High-Price Peak Window",
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
    st.plotly_chart(fig_power, use_container_width=True)

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
        st.plotly_chart(fig_storage, use_container_width=True)

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
        st.plotly_chart(fig_cost, use_container_width=True)

    # -------------------------------------------------------------------------
    # Chart 5 & 6 Side by Side: Tariffs & Peak Demand Analysis
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
        st.plotly_chart(fig_tariff, use_container_width=True)

    with c_right2:
        fig_bar = go.Figure()

        categories = ["Total 24h Grid (kWh)", "Peak-Window Grid (kWh)", "Peak Demand (kW)", "Total Cost ($)"]
        conv_vals = [
            metrics.conventional.total_grid_energy_kwh,
            metrics.conventional.peak_period_grid_energy_kwh,
            metrics.conventional.peak_period_demand_kw,
            metrics.conventional.total_cost,
        ]
        hb_vals = [
            metrics.heatbank.total_grid_energy_kwh,
            metrics.heatbank.peak_period_grid_energy_kwh,
            metrics.heatbank.peak_period_demand_kw,
            metrics.heatbank.total_cost,
        ]

        fig_bar.add_trace(
            go.Bar(
                name="Conventional AC",
                x=categories,
                y=conv_vals,
                marker_color="#F87171",
            )
        )
        fig_bar.add_trace(
            go.Bar(
                name="HeatBank",
                x=categories,
                y=hb_vals,
                marker_color="#34D399",
            )
        )

        fig_bar.update_layout(
            title="6. Direct System Comparison Across Key Metrics",
            barmode="group",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=60, b=40),
            height=360,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

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
        > - **Honest Accounting:** The model accounts for all electricity used to charge cold storage ($E_{\text{elec}} = Q_{\text{ch}} / \text{COP}_{\text{charge}}$), charging/discharging roundtrip losses, and standing thermal leakage.
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
        st.plotly_chart(fig_pie, use_container_width=True)

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
            use_container_width=True,
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
