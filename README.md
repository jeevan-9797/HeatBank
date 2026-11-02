# HeatBank — Intelligent Thermal Energy Management Simulation

> **Store cooling. Use it when it matters.**

[![Tests](https://img.shields.io/badge/pytest-21%20passed-brightgreen.svg)](tests/)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.66%2B-FF4B4B.svg)](https://streamlit.io/)
[![Plotly](https://img.shields.io/badge/plotly-interactive-3B82F6.svg)](https://plotly.com/)

HeatBank is an intelligent thermal energy management simulator that models a cold thermal energy storage (CTES) system—functioning as a **"battery for cooling."** 

Commercial and institutional air conditioning accounts for over **40% of peak electricity demand** on hot summer afternoons, driving high peak-demand charges and stressing electrical grids. HeatBank decouples cooling generation from cooling consumption: it pre-cools thermal storage during off-peak hours (or absorbs free surplus rooftop solar generation) and discharges that stored cooling during expensive peak hours, dramatically lowering energy costs and peak grid strain while maintaining indoor comfort.

---

> ⚠️ **Simulation Notice:** HeatBank is a software simulation prototype built for research, system exploration, and academic hackathon demonstrations. All displayed savings and performance metrics are simulated estimates derived from first-principles thermodynamic balances and have not been experimentally validated on physical field equipment.

---

## 🌟 Key Features

- **Physically Grounded 1R-1C Thermal Model:** Discrete-time energy conservation balancing outdoor ambient heat conduction, internal occupant/equipment loads, and solar gains.
- **Two Head-to-Head Operating Modes:**
  - **Mode 1 — Conventional AC:** Real-time direct cooling via air conditioning to meet indoor setpoints.
  - **Mode 2 — HeatBank System:** Dynamic scheduling with thermal storage arbitrage, solar surplus absorption, and backup AC dispatch.
- **Fair and Rigorous Energy Accounting:**
  - Accounts for chiller electrical power to charge storage: $P_{\text{elec, charge}} = P_{\text{charge, thermal}} / \text{COP}_{\text{charge}}$.
  - Incorporates round-trip charging ($\eta_{\text{charge}}$) and discharging ($\eta_{\text{discharge}}$) efficiencies.
  - Accounts for standing thermal dissipation losses ($\lambda_{\text{loss}}$) over time.
  - Honestly reports negative savings when losses outweigh tariff differentials.
- **6 Interactive Plotly Visualizations:**
  1. Indoor temperature tracking vs. target setpoint and comfort tolerance band.
  2. Grid electricity demand over time showing peak-shaving effects.
  3. Thermal storage state of charge (SOC %) and thermal flow rates (kW_th).
  4. Cumulative electricity operating cost comparison ($ USD).
  5. Time-of-Use (TOU) electricity tariff schedule and rooftop solar PV generation.
  6. Side-by-side grouped peak demand and consumption comparisons.
- **Pre-Configured Benchmark Scenarios:**
  - **Hot Sunny Day:** High ambient heat with strong midday solar availability.
  - **Hot Cloudy Day:** Overcast conditions testing grid-only off-peak charging economics.
  - **High Electricity Price Peak:** Extreme peak-to-off-peak tariff differentials ($0.65/kWh peak).
  - **Low Storage Capacity:** Storage exhaustion and seamless backup AC transition.
  - **Custom Settings:** Fully configurable building, weather, tariff, and storage parameters.
- **Safety & Zero-Division Resilience:**
  - Handles zero baseline consumption (`N/A (zero baseline)`).
  - Robust input validation rejecting negative or physically impossible values.
  - Fully automated unit and UI testing suite with 21 passing test cases.

---

## 🏗️ Architecture & Code Organization

The repository strictly separates the simulation engine, physical equations, metrics calculation, and UI:

```text
heatbank/
├── app.py                      # Interactive Streamlit web dashboard & Plotly charts
├── simulation/                 # Reusable, standalone simulation engine
│   ├── __init__.py             # Public package exports
│   ├── building.py             # 1R-1C Lumped capacitance building thermal model
│   ├── storage.py              # Cold thermal energy storage (CTES) state machine
│   ├── controller.py           # 24-hour simulation coordinator & dispatch controller
│   ├── scenarios.py            # Diurnal weather, solar PV, and TOU tariff generators
│   └── metrics.py              # KPI calculation, comfort violation hours, & safe savings
├── tests/                      # Automated test suite (21 pytest cases)
│   ├── __init__.py
│   ├── test_simulation.py      # 24-hour simulation integrity & bounds verification
│   ├── test_storage.py         # CTES conservation, limits, and standby loss tests
│   ├── test_metrics.py         # Zero-division safety and metrics tests
│   ├── test_energy_consistency.py # Strict 1st-law energy balance tests
│   └── test_app_ui.py          # End-to-end Streamlit AppTest integration tests
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
└── .gitignore                  # Clean repository ignores
```

---

## 📐 Mathematical Formulation & Physics

### 1. Building Envelope Thermal Dynamics (First Law of Thermodynamics)
The indoor air and envelope temperature $T_{\text{in}}(t)$ is governed by:

$$C_{\text{bldg}} \frac{dT_{\text{in}}}{dt} = UA \cdot (T_{\text{out}}(t) - T_{\text{in}}(t)) + Q_{\text{internal}}(t) + Q_{\text{solar}}(t) - Q_{\text{cooling}}(t)$$

Where:
- $C_{\text{bldg}}$: Effective thermal mass / heat capacity of space and furnishings ($\text{kWh}/^\circ\text{C}$).
- $UA$: Overall envelope heat-loss conductance ($\text{kW}/^\circ\text{C}$).
- $Q_{\text{internal}}$: Internal heat gains from occupants, computers, and lighting ($\text{kW}$).
- $Q_{\text{solar}}$: Solar heat gain through windows ($\text{kW}$).
- $Q_{\text{cooling}}$: Cooling power delivered to the space ($\text{kW}_{\text{th}}$).

### 2. Cooling Demand Calculation
At each discrete time step $\Delta t = 0.25\text{ h}$ (15 minutes), the uncooled free-floating temperature is calculated:

$$T_{\text{float}} = T_{\text{in}}(t) + \frac{\left[ UA \cdot (T_{\text{out}} - T_{\text{in}}) + Q_{\text{internal}} + Q_{\text{solar}} \right] \cdot \Delta t}{C_{\text{bldg}}}$$

If $T_{\text{float}} > T_{\text{target}}$, the required cooling power is:

$$Q_{\text{req}} = \min\left( Q_{\text{max, AC}}, \frac{C_{\text{bldg}} \cdot (T_{\text{float}} - T_{\text{target}})}{\Delta t} \right)$$

### 3. Thermal Energy Storage State Machine
For storage capacity $S_{\text{cap}}$ (kWh thermal) and stored cold $S(t)$:

- **Standing Losses:** Cold thermal energy lost to ambient per step:
  $$\Delta S_{\text{loss}} = S(t) \cdot \lambda_{\text{loss}} \cdot \Delta t$$
- **Charging State:** Thermal cooling added to storage medium:
  $$\Delta S_{\text{stored}} = P_{\text{charge}} \cdot \Delta t \cdot \eta_{\text{charge}}$$
  Electricity consumed by chiller to generate this charge:
  $$P_{\text{elec, charge}} = \frac{P_{\text{charge}}}{\text{COP}_{\text{chiller, charge}}}$$
- **Discharging State:** Cooling power delivered to offset building load:
  $$Q_{\text{discharge}} = \min\left( Q_{\text{req}}, P_{\text{dis, max}}, \frac{S(t) \cdot \eta_{\text{discharge}}}{\Delta t} \right)$$
  Stored energy extracted from CTES:
  $$\Delta S_{\text{drawn}} = \frac{Q_{\text{discharge}} \cdot \Delta t}{\eta_{\text{discharge}}}$$

### 4. Electricity & Operating Cost Accounting
For both systems, on-site solar generation $P_{\text{solar}}(t)$ is consumed first behind the meter:

$$P_{\text{grid}}(t) = \max\left(0, P_{\text{elec, total}}(t) - P_{\text{solar}}(t)\right)$$
$$\text{Cost}(t) = P_{\text{grid}}(t) \cdot \Delta t \cdot c_{\text{tariff}}(t)$$

---

## 🚀 Quickstart & Installation

### Prerequisites
- Python 3.11, 3.12, 3.13, or 3.14
- Git

### 1. Clone & Set Up Virtual Environment

```bash
git clone https://github.com/your-username/heatbank.git
cd heatbank

# Create and activate virtual environment
python -m venv venv

# On Windows:
.\venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the Interactive Web Application

```bash
streamlit run app.py
```
*(Or alternatively: `python -m streamlit run app.py`)*

The web dashboard will automatically open in your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests

HeatBank includes an extensive automated test suite covering storage physics, 24-hour simulation continuity, zero solar, zero capacity, energy conservation, safe zero-division calculations, and Streamlit UI execution:

```bash
pytest
```

To run with verbose output:
```bash
pytest -v
```

All 21 test cases execute in ~2 seconds.

---

## 💡 How HeatBank Works (3-Step Lifecycle)

1. **Store Cooling When Cheap or Free:** During nighttime off-peak tariff hours (or midday when surplus solar PV is available), HeatBank runs the chiller to charge the thermal storage medium.
2. **Discharge During Expensive Peak Hours:** When tariffs surge during afternoon peak hours (e.g., 14:00–20:00), the AC compressor is turned off. Cooling is drawn directly from the thermal storage tank.
3. **Seamless Backup AC Dispatch:** If storage is depleted or demand exceeds discharge limits, the backup AC compressor modulates to supply only the marginal residual cooling, ensuring indoor comfort setpoints are never compromised.

---

## 🔍 Assumptions and Limitations

- **Lumped Thermal Zone:** The building is modeled as a single well-mixed thermal capacitance zone; spatial temperature gradients between interior rooms are not modeled.
- **Linearized Heat Loss:** Envelope conductive heat transfer follows Newton's Law of Cooling ($UA \cdot \Delta T$). Radiation exchange between outdoor building surfaces and the sky is subsumed into the effective ambient temperature.
- **Deterministic Profiles:** Internal gains, tariffs, and solar profiles follow standard scheduled diurnal functions rather than stochastic real-time weather feeds.
- **Constant Equipment COP:** Nominal COP ratings are utilized; compressor performance degradation at extreme ambient temperatures is simplified.
- **Future Improvements:** Planned expansions include multi-zone building modeling, model predictive control (MPC) with price forecasting, hardware-in-the-loop (HIL) serial communication with physical ESP32 temperature sensors, and dynamic humidity/latent heat balance.

---

## 📄 License

MIT License. Designed and built for clean energy research and thermal storage innovation.
