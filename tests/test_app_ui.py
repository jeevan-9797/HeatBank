"""
Automated UI and integration test for Streamlit app.py using Streamlit AppTest.
"""

from pathlib import Path
import pytest
from streamlit.testing.v1 import AppTest

APP_PATH = Path(__file__).parent.parent / "app.py"


def test_app_loads_and_renders_without_exceptions():
    """Verify app.py initializes and executes full simulation without error."""
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception
    # Check that main title exists
    assert any("HeatBank" in str(el.value) for el in at.markdown)


def test_app_preset_selection():
    """Verify selecting a scenario preset updates the app state cleanly."""
    at = AppTest.from_file(APP_PATH, default_timeout=30).run()
    assert not at.exception

    # Select box for scenario preset
    scenario_sb = at.sidebar.selectbox[0]
    assert scenario_sb.label == "Scenario Preset"

    # Switch to High Electricity Price Peak
    scenario_sb.select("High Electricity Price Peak").run()
    assert not at.exception
    assert at.session_state["selected_scenario_id"] == "high_price_peak"
