"""
HeatBank Thermal Energy Simulation Engine.

Contains modular components for:
- Building thermal dynamics (building.py)
- Thermal energy storage (storage.py)
- Operational controllers (controller.py)
- Standard scenario profiles (scenarios.py)
- Performance and savings metrics (metrics.py)
"""

from .building import BuildingConfig, BuildingModel
from .storage import ThermalStorageConfig, ThermalStorage
from .controller import SimulationEngine, SimulationResult, SimulationStepResult
from .scenarios import ScenarioPreset, get_scenario_preset, list_scenarios
from .metrics import SimulationMetrics, calculate_metrics, calculate_savings

__all__ = [
    "BuildingConfig",
    "BuildingModel",
    "ThermalStorageConfig",
    "ThermalStorage",
    "SimulationEngine",
    "SimulationResult",
    "SimulationStepResult",
    "ScenarioPreset",
    "get_scenario_preset",
    "list_scenarios",
    "SimulationMetrics",
    "calculate_metrics",
    "calculate_savings",
]
