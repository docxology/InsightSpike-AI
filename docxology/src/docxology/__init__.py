"""Docxology — InsightSpike-AI Sidecar Framework.

A comprehensive sidecar framework for orchestrating, testing, analyzing,
and visualizing all methods from the InsightSpike-AI package.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Kazuyoshi Miyauchi"

# Lazy imports to avoid circular dependencies
def __getattr__(name: str):
    """Lazy module loading for heavy imports."""
    if name == "ModuleScanner":
        from .discovery import ModuleScanner
        return ModuleScanner
    if name == "MethodRegistry":
        from .discovery import MethodRegistry
        return MethodRegistry
    if name == "Pipeline":
        from .orchestrator import Pipeline
        return Pipeline
    if name == "ScriptRunner":
        from .runners import ScriptRunner
        return ScriptRunner
    if name == "ModuleRunner":
        from .runners import ModuleRunner
        return ModuleRunner
    if name == "ResultsAnalyzer":
        from .analysis import ResultsAnalyzer
        return ResultsAnalyzer
    if name == "Plotter":
        from .visualization import Plotter
        return Plotter
    if name == "Exporter":
        from .visualization import Exporter
        return Exporter
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "__author__",
    # Discovery
    "ModuleScanner",
    "MethodRegistry",
    # Orchestrator
    "Pipeline",
    # Runners
    "ScriptRunner",
    "ModuleRunner",
    # Analysis
    "ResultsAnalyzer",
    # Visualization
    "Plotter",
    "Exporter",
]
