"""Pytest configuration for docxology tests."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add paths for imports
DOCXOLOGY_ROOT = Path(__file__).parent.parent
REPO_ROOT = DOCXOLOGY_ROOT.parent
SRC_PATH = DOCXOLOGY_ROOT / "src"

sys.path.insert(0, str(SRC_PATH))
sys.path.insert(0, str(REPO_ROOT / "src"))


@pytest.fixture
def docxology_root() -> Path:
    """Get docxology root directory."""
    return DOCXOLOGY_ROOT


@pytest.fixture
def repo_root() -> Path:
    """Get repository root directory."""
    return REPO_ROOT


@pytest.fixture
def scripts_dir() -> Path:
    """Get scripts directory."""
    return REPO_ROOT / "scripts"


@pytest.fixture
def sample_results() -> dict:
    """Sample results data for testing."""
    return {
        "success": True,
        "duration": 1.5,
        "metrics": {
            "accuracy": 0.95,
            "precision": 0.92,
            "recall": 0.88,
        },
        "config": {
            "maze_size": 15,
            "max_steps": 250,
        },
    }


@pytest.fixture
def sample_methods() -> list[dict]:
    """Sample method data for testing."""
    return [
        {
            "name": "compute_gedig",
            "module": "insightspike.algorithms.gedig_calculator",
            "type": "function",
            "signature": "(graph1, graph2, lambda_param=0.5)",
        },
        {
            "name": "MainAgent",
            "module": "insightspike.implementations.agents.main_agent",
            "type": "class",
            "signature": "(config=None)",
        },
    ]
