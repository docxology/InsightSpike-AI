"""Configuration loader for docxology.

Provides centralized configuration management with TOML support.
All parameters are loaded from config/config.toml and can be overridden via environment.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import toml

logger = logging.getLogger(__name__)

# Cache for loaded configuration
_config_cache: dict[str, Any] | None = None

# Default config path relative to this file
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.toml"


def get_config_path() -> Path:
    """Get the configuration file path.
    
    Checks in order:
    1. DOCXOLOGY_CONFIG environment variable
    2. Default config.toml in config/ directory
    
    Returns:
        Path to configuration file.
    """
    env_path = os.environ.get("DOCXOLOGY_CONFIG")
    if env_path:
        return Path(env_path)
    return DEFAULT_CONFIG_PATH


def load_config(path: Path | str | None = None, reload: bool = False) -> dict[str, Any]:
    """Load configuration from TOML file.
    
    Args:
        path: Optional path to config file. If None, uses default.
        reload: If True, force reload from disk (ignore cache).
        
    Returns:
        Configuration dictionary.
        
    Raises:
        FileNotFoundError: If config file does not exist.
        toml.TomlDecodeError: If config file is invalid TOML.
    """
    global _config_cache
    
    if _config_cache is not None and not reload and path is None:
        return _config_cache
    
    config_path = Path(path) if path else get_config_path()
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return get_defaults()
    
    logger.debug(f"Loading configuration from: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = toml.load(f)
    
    # Apply environment overrides
    config = _apply_env_overrides(config)
    
    # Cache if loading from default path
    if path is None:
        _config_cache = config
    
    return config


def get_defaults() -> dict[str, Any]:
    """Get default configuration values.
    
    Returns:
        Dictionary of default configuration values.
    """
    return {
        "project": {
            "name": "docxology",
            "version": "0.1.0",
        },
        "discovery": {
            "packages": ["insightspike"],
            "exclude": ["deprecated", "experimental", "__pycache__", "tests"],
            "include_private": False,
            "max_depth": 10,
        },
        "execution": {
            "parallel": True,
            "max_workers": 4,
            "timeout": 300,
            "retry_count": 0,
            "working_dir": "..",
        },
        "runners": {
            "scripts_dir": "scripts",
            "python": "python",
            "env": {
                "INSIGHTSPIKE_LITE_MODE": "0",
                "PYTHONPATH": "src",
            },
        },
        "analysis": {
            "output_dir": "output/analysis",
            "metrics": ["mean", "std", "min", "max", "count"],
            "comparison_method": "relative",
        },
        "visualization": {
            "style": "seaborn-v0_8-whitegrid",
            "dpi": 150,
            "format": "png",
            "figsize": [10, 6],
            "palette": "viridis",
            "figures_dir": "output/figures",
        },
        "export": {
            "formats": ["json", "csv", "html", "markdown"],
            "reports_dir": "output/reports",
            "timestamp_files": True,
            "json_indent": 2,
        },
        "logging": {
            "level": "INFO",
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            "output_dir": "output/logs",
            "file_enabled": True,
            "filename": "docxology.log",
            "console_enabled": True,
        },
        "testing": {
            "timeout": 60,
            "coverage_threshold": 50,
            "default_markers": [],
            "output_dir": "output/tests",
        },
    }


def _apply_env_overrides(config: dict[str, Any]) -> dict[str, Any]:
    """Apply environment variable overrides to config.
    
    Environment variables are prefixed with DOCXOLOGY_ and use double underscores
    for nesting. For example:
        DOCXOLOGY_LOGGING__LEVEL=DEBUG -> config["logging"]["level"] = "DEBUG"
    
    Args:
        config: Base configuration dictionary.
        
    Returns:
        Configuration with environment overrides applied.
    """
    prefix = "DOCXOLOGY_"
    
    for key, value in os.environ.items():
        if not key.startswith(prefix):
            continue
        
        # Remove prefix and split by double underscore
        config_key = key[len(prefix):].lower()
        parts = config_key.split("__")
        
        # Navigate to nested dict and set value
        current = config
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        # Convert value types
        final_key = parts[-1]
        if value.lower() in ("true", "false"):
            current[final_key] = value.lower() == "true"
        elif value.isdigit():
            current[final_key] = int(value)
        else:
            try:
                current[final_key] = float(value)
            except ValueError:
                current[final_key] = value
    
    return config


def get_section(section: str) -> dict[str, Any]:
    """Get a specific configuration section.
    
    Args:
        section: Section name (e.g., "logging", "execution").
        
    Returns:
        Configuration section dictionary.
    """
    config = load_config()
    return config.get(section, {})


def get_value(key: str, default: Any = None) -> Any:
    """Get a configuration value using dot notation.
    
    Args:
        key: Configuration key in dot notation (e.g., "logging.level").
        default: Default value if key not found.
        
    Returns:
        Configuration value.
    """
    config = load_config()
    parts = key.split(".")
    
    current = config
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return default
    
    return current


# Module-level convenience functions
def get_project_root() -> Path:
    """Get the docxology project root directory."""
    return Path(__file__).parent.parent


def get_repo_root() -> Path:
    """Get the InsightSpike-AI repository root directory."""
    return get_project_root().parent


def get_output_dir() -> Path:
    """Get the output directory, creating if needed."""
    output_dir = get_project_root() / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir
