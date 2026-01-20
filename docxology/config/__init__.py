"""Configuration package for docxology."""

from .loader import (
    get_config_path,
    get_defaults,
    get_output_dir,
    get_project_root,
    get_repo_root,
    get_section,
    get_value,
    load_config,
)

__all__ = [
    "load_config",
    "get_config_path",
    "get_defaults",
    "get_section",
    "get_value",
    "get_project_root",
    "get_repo_root",
    "get_output_dir",
]
