# Configuration System

Central configuration system for docxology using TOML format.

## Files

| File | Purpose |
|------|---------|
| `config.toml` | Main configuration file |
| `loader.py` | Configuration loading utilities |
| `__init__.py` | Module exports |

## Configuration Sections

```toml
[discovery]
packages = ["insightspike"]
exclude = ["deprecated", "experimental"]

[execution]
parallel = true
max_workers = 4
timeout = 300

[logging]
level = "INFO"
output_dir = "output/logs"

[visualization]
style = "seaborn"
dpi = 150
format = "png"
```

## Usage

```python
from config import load_config, get_section

config = load_config()
discovery_config = get_section("discovery")
```

## See Also

- [Root README](../README.md) — Main documentation
- [Root AGENTS.md](../AGENTS.md) — AI guidelines
