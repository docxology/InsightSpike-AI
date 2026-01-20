# Docxology — InsightSpike-AI Sidecar Framework

[![Parent: InsightSpike-AI](https://img.shields.io/badge/parent-InsightSpike--AI-blue)](../)

A comprehensive sidecar framework for **orchestrating**, **testing**, **analyzing**, and **visualizing** all methods from the InsightSpike-AI package.

## Design Principles

1. **Real Methods Only** — No mocks, stubs, or fake implementations
2. **Modular Architecture** — Each component is independently testable
3. **Configurable Execution** — All parameters driven via TOML configuration
4. **Comprehensive Logging** — All operations logged for traceability
5. **Maximum Reuse** — Leverages existing `src/insightspike` and `scripts/` methods

---

## Quick Start

```bash
# From repository root
cd docxology

# Install in development mode
pip install -e .

# Run everything (tests, discovery, examples)
python run_all.py

# Quick smoke tests only
python run_all.py --quick

# Run tests only
python run_all.py --tests

# Run discovery only
python run_all.py --discovery

# Run examples only
python run_all.py --examples
```

---

## Directory Structure

```
docxology/
├── config/                 # Configuration system
│   ├── config.toml         # Main configuration
│   └── loader.py           # Config loading utilities
│
├── src/docxology/          # Core package
│   ├── discovery/          # Method discovery system
│   ├── orchestrator/       # Pipeline execution engine
│   ├── runners/            # Script and module runners
│   ├── analysis/           # Results analysis utilities
│   └── visualization/      # Plotting and export
│
├── tests/                  # Real functional tests
│   ├── test_smoke.py       # Smoke tests
│   ├── test_discovery.py   # Discovery tests
│   └── integration/        # End-to-end tests
│
├── output/                 # Generated outputs
│   ├── reports/
│   ├── figures/
│   └── logs/
│
└── examples/               # Usage examples
```

---

## Core Features

### 1. Method Discovery

Automatically scan `src/insightspike` to discover all public methods:

```python
from docxology.discovery import ModuleScanner

scanner = ModuleScanner()
methods = scanner.scan_package("insightspike")
print(f"Found {len(methods)} methods")
```

### 2. Orchestration

Execute multi-stage pipelines with dependency management:

```python
from docxology.orchestrator import Pipeline

pipeline = Pipeline("analysis_pipeline")
pipeline.add_stage("discovery", discover_methods)
pipeline.add_stage("execution", run_methods, depends_on=["discovery"])
pipeline.add_stage("analysis", analyze_results, depends_on=["execution"])
pipeline.run()
```

### 3. Script Execution

Run existing scripts from `/scripts` with configuration:

```python
from docxology.runners import ScriptRunner

runner = ScriptRunner()
result = runner.execute(
    script="run_fixed_mazes.py",
    env={"MAZE_SIZE": "15", "SEEDS": "10"}
)
```

### 4. Analysis & Visualization

Process results and generate visualizations:

```python
from docxology.analysis import ResultsAnalyzer
from docxology.visualization import Plotter, Exporter

analyzer = ResultsAnalyzer()
stats = analyzer.compute_summary(results)

plotter = Plotter()
plotter.performance_chart(stats, output="output/figures/performance.png")

exporter = Exporter()
exporter.to_json(stats, "output/reports/summary.json")
exporter.to_html(stats, "output/reports/summary.html")
```

---

## Configuration

Edit `config/config.toml` to customize behavior:

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

---

## Testing

All tests use **real functional methods** (Zero-Mock policy):

```bash
# Run all tests
pytest tests/ -v

# Smoke tests only (fast)
pytest tests/test_smoke.py -v

# Integration tests
pytest tests/integration/ -v

# With coverage
pytest tests/ --cov=docxology --cov-report=html
```

---

## Integration with Parent Package

Docxology directly imports and uses methods from `src/insightspike`:

```python
# Direct access to parent package
from insightspike.algorithms import gedig_calculator
from insightspike.core import MainAgent
from insightspike.config import get_config

# All executions use real implementations
agent = MainAgent()
result = agent.process_question("What patterns exist?")
```

---

## Output Formats

- **JSON** — Structured data with full metadata
- **CSV** — Tabular data for spreadsheet analysis
- **HTML** — Interactive reports with visualizations
- **Markdown** — Documentation-ready summaries

---

## See Also

- [AGENTS.md](./AGENTS.md) — AI assistant integration guide
- [Parent README](../README.md) — Main InsightSpike-AI documentation
- [Scripts](../scripts/) — Available execution scripts
- [Main Package](../src/insightspike/) — Core InsightSpike-AI modules
