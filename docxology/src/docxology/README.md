# Docxology Core Package

The main docxology Python package for orchestrating, testing, analyzing, and visualizing InsightSpike-AI methods.

## Modules

| Module | Purpose |
|--------|---------|
| [analysis/](./analysis/) | Results analysis and metrics |
| [discovery/](./discovery/) | Method discovery system |
| [orchestrator/](./orchestrator/) | Pipeline execution engine |
| [runners/](./runners/) | Script and module runners |
| [visualization/](./visualization/) | Plotting and export |

## Entry Points

- `cli.py` — CLI commands (`docx discover`, `docx run`, etc.)
- `__init__.py` — Package version and public exports

## Usage

```python
from docxology.discovery import ModuleScanner
from docxology.runners import ScriptRunner
from docxology.orchestrator import Pipeline
from docxology.analysis import ResultsAnalyzer
from docxology.visualization import Plotter, Exporter
```

## See Also

- [Root README](../../README.md)
- [Root AGENTS.md](../../AGENTS.md)
