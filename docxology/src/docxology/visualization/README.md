# Visualization Module

Plotting, export, and report generation.

## Classes

| Class | Purpose |
|-------|---------|
| `Plotter` | Basic matplotlib charts |
| `AdvancedPlotter` | Pie, radar, treemap, network, animations |
| `Exporter` | Export to JSON/CSV/HTML/Markdown |
| `Reporter` | Generate comprehensive reports |

## Usage

```python
from docxology.visualization import Plotter, Exporter, Reporter

# Create plots
plotter = Plotter(dpi=150)
plotter.performance_chart(stats, output="output/figures/perf.png")

# Export data
exporter = Exporter()
exporter.to_json(data, "output/reports/summary.json")
exporter.to_html(data, "output/reports/summary.html")

# Generate report
reporter = Reporter(output_dir="output/reports")
reporter.generate("analysis", results)
```

## Files

- `plotters.py` — `Plotter` class
- `advanced_plotters.py` — `AdvancedPlotter` with animations
- `exporters.py` — `Exporter` class
- `reporters.py` — `Reporter` class
