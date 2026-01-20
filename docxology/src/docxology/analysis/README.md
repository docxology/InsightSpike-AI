# Analysis Module

Results analysis and metrics collection for docxology.

## Classes

| Class | Purpose |
|-------|---------|
| `ResultsAnalyzer` | Analyze JSON/log outputs, compute statistics |
| `MetricsCollector` | Collect and aggregate metrics |
| `ResultsComparison` | Compare baseline vs experiment results |

## Usage

```python
from docxology.analysis import ResultsAnalyzer, MetricsCollector

# Analyze directory of results
analyzer = ResultsAnalyzer()
stats = analyzer.analyze_directory("output/results")

# Collect metrics
collector = MetricsCollector()
collector.record("latency", 1.5)
stats = collector.get_stats("latency")
```

## Files

- `results_analyzer.py` — `ResultsAnalyzer` class
- `metrics_collector.py` — `MetricsCollector` class
- `comparison.py` — `ResultsComparison` class
