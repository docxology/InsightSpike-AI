# Orchestrator Module

Pipeline execution engine with dependency management.

## Classes

| Class | Purpose |
|-------|---------|
| `Pipeline` | Multi-stage pipeline with dependency ordering |
| `Stage` | Individual pipeline stage |

## Usage

```python
from docxology.orchestrator import Pipeline

pipeline = Pipeline("analysis")
pipeline.add_stage("discover", discover_methods)
pipeline.add_stage("execute", run_methods, depends_on=["discover"])
pipeline.add_stage("analyze", analyze_results, depends_on=["execute"])
result = pipeline.run()
```

## Files

- `pipeline.py` — `Pipeline`, `Stage` classes
