# Spike Pipeline Module

End-to-end pipeline for spike detection and analysis.

## Components

| File | Purpose |
|------|---------|
| `pipeline.py` | Main pipeline orchestration |
| `detector.py` | Spike detection algorithms |
| `analyzer.py` | Spike analysis and metrics |
| `collector.py` | Data collection utilities |
| `processor.py` | Signal processing |

## Usage

```python
from insightspike.spike_pipeline import pipeline

# Run the full detection pipeline
results = pipeline.run(data)
```

## Pipeline Stages

1. **Collection** — Gather raw data
2. **Processing** — Filter and normalize signals
3. **Detection** — Identify spike events
4. **Analysis** — Compute metrics and classify
