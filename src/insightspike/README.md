# InsightSpike Package

Brain-inspired multi-agent architecture for Active Inference, knowledge restructuring, and insight detection.

## Package Version

**v0.8.0** with lazy loading and lite mode support.

## Directory Structure

| Module | Purpose |
|--------|---------|
| [core/](./core/) | Base classes, generic interfaces |
| [implementations/](./implementations/) | Agent layers (L1-L4), factories |
| [algorithms/](./algorithms/) | geDIG, graph calculations |
| [config/](./config/) | Configuration system |
| [cli/](./cli/) | Command-line interface |
| [providers/](./providers/) | LLM provider integrations |
| [graph/](./graph/) | Graph operations |
| [detection/](./detection/) | Eureka spike detection |
| [metrics/](./metrics/) | Graph metrics |
| [visualization/](./visualization/) | Plotting |
| [utils/](./utils/) | Utility functions |

## Quick Start

```python
from insightspike import create_agent, quick_demo

# Create an agent
agent = create_agent()

# Run quick demo
quick_demo()
```

## Lite Mode

Set `INSIGHTSPIKE_LITE_MODE=1` for fast, minimal imports.

## See Also

- [Parent README](../../README.md)
- [docxology](../../docxology/) — Testing framework
