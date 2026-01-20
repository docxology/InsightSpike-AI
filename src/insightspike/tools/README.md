# Tools Module

Standalone tools and experimental utilities.

## Submodules

| Submodule | Purpose |
|-----------|---------|
| `standalone/` | Standalone L3 reasoner tools |
| `experiments/` | Experimental utilities |
| `experiments.py` | Experiment runner |

## Standalone Reasoner

```python
from insightspike.tools.standalone import (
    StandaloneL3GraphReasoner,
    create_standalone_reasoner,
    analyze_documents_simple
)

reasoner = create_standalone_reasoner()
results = analyze_documents_simple(documents)
```
