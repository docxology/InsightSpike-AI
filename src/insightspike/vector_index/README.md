# Vector Index Module

Vector similarity search with multiple backends.

## Components

| File | Purpose |
|------|---------|
| `interface.py` | Abstract index interface |
| `factory.py` | Index factory for backend selection |
| `numpy_index.py` | NumPy-based fallback index |

## Usage

```python
from insightspike.vector_index import factory

# Create index (auto-selects FAISS if available, else NumPy)
index = factory.create_index(dimension=768)

# Add vectors
index.add(vectors)

# Search
results = index.search(query_vector, k=10)
```

## Backends

- **FAISS** — High-performance (if installed)
- **NumPy** — Fallback implementation
