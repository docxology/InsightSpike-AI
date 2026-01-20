# API Reference

Comprehensive API documentation for InsightSpike-AI.

## Quick Start

```python
from insightspike import create_agent, quick_demo

# Create configured agent (Defaults to Ollama/Ministral-3B)
agent = create_agent()

# Process a question
result = agent.process_question("What patterns exist in this data?")
print(result['response'])

# Run quick demo
quick_demo()
```

---

## Core Exports

### Package Level (`insightspike`)

```python
from insightspike import (
    # Core agent
    MainAgent,
    CycleResult,
    
    # Factory functions
    create_agent,
    quick_demo,
    
    # Configuration
    get_config,
    Config,
    About,
    about,
    
    # Layers
    ErrorMonitor,           # L1
    L2MemoryManager,        # L2
    L3GraphReasoner,        # L3
    get_llm_provider,       # L4
    
    # Agent system
    GenericInsightSpikeAgent,
    InsightSpikeAgentFactory,
    AgentConfigBuilder,
    create_maze_agent,
    create_configured_maze_agent,
    
    # Interfaces
    TaskType,
    EnvironmentInterface,
    InsightMoment,
    
    # Standalone tools
    StandaloneL3GraphReasoner,
    create_standalone_reasoner,
    analyze_documents_simple,
    
    # Metrics
    graph_metrics,
    eureka_spike,
)
```

---

## MainAgent

Primary agent orchestrating all 4 layers.

### Initialization

```python
from insightspike import MainAgent

agent = MainAgent(config=None)  # Uses default config if None
agent.initialize()
```

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `initialize()` | `bool` | Initialize all layers |
| `process_question(question, **kwargs)` | `dict` | Process a user question |
| `cleanup()` | `None` | Release resources |

### Process Question Response

```python
result = agent.process_question("What is quantum computing?")
# Result structure:
{
    "response": str,      # LLM response
    "success": bool,      # Processing succeeded
    "metrics": dict,      # Performance metrics
    "spike_detected": bool  # Eureka moment detected
}
```

---

## L3GraphReasoner

Graph-based reasoning with geDIG metrics.

### Initialization

```python
from insightspike import L3GraphReasoner

reasoner = L3GraphReasoner(config=None)
reasoner.initialize()
```

### Core Methods

```python
# Analyze documents
result = reasoner.analyze_documents(
    documents=[{"text": "...", "embedding": [...]}],
    context={"query_vector": [...]}
)

# Build graph from embeddings
graph = reasoner.build_graph(embeddings)  # numpy array or list of dicts

# Calculate metrics
ged = reasoner.calculate_ged(graph1, graph2)
ig = reasoner.calculate_ig(old_state, new_state)

# Detect eureka spike
spike = reasoner.detect_eureka_spike(delta_ged, delta_ig)

# Legacy compatibility
result = reasoner.analyze_graph(documents, query_vector=query)
```

### Analyze Documents Response

```python
{
    "graph": Data,           # PyTorch Geometric graph
    "metrics": {
        "delta_ged": float,
        "delta_ig": float,
        "delta_ged_norm": float,
        "delta_sp": float,
    },
    "conflicts": {
        "structural": float,
        "semantic": float,
        "temporal": float,
        "total": float,
    },
    "reward": {
        "base": float,
        "structure": float,
        "novelty": float,
        "total": float,
    },
    "spike_detected": bool,
    "reasoning_quality": float,
    "graph_context": dict,
}
```

---

## Metrics

### Simple Metrics

```python
from insightspike.metrics.graph_metrics import delta_ged, delta_ig

# Calculate graph edit distance
ged = delta_ged(graph1, graph2)

# Calculate information gain
ig = delta_ig(old_state, new_state)
```

### Advanced Metrics

```python
from insightspike.metrics.advanced_graph_metrics import (
    delta_ged as advanced_ged,
    delta_ig as advanced_ig
)

# With normalization and advanced algorithms
ged = advanced_ged(graph1, graph2, normalize=True)
```

### Metrics Selector

```python
from insightspike.algorithms.metrics_selector import MetricsSelector

selector = MetricsSelector(config)
info = selector.get_algorithm_info()
# {
#     'ged_algorithm': 'spectral',
#     'ig_algorithm': 'linkset_entropy',
#     'advanced_available': True,
#     'networkx_available': True
# }
```

---

## Graph Operations

### Message Passing

```python
from insightspike.graph.message_passing import MessagePassing

mp = MessagePassing(
    hidden_dim=64,
    num_heads=4,
    dropout=0.1
)
updated_representations = mp.forward(graph, query_vector)
```

### Edge Reevaluator

```python
from insightspike.graph.edge_reevaluator import EdgeReevaluator

reevaluator = EdgeReevaluator(config)
updated_edges = reevaluator.reevaluate(graph, context)
```

### Graph Construction

```python
from insightspike.graph.construction import GraphConstructor

constructor = GraphConstructor(config)
graph = constructor.build(documents, embeddings)
```

---

## Vector Index

```python
from insightspike.vector_index import factory

# Create index (auto-selects FAISS if available)
index = factory.create_index(dimension=768)

# Add vectors
index.add(vectors)  # numpy array (n, d)

# Search
distances, indices = index.search(query_vector, k=10)
```

---

## Standalone Reasoner

Independent reasoning without full agent:

```python
from insightspike.tools.standalone import (
    StandaloneL3GraphReasoner,
    create_standalone_reasoner,
    analyze_documents_simple
)

# Create reasoner
reasoner = create_standalone_reasoner()

# Simple analysis
results = analyze_documents_simple(
    documents=[{"text": "..."}],
    query="What patterns exist?"
)
```

---

## Configuration

```python
from insightspike.config import get_config, load_config

# Get default config
config = get_config()

# Load from file
config = load_config("config.yaml")

# Access settings
print(config.llm.provider)
print(config.graph.spike_ged_threshold)
print(config.embedding.dimension)
```

### Key Config Sections

| Section | Parameters |
|---------|------------|
| `llm` | `provider`, `model`, `temperature`, `max_tokens` |
| `graph` | `spike_ged_threshold`, `spike_ig_threshold`, `lambda_weight`, `sp_beta` |
| `embedding` | `dimension`, `model` |
| `memory` | `max_episodes`, `consolidation_threshold` |

---

## Providers

```python
from insightspike.providers import ProviderFactory

from insightspike.providers import ProviderFactory

# Create Ollama provider (Default)
provider = ProviderFactory.create("ollama", {
    "model": "ministral-3:3b",
    "api_base": "http://localhost:11434/v1",
    "api_key": "ollama"
})

# Create OpenAI provider
provider = ProviderFactory.create("openai", {
    "api_key": "sk-...",
    "model": "gpt-4",
    "temperature": 0.7
})

# Generate response
response = provider.generate(prompt, system_prompt=None)
```

### Provider Interface

```python
class LLMProvider:
    def generate(self, prompt: str, **kwargs) -> str: ...
    def generate_stream(self, prompt: str, **kwargs) -> Iterator[str]: ...
    def embed(self, texts: List[str]) -> np.ndarray: ...
```

---

## Implementation Source Files

| Category | File | Key Class |
|----------|------|----------|
| **Providers** | [provider_factory.py](../../src/insightspike/providers/provider_factory.py) | `ProviderFactory` |
| **Providers** | [openai_provider.py](../../src/insightspike/providers/openai_provider.py) | `OpenAIProvider` |
| **Providers** | [anthropic_provider.py](../../src/insightspike/providers/anthropic_provider.py) | `AnthropicProvider` |
| **Config** | [models.py](../../src/insightspike/config/models.py) | `LLMConfig`, `InsightSpikeConfig` |
| **Config** | [presets.py](../../src/insightspike/config/presets.py) | `ConfigPresets` |
| **Agents** | [main_agent.py](../../src/insightspike/implementations/agents/main_agent.py) | `MainAgent` |
| **Layers** | [layer3_graph_reasoner.py](../../src/insightspike/implementations/layers/layer3_graph_reasoner.py) | `L3GraphReasoner` |

---

## Source Documents

| Document | Path |
|----------|------|
| API Summary | [api-reference/CORRECT_API_SUMMARY.md](../../docs/api-reference/CORRECT_API_SUMMARY.md) |
| Detailed Docs | [api-reference/DETAILED_DOCUMENTATION.md](../../docs/api-reference/DETAILED_DOCUMENTATION.md) |
| Quick Start | [api-reference/quick_start.md](../../docs/api-reference/quick_start.md) |
| Public API | [api-reference/public_api.md](../../docs/api-reference/public_api.md) |
| Graph SP Engine | [api-reference/graph_sp_engine.md](../../docs/api-reference/graph_sp_engine.md) |
| Package Init | [src/insightspike/__init__.py](../../src/insightspike/__init__.py) |
