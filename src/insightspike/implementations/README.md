# Implementations Module

Concrete agent implementations with 4-layer architecture.

## Architecture

```
Layer 1: ErrorMonitor — Error detection and handling
Layer 2: MemoryManager — Episodic and semantic memory
Layer 3: GraphReasoner — Graph neural network reasoning
Layer 4: LLMInterface — Language model integration
```

## Submodules

| Module | Purpose |
|--------|---------|
| `agents/` | MainAgent, GenericAgent, factories |
| `layers/` | L1-L4 layer implementations |
| `memory/` | Memory systems |
| `graph/` | Graph reasoning |

## Key Classes

- `MainAgent` — Primary agent orchestrator
- `GenericInsightSpikeAgent` — Generic agent base
- `InsightSpikeAgentFactory` — Agent creation factory
- `AgentConfigBuilder` — Configuration builder
