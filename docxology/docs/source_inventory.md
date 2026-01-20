# Source File Inventory

Complete inventory of key source files in InsightSpike-AI with descriptions and links.

---

## Core Package (`src/insightspike/`)

### Entry Points

| File | Purpose | Key Exports |
|------|---------|-------------|
| [__init__.py](../../src/insightspike/__init__.py) | Package initialization | `create_agent`, `MainAgent`, `L3GraphReasoner` |
| [about.py](../../src/insightspike/about.py) | Version information | `about()`, `About` |

---

### Agents (`implementations/agents/`)

| File | Class | Description |
|------|-------|-------------|
| [main_agent.py](../../src/insightspike/implementations/agents/main_agent.py) | `MainAgent` | Primary 4-layer orchestrator |
| [generic_agent.py](../../src/insightspike/implementations/agents/generic_agent.py) | `GenericInsightSpikeAgent` | Configurable agent base |
| [datastore_agent.py](../../src/insightspike/implementations/agents/datastore_agent.py) | `DataStoreMainAgent` | SQLite-backed agent |
| [agent_factory.py](../../src/insightspike/implementations/agents/agent_factory.py) | `InsightSpikeAgentFactory` | Agent creation factory |

---

### Layer Implementations (`implementations/layers/`)

| Layer | File | Key Class |
|-------|------|-----------|
| **L4: LLM** | [layer4_llm_interface.py](../../src/insightspike/implementations/layers/layer4_llm_interface.py) | `L4LLMInterface` |
| **L3: Graph** | [layer3_graph_reasoner.py](../../src/insightspike/implementations/layers/layer3_graph_reasoner.py) | `L3GraphReasoner` |
| **L2: Memory** | [layer2_memory_manager.py](../../src/insightspike/implementations/layers/layer2_memory_manager.py) | `L2MemoryManager` |
| **L1: Error** | [layer1_error_monitor.py](../../src/insightspike/implementations/layers/layer1_error_monitor.py) | `ErrorMonitor` |

---

### LLM Providers (`providers/`)

| File | Provider | Compatible With |
|------|----------|-----------------|
| [provider_factory.py](../../src/insightspike/providers/provider_factory.py) | `ProviderFactory` | All providers |
| [openai_provider.py](../../src/insightspike/providers/openai_provider.py) | `OpenAIProvider` | OpenAI, **Ollama** |
| [anthropic_provider.py](../../src/insightspike/providers/anthropic_provider.py) | `AnthropicProvider` | Anthropic Claude |
| [local_provider.py](../../src/insightspike/providers/local_provider.py) | `LocalProvider` | HuggingFace |

---

### Configuration (`config/`)

| File | Purpose | Key Classes |
|------|---------|-------------|
| [models.py](../../src/insightspike/config/models.py) | Pydantic config models | `LLMConfig`, `InsightSpikeConfig` |
| [presets.py](../../src/insightspike/config/presets.py) | Environment presets | `ConfigPresets` |
| [loader.py](../../src/insightspike/config/loader.py) | Config file loading | `load_config`, `get_config` |

---

### Algorithms (`algorithms/`)

| File | Algorithm | Formula |
|------|-----------|---------|
| [gedig_core.py](../../src/insightspike/algorithms/gedig_core.py) | geDIG gauge | `F = ΔEPC - λ(ΔH + γ·ΔSP)` |
| [information_gain.py](../../src/insightspike/algorithms/information_gain.py) | Entropy calculation | `ΔIG = H(old) - H(new)` |
| [graph_sp_engine.py](../../src/insightspike/algorithms/graph_sp_engine.py) | Shortest path | `ΔSP_rel` |
| [metrics_selector.py](../../src/insightspike/algorithms/metrics_selector.py) | Algorithm dispatch | Auto-selection |

---

### Graph Operations (`graph/`)

| File | Component | Function |
|------|-----------|----------|
| [construction.py](../../src/insightspike/graph/construction.py) | Graph builder | Build similarity graphs |
| [message_passing.py](../../src/insightspike/graph/message_passing.py) | GNN propagation | Message passing layers |
| [edge_reevaluator.py](../../src/insightspike/graph/edge_reevaluator.py) | Dynamic edges | Edge weight updates |

---

### Metrics (`metrics/`)

| File | Metrics |
|------|---------|
| [graph_metrics.py](../../src/insightspike/metrics/graph_metrics.py) | `delta_ged`, `delta_ig` |
| [advanced_graph_metrics.py](../../src/insightspike/metrics/advanced_graph_metrics.py) | Normalized variants |

---

## Docxology Framework (`docxology/`)

### Tests

| File | Coverage |
|------|----------|
| [test_smoke.py](../../docxology/tests/test_smoke.py) | Core imports, discovery |
| [test_llm_providers.py](../../docxology/tests/test_llm_providers.py) | Provider initialization |
| [test_ollama_integration.py](../../docxology/tests/test_ollama_integration.py) | Ollama/Ministral |

### Scripts

| Script | Purpose |
|--------|---------|
| [verify_defaults.py](../../docxology/scripts/verify_defaults.py) | Verify global config |
| [verify_ministral.py](../../docxology/scripts/verify_ministral.py) | Verify Ministral model |

### Examples

| Example | Description |
|---------|-------------|
| [run_all_methods.py](../../docxology/examples/run_all_methods.py) | Method discovery |
| [extended_analysis.py](../../docxology/examples/extended_analysis.py) | Deep analysis |
| [generate_all_visualizations.py](../../docxology/examples/generate_all_visualizations.py) | Visualization suite |

---

## Configuration Files

| File | Format | Purpose |
|------|--------|---------|
| [config.yaml](../../config.yaml) | YAML | Runtime configuration |
| [pyproject.toml](../../pyproject.toml) | TOML | Project metadata |
| [docxology/pyproject.toml](../../docxology/pyproject.toml) | TOML | Docxology config |

---

## Documentation (`docs/`)

| Category | Key Files |
|----------|-----------|
| **Architecture** | [layer_architecture.md](../../docs/architecture/layer_architecture.md), [agent_types.md](../../docs/architecture/agent_types.md) |
| **Concepts** | [gedig_spec.md](../../docs/gedig_spec.md), [gedig_in_5_minutes.md](../../docs/concepts/gedig_in_5_minutes.md) |
| **API** | [CORRECT_API_SUMMARY.md](../../docs/api-reference/CORRECT_API_SUMMARY.md), [public_api.md](../../docs/api-reference/public_api.md) |
| **User Guide** | [llm_providers_guide.md](../../docs/user-guide/llm_providers_guide.md), [configuration_guide.md](../../docs/user-guide/configuration_guide.md) |

---

*Last Updated: January 2026*
