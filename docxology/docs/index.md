# Docxology Documentation Hub

Comprehensive English documentation for InsightSpike-AI with complete signposting.

## Quick Navigation

| Document | Description | Key Topics |
|----------|-------------|------------|
| [System Overview](./system_overview.md) | **Complete system summary** | All components linked |
| [Source Inventory](./source_inventory.md) | **Source file index** | Direct file links |
| [Getting Started](./getting_started.md) | Installation and setup | Install, config, verify |
| [Architecture](./architecture.md) | System design | 4-layer design, components |
| [API Reference](./api_reference.md) | API documentation | Classes, methods, responses |
| [User Guide](./user_guide.md) | CLI and configuration | LLM providers, config |
| [Concepts](./concepts.md) | Core theory | geDIG equations, metrics |
| [Experiments](./experiments.md) | Research validation | Maze, HotPotQA, analogies |
| [Design](./design.md) | Design documents | Implementation plans |
| [Development](./development.md) | Contributing | Testing, standards |

---

## Key Equations

### geDIG Gauge (Paper v4)

```
F = ΔEPC_norm - λ ( ΔH_norm + γ · ΔSP_rel )
```

### Core Insight Equation

```
T* = argmin_T GED(T(G₁), G₂)
```

> *Find the minimal transformation that makes two knowledge structures isomorphic. That transformation IS the insight.*

See: [Concepts](./concepts.md) for full equation details.

---

## Supported LLM Providers

| Provider | Models | Setup |
|----------|--------|-------|
| **Ollama** | **Ministral-3B (Default)**, Llama 2 | **Required** (Zero Mock) |
| **OpenAI** | GPT-4, GPT-3.5-Turbo | `OPENAI_API_KEY` |
| **Anthropic** | Claude 3 Opus/Sonnet | `ANTHROPIC_API_KEY` |

See: [User Guide](./user_guide.md) for full configuration.

---

## Parent Documentation Map

All documentation is in `/docs/` ([view](../../docs/README.md)):

```
docs/                           # 103+ markdown files
├── README.md                   # Documentation hub
├── index.md                    # Landing page (Japanese)
├── gedig_spec.md               # geDIG v4 specification ⭐
├── glossary.md                 # Term definitions
│
├── getting-started/            # Setup guides
│   ├── ENVIRONMENT_SETUP.md
│   ├── setup_guide.md
│   └── quick_test_discover.md
│
├── user-guide/                 # User documentation
│   ├── cli_commands.md
│   ├── configuration_guide.md
│   └── llm_providers_guide.md  ⭐
│
├── api-reference/              # API documentation
│   ├── CORRECT_API_SUMMARY.md
│   ├── DETAILED_DOCUMENTATION.md
│   └── public_api.md
│
├── architecture/               # System architecture
│   ├── layer_architecture.md   ⭐
│   ├── agent_types.md
│   ├── configuration.md
│   └── ... (21 files)
│
├── concepts/                   # Core concepts
│   ├── gedig_in_5_minutes.md   ⭐
│   ├── intuition.md
│   └── universal_principle_hypothesis.md
│
├── design/                     # Design documents
│   ├── level3_isomorphism_discovery.md
│   └── hotpotqa_case_studies.md
│
├── experiments/                # Experiment results
│   └── structural_similarity_results.md
│
└── paper/                      # Research papers
    └── geDIG_onegauge_improved_v6.pdf
```

---

## Quick Reference

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI API key |
| `ANTHROPIC_API_KEY` | Anthropic API key |
| `INSIGHTSPIKE_LLM_PROVIDER` | `ollama` (default), `openai`, `anthropic` |
| `INSIGHTSPIKE_LITE_MODE=1` | Skip heavy imports |
| `INSIGHTSPIKE_PRESET=paper` | Use paper parameters |

### Key Classes

| Class | Purpose |
|-------|---------|
| `MainAgent` | Full 4-layer agent |
| `L3GraphReasoner` | Graph reasoning |
| `create_agent()` | Factory function |
| `get_config()` | Configuration |

### Key Metrics

| Metric | Formula |
|--------|---------|
| ΔGED | `GED(G_new, G_old)` |
| ΔIG | `H(G_old) - H(G_new)` |
| Gauge | `F = ΔEPC - λ(ΔH + γ·ΔSP)` |

---

## See Also

- [docxology README](../README.md) — Main docxology documentation
- [InsightSpike README](../../README.md) — Project overview
- [Full docs](../../docs/README.md) — Complete documentation (103+ files)
- [geDIG Spec](../../docs/gedig_spec.md) — Canonical gauge definition
