# User Guide

Comprehensive summary of user documentation including CLI, configuration, and LLM providers.

## CLI Commands

### Basic Commands

```bash
# Discovery mode - analyze files for patterns
insightspike discover <path>

# Interactive chat mode
insightspike chat

# Single query mode
insightspike query "What are the main concepts?"

# Analysis mode
insightspike analyze <documents>

# Help
insightspike --help
```

### Spike Commands

```bash
# Query with spike detection
spike query "量子コンピューティングとは何ですか？"

# Use specific config
spike query --config config_openai.yaml "AI question"

# Specify LLM provider directly
spike query --llm-provider anthropic "What is consciousness?"
```

See: [cli_commands.md](../../docs/user-guide/cli_commands.md) | [spike_commands_summary.md](../../docs/user-guide/spike_commands_summary.md)

---

## LLM Providers

### Supported Providers

| Provider | Models | Use Case |
|----------|--------|----------|
| **Ollama** | **Ministral-3B**, Llama 2 | **Primary (Zero Mock)** |
| **OpenAI** | GPT-4, GPT-3.5-Turbo | General purpose |
| **Anthropic** | Claude 3 Opus, Claude 3 Sonnet | Complex reasoning |
| **Mock** | — | Testing (no API) |

### OpenAI Configuration

```yaml
llm:
  provider: openai
  model: gpt-4           # or gpt-3.5-turbo, gpt-4-turbo-preview
  temperature: 0.7       # 0.0-2.0 (creativity)
  max_tokens: 1000       # max output tokens
  top_p: 0.9             # nucleus sampling
  timeout: 30            # timeout (seconds)
```

**Models:**
- `gpt-3.5-turbo` — Fast and economical
- `gpt-3.5-turbo-16k` — Long context support
- `gpt-4` — Highest performance
- `gpt-4-turbo-preview` — GPT-4 optimized for speed

### Anthropic Configuration

```yaml
llm:
  provider: anthropic
  model: claude-3-opus-20240229  # highest performance
  temperature: 0.7
  max_tokens: 1000
```

**Models:**
- `claude-3-sonnet-20240229` — Balanced
- `claude-3-opus-20240229` — Highest performance

### Local LLM (Ollama) - Default

```yaml
llm:
  provider: ollama
  model: ministral-3:3b
  base_url: http://localhost:11434/v1
  api_key: ollama
```

### Environment Variables

```bash
# API Keys
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"
export ANTHROPIC_API_KEY="sk-ant-xxxxxxxxxx"

# Provider selection
export INSIGHTSPIKE_LLM_PROVIDER=openai

# Lite mode (skip heavy imports)
export INSIGHTSPIKE_LITE_MODE=1
```

### Python API

```python
from insightspike.providers import ProviderFactory

# Create provider directly
provider = ProviderFactory.create("openai", {
    "api_key": "your-key",
    "model": "gpt-4",
    "temperature": 0.5
})

# Generate response
response = provider.generate("What is quantum entanglement?")
```

### Dynamic Provider Switching

```python
from insightspike.config import load_config
from insightspike.implementations.agents.datastore_agent import DataStoreMainAgent

# Simple questions with GPT-3.5
simple_agent = DataStoreMainAgent(datastore, load_config("config_gpt35.yaml"))
simple_result = simple_agent.process("What is the weather?")

# Complex reasoning with GPT-4
advanced_agent = DataStoreMainAgent(datastore, load_config("config_gpt4.yaml"))
complex_result = advanced_agent.process("Explain quantum entanglement and consciousness")
```

### Custom System Prompt

```yaml
llm:
  provider: openai
  model: gpt-4
  system_prompt: |
    You are an expert in quantum physics.
    Provide scientifically accurate and clear explanations.
```

---

## Configuration

### Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `INSIGHTSPIKE_LLM_PROVIDER` | LLM provider | `openai`, `anthropic`, `ollama` |
| `OPENAI_API_KEY` | OpenAI API key | `sk-...` |
| `ANTHROPIC_API_KEY` | Anthropic API key | `sk-ant-...` |
| `INSIGHTSPIKE_LITE_MODE` | Minimal import mode | `1` |
| `INSIGHTSPIKE_DISABLE_GNN` | Disable GNN | `1` |
| `INSIGHTSPIKE_PRESET` | Apply preset | `paper` |

### Config File (TOML/YAML)

```toml
# config.toml
[llm]
provider = "openai"
model = "gpt-4"
temperature = 0.7
max_tokens = 1000

[graph]
spike_ged_threshold = -0.5
spike_ig_threshold = 0.2
lambda_weight = 1.0
sp_beta = 1.0

[embedding]
dimension = 384
model = "all-MiniLM-L6-v2"
```

### Pricing Reference (2024)

| Provider | Model | Input | Output |
|----------|-------|-------|--------|
| OpenAI | GPT-3.5-turbo | $0.0005/1K | $0.0015/1K |
| OpenAI | GPT-4 | $0.03/1K | $0.06/1K |
| Anthropic | Claude 3 Sonnet | $0.003/1K | $0.015/1K |
| Anthropic | Claude 3 Opus | $0.015/1K | $0.075/1K |

---

## Best Practices

### API Key Security
- **Never** commit API keys to Git
- Use environment variables or `.env` files
- Rotate keys regularly

### Cost Management
- Use `gpt-3.5-turbo` or `claude-3-sonnet` during development
- Set appropriate `max_tokens` limits
- Monitor usage via provider dashboards

### Performance
- Cache frequent queries
- Use batch processing for efficiency
- Set appropriate timeouts

---

## Source Documents

| Document | Path |
|----------|------|
| CLI Commands | [user-guide/cli_commands.md](../../docs/user-guide/cli_commands.md) |
| Configuration | [user-guide/configuration_guide.md](../../docs/user-guide/configuration_guide.md) |
| LLM Providers | [user-guide/llm_providers_guide.md](../../docs/user-guide/llm_providers_guide.md) |
| Spike Commands | [user-guide/spike_commands_summary.md](../../docs/user-guide/spike_commands_summary.md) |
| July 2024 Features | [user-guide/july_2024_features_quickstart.md](../../docs/user-guide/july_2024_features_quickstart.md) |
| User Guide (JA) | [USER_GUIDE_JA.md](../../docs/USER_GUIDE_JA.md) |
