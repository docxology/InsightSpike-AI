# Getting Started

Comprehensive guide for installing and running InsightSpike-AI.

## Requirements

### System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Python | 3.10+ | 3.11+ |
| RAM | 4 GB | 16 GB |
| Storage | 1 GB | 10 GB |
| GPU | Optional | CUDA-capable |

### Core Dependencies

| Package | Purpose | Required |
|---------|---------|----------|
| `torch` | Neural networks | Yes |
| `torch_geometric` | Graph neural networks | Yes |
| `numpy`, `scipy` | Numerical computing | Yes |
| `networkx` | Graph algorithms | Yes |
| `sentence-transformers` | Embeddings | Recommended |
| `pyyaml`, `toml` | Configuration | Yes |

---

## Installation

### Quick Install (Poetry)

```bash
# Clone repository
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI

# Install with Poetry
poetry install

# Activate environment
poetry shell
```

### Install with pip

```bash
# Clone repository
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install
pip install -e .
```

### Verify Installation

```bash
# Check version
python -c "import insightspike; print(insightspike.about())"
# Expected: {'name': 'InsightSpike-AI', 'version': '0.8.0', 'lite_mode': False}

# Run quick test
python -m insightspike discover examples/
```

---

## Configuration

### Step 1: API Keys

```bash
# Create .env file
cp .env.example .env

# Set API keys
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"
# or
export ANTHROPIC_API_KEY="sk-ant-xxxxxxxxxx"
```

### Step 2: Config File

```bash
# Copy example config
cp config_examples/openai_config.yaml config.yaml
```

Or create `config.toml`:

```toml
[llm]
provider = "openai"
model = "gpt-4"
temperature = 0.7

[graph]
spike_ged_threshold = -0.5
spike_ig_threshold = 0.2

[embedding]
dimension = 384
model = "all-MiniLM-L6-v2"
```

### Step 3: Environment Variables (Optional)

```bash
# LLM provider
export INSIGHTSPIKE_LLM_PROVIDER=openai

# Lite mode (skip heavy imports)
export INSIGHTSPIKE_LITE_MODE=1

# Disable GNN (for stability on some systems)
export INSIGHTSPIKE_DISABLE_GNN=1

# Use paper preset
export INSIGHTSPIKE_PRESET=paper
```

---

## Quick Start

### Python API

```python
from insightspike import create_agent, quick_demo

# Create agent
agent = create_agent()
agent.initialize()

# Process question
result = agent.process_question("What are the main concepts?")
print(result['response'])

# Or run quick demo
quick_demo()
```

### CLI

```bash
# Interactive chat
insightspike chat

# Single query
insightspike query "What is quantum computing?"

# Discovery mode
insightspike discover ./documents/

# With specific config
spike query --config config.yaml "AI question"

# With specific provider
spike query --llm-provider anthropic "Complex question"
```

---

## Verify Setup

### Run Tests

```bash
# Full test suite
pytest tests/ -v

# Quick smoke tests
pytest tests/test_smoke.py -v

# With coverage
pytest tests/ -v --cov=src/insightspike
```

### Run Discovery

```bash
# Using docxology
cd docxology
python run_all.py --quick

# Expected output:
# ✅ docxology-tests: 22 passed
# ✅ discovery: 1105 methods found
# 🎉 All components passed!
```

### Test Import

```python
# Test all key imports
from insightspike import (
    create_agent,
    L3GraphReasoner,
    get_config,
    MainAgent
)
print("All imports successful!")
```

---

## Troubleshooting

### ModuleNotFoundError: torch_geometric

```bash
pip install torch_geometric
```

### API key not found

```bash
# Check environment variable
echo $OPENAI_API_KEY

# Load from .env
source .env
```

### CUDA not available

```python
import torch
print(torch.cuda.is_available())  # Should be True for GPU
```

If False, install CUDA-enabled PyTorch:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

---

## Source Documents

| Document | Path |
|----------|------|
| Environment Setup | [getting-started/ENVIRONMENT_SETUP.md](../../docs/getting-started/ENVIRONMENT_SETUP.md) |
| Setup Guide | [getting-started/setup_guide.md](../../docs/getting-started/setup_guide.md) |
| Quick Test | [getting-started/quick_test_discover.md](../../docs/getting-started/quick_test_discover.md) |
| Quick Start | [QUICKSTART.md](../../docs/QUICKSTART.md) |
| Main README | [README.md](../../README.md) |
