# Experiments

Comprehensive summary of research experiments and validation.

## Overview

InsightSpike experiments validate geDIG across multiple domains:

| Experiment | Description | Results |
|------------|-------------|---------|
| Maze Navigation | POMDP Active Inference | ~91% greedy success |
| HotPotQA | Cross-domain QA | +60% F1 improvement |
| Structural Similarity | Analogy detection | 3 historical discoveries |
| Isomorphism Discovery | Level-3 insight | Novel analogies found |

---

## Maze Navigation Experiment

### Description

POMDP-based maze environment testing Active Inference decision-making.

### Setup

```bash
cd experiments/maze
python run_experiment.py
```

### Results

| Agent Type | Success Rate | Notes |
|------------|--------------|-------|
| Greedy | ~91% | Deterministic policy |
| Stochastic (τ=0.3) | 3.3% ± 1.0% | Temperature-scaled |
| Random | ~12.5% (1/8) | Baseline |

### Key Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Grid size | 4x4 | State space = 16 |
| Horizon | 4-10 | Planning steps |
| Temperature τ | 0.1-1.0 | Policy stochasticity |

### Metrics

```python
# From experiment output
{
    "success_rate": 0.91,
    "mean_steps": 3.2,
    "planning_horizon": 4,
    "temperature": 0.0  # greedy
}
```

See: [MAZE_NAV_SPEC.md](../../docs/MAZE_NAV_SPEC.md) | [HOWTO_maze_metrics.md](../../docs/HOWTO_maze_metrics.md)

---

## HotPotQA Experiments

### Description

Cross-domain question answering with structural similarity enhancement.

### Results

| Metric | Baseline | With geDIG | Improvement |
|--------|----------|------------|-------------|
| F1 Score | 0.06 | 0.66 | **+60%** |
| Real-world scenarios | 0/7 | 7/7 | **100%** |

### Case Studies

| Case | Domain Cross | Result |
|------|--------------|--------|
| Case 1 | Physics → Chemistry | ✅ Correct |
| Case 2 | Biology → Computer Science | ✅ Correct |
| Case 3 | Economics → Ecology | ✅ Correct |

### Error Analysis

Common failure patterns identified:
- Insufficient structural information
- Domain vocabulary mismatch
- Overly abstract analogies

See: [design/hotpotqa_case_studies.md](../../docs/design/hotpotqa_case_studies.md)

---

## Structural Similarity Validation

### Historical Scientific Discoveries

geDIG successfully detected structural analogies behind major scientific insights:

| Discovery | Year | Analogy | Similarity Score |
|-----------|------|---------|-----------------|
| Bohr's Atomic Model | 1913 | Solar system → Atom | **0.995** |
| Kekulé's Benzene Ring | 1865 | Ouroboros → Ring structure | **0.967** |
| Darwin's Natural Selection | 1859 | Malthus → Biology | **0.985** |

### Novel Analogies Discovered

AI-discovered cross-domain analogies (not taught):

| Discovery | Meaning |
|-----------|---------|
| **Revolution ≈ Emotion** | Social revolution as collective emotional response |
| **Compiler ≈ Rock Cycle** | Both involve staged transformation |
| **Immune System ≈ Information Spread** | Virus and viral share same structure |
| **Gene Expression ≈ Learning** | Same process at molecular and cognitive levels |
| **Revolution ≈ Story** | Revolution follows Freytag's pyramid |

### Running Structural Similarity Analysis

```bash
cd experiments/isomorphism_discovery
python novel_analogy_discovery.py
```

See: [experiments/structural_similarity_results.md](../../docs/experiments/structural_similarity_results.md)

---

## Performance Benchmarks

### Scalability

| Nodes | Edges | Time | Memory |
|-------|-------|------|--------|
| 100 | 1,000 | 0.3s | 50 MB |
| 500 | 10,000 | 1.8s | 200 MB |
| 1,000 | 50,000 | 5.5s | 800 MB |

### geDIG Computation

| Operation | Time (1000 nodes) |
|-----------|-------------------|
| Graph construction | 0.5s |
| ΔGED calculation | 1.2s |
| ΔIG calculation | 0.8s |
| Spike detection | 0.1s |

---

## Reproducibility

### Environment Setup

```bash
# Clone repository
git clone https://github.com/miyauchikazuyoshi/InsightSpike-AI.git
cd InsightSpike-AI

# Install dependencies
poetry install

# Set paper preset
export INSIGHTSPIKE_PRESET=paper
```

### Running Experiments

```bash
# Maze experiment
python experiments/maze/run_experiment.py

# Analogy discovery
python experiments/isomorphism_discovery/novel_analogy_discovery.py

# HotPotQA
python experiments/hotpotqa/run_evaluation.py
```

### Verification

```bash
# Using docxology
cd docxology
python run_all.py
```

---

## Source Documents

| Document | Path |
|----------|------|
| Experiment Index | [EXPERIMENTS.md](../../docs/EXPERIMENTS.md) |
| Maze Spec | [MAZE_NAV_SPEC.md](../../docs/MAZE_NAV_SPEC.md) |
| Maze Metrics | [HOWTO_maze_metrics.md](../../docs/HOWTO_maze_metrics.md) |
| Phase 1 | [phase1.md](../../docs/phase1.md) |
| Structural Similarity Results | [experiments/structural_similarity_results.md](../../docs/experiments/structural_similarity_results.md) |
| HotPotQA Cases | [design/hotpotqa_case_studies.md](../../docs/design/hotpotqa_case_studies.md) |
| Research Directory | [research/](../../docs/research/) |
| Paper Directory | [paper/](../../docs/paper/) |
