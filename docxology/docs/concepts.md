# Core Concepts

Comprehensive summary of theory and concepts from InsightSpike-AI documentation.

## geDIG: Graph Edit Distance as Insight Gauge

### The Core Insight

**Insight = discovering minimal graph transformations that achieve structural isomorphism.**

The fundamental equation:

```
T* = argmin_T GED(T(G₁), G₂)
```

> *Find the transformation T that minimizes the graph edit distance between transformed G₁ and G₂. That transformation IS the insight.*

---

## Canonical Gauge Definition (Paper v4)

### Primary Gauge Equation

```
F = ΔEPC_norm - λ ( ΔH_norm + γ · ΔSP_rel )
```

Where:
- **ΔEPC_norm**: Normalized Edit Path Cost (Structure Cost)
- **ΔH_norm**: Entropy difference (after - before) normalized by log(K)
- **ΔSP_rel**: Relative Shortest Path Gain = (L_before - L_after) / L_before
- **λ (lambda_weight)**: Information Temperature (default: 1.0)
- **γ (sp_beta)**: SP Gain Weight (default: 1.0)

### Normalization

| Symbol | Normalization | Config |
|--------|---------------|--------|
| ΔH_norm | Divided by log(K), K = candidates in 'after' | `ig_norm_strategy` |
| ΔEPC_norm | Divided by upper bound | `ged_norm_scheme = "candidate_base"` |

### Two-Stage Gating (AG/DG)

```
0-hop:     g₀ = ΔEPC_norm - λ · ΔH_norm

Multi-hop: g_min = min_h { ΔEPC_norm - λ(ΔH_norm + γ · ΔSP_rel^(h)) }

Accept when: AG indicates high novelty AND min{g₀, g_min} ≤ θ_DG
```

---

## Three Levels of Understanding

```
┌───────────────────────────────────────────────────────┐
│  Level 3: Isomorphism Discovery      [Einstein-level] │
│  T* = argmin_T GED(T(G₁), G₂)                         │
│  "Discover transformations that unify theories"       │
└─────────────────────┬─────────────────────────────────┘
                      │
┌─────────────────────┴─────────────────────────────────┐
│  Level 2: Analogy Detection              [Bohr-level] │
│  SS(G₁, G₂) > θ                                       │
│  "Detect structural correspondence across domains"    │
└─────────────────────┬─────────────────────────────────┘
                      │
┌─────────────────────┴─────────────────────────────────┐
│  Level 1: Pattern Matching             [Standard RAG] │
│  sim(a,b) = cos(φ(a), φ(b))                           │
│  "Element-level semantic similarity"                  │
└───────────────────────────────────────────────────────┘
```

---

## Key Metrics

| Metric | Formula | Meaning | Threshold |
|--------|---------|---------|-----------|
| ΔGED | `GED(G_new, G_old)` | Structural change | < -0.5 |
| ΔIG | `H(G_old) - H(G_new)` | Information gain | > 0.2 |
| Spike | `ΔGED < θ_ged && ΔIG > θ_ig` | Eureka moment | — |

### Spike Detection

```python
def detect_eureka_spike(delta_ged, delta_ig):
    """Detect if current state constitutes a eureka spike."""
    ged_threshold = config.graph.spike_ged_threshold  # default: -0.5
    ig_threshold = config.graph.spike_ig_threshold    # default: 0.2
    
    return (delta_ged < ged_threshold) and (delta_ig > ig_threshold)
```

---

## Implementation Mapping

### Key Parameters

| Parameter | Config Key | Default | File |
|-----------|------------|---------|------|
| λ | `lambda_weight` | 1.0 | `gedig_core.py` |
| γ | `sp_beta` | 1.0 | `gedig_core.py` |
| ΔH temperature | `entropy_tau` | 1.0 | `gedig_core.py` |
| Norm strategy | `ged_norm_scheme` | `candidate_base` | `presets.py` |

### Paper Preset Configuration

```python
# src/insightspike/config/presets.py -> paper()
graph.sp_scope_mode = "union"
graph.sp_eval_mode = "fixed_before_pairs"
graph.ged_norm_scheme = "candidate_base"
graph.ig_source_mode = "linkset"
graph.lambda_weight = 1.0
graph.sp_beta = 1.0
metrics.ig_denominator = "fixed_kstar"
metrics.use_local_normalization = True
```

---

## Active Inference Connection

geDIG implements Active Inference principles:

| AI Principle | geDIG Implementation |
|--------------|---------------------|
| Free Energy Minimization | Minimize gauge F |
| Bayesian Inference | Update graph structure |
| Prediction Error | ΔIG measurement |
| Expected Free Energy | G = ΔEPC - λ·ΔH |

---

## Universal Principle Hypothesis

The structural similarity principle is universal across domains:

| Source | Insight | Target |
|--------|---------|--------|
| 🌞 Solar system | Orbital structure | ⚛️ Bohr's atomic model (1913) |
| 💧 Water flow | Potential difference | ⚡ Electric circuits |
| 🐍 Snake ring (ouroboros) | Ring structure | 💎 Kekulé's benzene (1865) |
| 🧬 Genetic code | Information replication | 💻 Computer programs |

**Performance on Historical Discoveries:**

| Discovery | Year | Structural Similarity | Detection |
|-----------|------|----------------------|-----------|
| Bohr's Atomic Model | 1913 | 0.995 | ✅ |
| Kekulé's Benzene Ring | 1865 | 0.967 | ✅ |
| Darwin's Natural Selection | 1859 | 0.985 | ✅ |

---

## Source Documents

| Document | Path |
|----------|------|
| geDIG Specification (v4) | [gedig_spec.md](../../docs/gedig_spec.md) |
| geDIG in 5 Minutes | [concepts/gedig_in_5_minutes.md](../../docs/concepts/gedig_in_5_minutes.md) |
| Intuition | [concepts/intuition.md](../../docs/concepts/intuition.md) |
| Universal Principle | [concepts/universal_principle_hypothesis.md](../../docs/concepts/universal_principle_hypothesis.md) |
| Theory | [theory.md](../../docs/theory.md) |
| Glossary | [glossary.md](../../docs/glossary.md) |
| Algorithm: geDIG Core | [src/insightspike/algorithms/gedig_core.py](../../src/insightspike/algorithms/gedig_core.py) |
| Algorithm: Information Gain | [src/insightspike/algorithms/information_gain.py](../../src/insightspike/algorithms/information_gain.py) |
