# Graph SP Engine and NormSpec — API Reference

Status: current (2025-10-30)

## Overview
- SP engine switch: `core | cached | cached_incr`
- NormSpec propagation via `context['norm_spec']` and echo in L3 metrics
- Candidate edges interface for `cached_incr`: `List[Tuple[int, int, Dict]]`

## Config keys (config.yaml)
```yaml
graph:
  sp_engine: core            # core | cached | cached_incr
  lambda_weight: 1.0         # weight λ
  sp_beta: 0.2               # ΔSP weight γ
  sp_scope_mode: auto        # auto | union
  sp_boundary_mode: trim     # induced | trim | nodes
  cached_incr_budget: 1      # max adopted edges (sequential)
  # NormSpec (optional override)
  norm_spec:
    metric: cosine
    radius_mode: intuitive
    intuitive: { outer: 0.6, inner: 0.2 }
    dimension: 384
    scope: sphere            # sphere | donut
```

## ENV overrides
```
INSIGHTSPIKE_SP_ENGINE=cached_incr
INSIGHTSPIKE_SP_PAIR_SAMPLES=200
INSIGHTSPIKE_SP_REGISTRY=/path/to/pairsets.json
INSIGHTSPIKE_GEDIG_LAMBDA=1.0
INSIGHTSPIKE_SP_BETA=0.2
INSIGHTSPIKE_SP_BUDGET=2
INSIGHTSPIKE_CAND_TOPK=10
```

## Context interface (L1/L2 → L3)
```python
context = {
  'centers': [int, ...],
  'candidate_edges': [(u, v, {'score': float}), ...],
  'candidate_selection': {'k_star': int, 'l1_candidates': int, 'log_k_star': float},
  'norm_spec': {
     'metric': 'cosine',
     'radius_mode': 'intuitive',
     'intuitive': {'outer': 0.6, 'inner': 0.2},
     'dimension': 384,
     'scope': 'sphere',
     'effective': {'theta_link': 0.35, 'theta_cand': 0.45}
  }
}
```

## Metrics output (L3)
- `metrics.sp_engine`: `core | cached | cached_incr`
- `metrics.norm_spec`: echo of `context['norm_spec']` (or config fallback)
- `metrics.delta_ged, delta_ig, delta_h, delta_sp, g0, gmin, graph_size_*`

## Behavior summary
- `core`: multi-hop (GeDIGCore) with SP gain integrated per hop
- `cached`: hop0 for ΔEPC_norm/ΔH + between-graphs ΔSP (fixed-before pairs)
- `cached_incr`: candidate edges sequential adoption with ΔSP recompute (greedy), budgeted; auto-candidate fallback if `candidate_edges` missing

## Notes
- NormSpec defaults derive from WakeSleep SphereSearchConfig (intuitive radii) when not explicitly configured
- ENV overrides > config > defaults

