from __future__ import annotations

import os
import time
import numpy as np

from insightspike.implementations.layers.layer3_graph_reasoner import L3GraphReasoner


def _mk_docs(n: int = 20, dim: int = 16):
    rng = np.random.default_rng(42)
    base = rng.normal(0, 1, size=(n, dim)).astype(np.float32)
    base /= np.linalg.norm(base, axis=1, keepdims=True) + 1e-12
    return [{"text": f"d{i}", "embedding": base[i]} for i in range(n)]


def test_sequential_cached_incr_perf_smoke():
    os.environ.pop('INSIGHTSPIKE_MIN_IMPORT', None)
    os.environ['INSIGHTSPIKE_QUERY_CENTRIC'] = '1'
    os.environ['INSIGHTSPIKE_SP_ENGINE'] = 'cached_incr'
    os.environ['INSIGHTSPIKE_SP_PAIR_SAMPLES'] = '60'
    os.environ['INSIGHTSPIKE_SP_BUDGET'] = '2'

    l3 = L3GraphReasoner()
    prev_docs = _mk_docs(n=15, dim=16)
    res0 = l3.analyze_documents(prev_docs, {})
    prev_graph = res0.get('graph')

    times = []
    for _ in range(5):
        docs = _mk_docs(n=16, dim=16)
        t0 = time.time()
        res = l3.analyze_documents(docs, {"previous_graph": prev_graph, "centers": [0]})
        dt = (time.time() - t0) * 1000.0
        assert 'metrics' in res and res['metrics'].get('sp_engine') in ('cached_incr', 'cached', 'core')
        times.append(dt)

    # Smoke: median under a loose bound on CI
    times.sort()
    p50 = times[len(times)//2]
    assert p50 < 5000.0

