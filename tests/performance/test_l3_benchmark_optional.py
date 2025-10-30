from __future__ import annotations

import os
import numpy as np
import pytest


pytest.importorskip("pytest_benchmark")

from insightspike.implementations.layers.layer3_graph_reasoner import L3GraphReasoner


def _mk_docs(n: int = 24, dim: int = 16, seed: int = 123):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, size=(n, dim)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return [{"text": f"d{i}", "embedding": x[i]} for i in range(n)]


def test_benchmark_cached_incr(benchmark):
    os.environ.pop('INSIGHTSPIKE_MIN_IMPORT', None)
    os.environ['INSIGHTSPIKE_QUERY_CENTRIC'] = '1'
    os.environ['INSIGHTSPIKE_SP_ENGINE'] = 'cached_incr'
    os.environ['INSIGHTSPIKE_SP_PAIR_SAMPLES'] = '80'
    os.environ['INSIGHTSPIKE_SP_BUDGET'] = '2'
    # modest Phase1 knobs (kept conservative)
    os.environ['INSIGHTSPIKE_CAND_PRUNE_TOPK'] = '12'
    os.environ['INSIGHTSPIKE_SP_ENDP_SSSP_WINDOW'] = '1'
    os.environ['INSIGHTSPIKE_SP_SOURCES_CAP'] = '64'
    os.environ['INSIGHTSPIKE_SP_SOURCES_FOCUS'] = 'near'

    l3 = L3GraphReasoner()
    prev_docs = _mk_docs(20, 16, 1)
    res0 = l3.analyze_documents(prev_docs, {})
    prev_graph = res0.get('graph')
    docs = _mk_docs(24, 16, 2)

    def run():
        res = l3.analyze_documents(docs, {"previous_graph": prev_graph, "centers": [0]})
        assert 'metrics' in res

    benchmark(run)

