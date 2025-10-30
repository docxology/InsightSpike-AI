from __future__ import annotations

import os
import numpy as np

from insightspike.implementations.layers.layer3_graph_reasoner import L3GraphReasoner


def _mk_docs_base(n: int = 5, dim: int = 8):
    base = np.zeros(dim, dtype=np.float32)
    base[0] = 1.0
    docs = []
    for i in range(n):
        v = base.copy()
        v[min(i, dim - 1)] += 0.05
        docs.append({"text": f"b{i}", "embedding": v})
    return docs


def _mk_docs_variant(close: bool, n_base: int = 5, dim: int = 8):
    docs = _mk_docs_base(n_base, dim)
    v = np.zeros(dim, dtype=np.float32)
    if close:
        v[0] = 1.0
        v[1] = 0.2  # near existing center
    else:
        v[-1] = 1.0  # far direction
    docs.append({"text": "extra", "embedding": v})
    return docs


def test_rank_correlation_core_vs_cached_incr_small():
    # Ensure query-centric path is used
    os.environ.pop('INSIGHTSPIKE_MIN_IMPORT', None)
    os.environ['INSIGHTSPIKE_QUERY_CENTRIC'] = '1'
    os.environ['INSIGHTSPIKE_SP_PAIR_SAMPLES'] = '30'

    l3 = L3GraphReasoner()

    # Build previous graph
    prev_docs = _mk_docs_base()
    res0 = l3.analyze_documents(prev_docs, {})
    prev_graph = res0.get('graph')

    # Variant A: extra node close to center; Variant B: far
    docs_A = _mk_docs_variant(close=True)
    docs_B = _mk_docs_variant(close=False)

    # Core baseline ordering
    os.environ['INSIGHTSPIKE_SP_ENGINE'] = 'core'
    resA_core = l3.analyze_documents(docs_A, {"previous_graph": prev_graph, "centers": [0]})
    resB_core = l3.analyze_documents(docs_B, {"previous_graph": prev_graph, "centers": [0]})
    gA_core = resA_core.get('metrics', {}).get('gmin', 0.0)
    gB_core = resB_core.get('metrics', {}).get('gmin', 0.0)

    # Cached_incr ordering
    os.environ['INSIGHTSPIKE_SP_ENGINE'] = 'cached_incr'
    resA_ci = l3.analyze_documents(docs_A, {"previous_graph": prev_graph, "centers": [0]})
    resB_ci = l3.analyze_documents(docs_B, {"previous_graph": prev_graph, "centers": [0]})
    gA_ci = resA_ci.get('metrics', {}).get('gmin', 0.0)
    gB_ci = resB_ci.get('metrics', {}).get('gmin', 0.0)

    # Expect A (close) to be not worse than B in both engines (minimize gmin)
    assert gA_core <= gB_core + 1e-6
    assert gA_ci <= gB_ci + 1e-6

