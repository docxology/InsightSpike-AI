from __future__ import annotations

import os
import numpy as np

from insightspike.implementations.layers.layer3_graph_reasoner import L3GraphReasoner


def _mk_base(n=6, dim=8, noise=0.0):
    rng = np.random.default_rng(0)
    base = np.zeros(dim, dtype=np.float32)
    base[0] = 1.0
    docs = []
    for i in range(n):
        v = base.copy()
        v[min(i, dim - 1)] += 0.05
        if noise:
            v += rng.normal(0, noise, size=dim).astype(np.float32)
        docs.append({"text": f"b{i}", "embedding": v})
    return docs


def _variant(close_shift=0.2, far_axis=-1, dim=8):
    v_close = np.zeros(dim, dtype=np.float32); v_close[0] = 1.0; v_close[1] = close_shift
    v_far = np.zeros(dim, dtype=np.float32); v_far[far_axis] = 1.0
    return {
        'A': {"text": "extraA", "embedding": v_close},
        'B': {"text": "extraB", "embedding": v_far},
    }


def _eval_gmin(l3, prev_graph, docs_extra, engine):
    os.environ['INSIGHTSPIKE_SP_ENGINE'] = engine
    res = l3.analyze_documents(docs_extra, {"previous_graph": prev_graph, "centers": [0]})
    return res.get('metrics', {}).get('gmin', 0.0)


def test_rank_correlation_multiple_cases():
    os.environ.pop('INSIGHTSPIKE_MIN_IMPORT', None)
    os.environ['INSIGHTSPIKE_QUERY_CENTRIC'] = '1'
    os.environ['INSIGHTSPIKE_SP_PAIR_SAMPLES'] = '50'

    l3 = L3GraphReasoner()

    # Build previous graph with light noise
    prev_docs = _mk_base(n=6, dim=8, noise=0.01)
    res0 = l3.analyze_documents(prev_docs, {})
    prev_graph = res0.get('graph')

    # 3 cases of variants (vary far axis and close shift)
    cases = [
        {'close_shift': 0.2, 'far_axis': -1},
        {'close_shift': 0.15, 'far_axis': -2},
        {'close_shift': 0.25, 'far_axis': -3},
    ]

    for c in cases:
        var = _variant(close_shift=c['close_shift'], far_axis=c['far_axis'])
        docs_A = prev_docs + [var['A']]
        docs_B = prev_docs + [var['B']]
        gA_core = _eval_gmin(l3, prev_graph, docs_A, 'core')
        gB_core = _eval_gmin(l3, prev_graph, docs_B, 'core')
        gA_ci = _eval_gmin(l3, prev_graph, docs_A, 'cached_incr')
        gB_ci = _eval_gmin(l3, prev_graph, docs_B, 'cached_incr')
        tol = 1e-3
        assert gA_core <= gB_core + tol
        assert gA_ci <= gB_ci + tol
