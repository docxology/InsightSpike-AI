"""
PSZ (Performance Safe Zone) utilities.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Dict, Any
import numpy as np


@dataclass(frozen=True)
class PSZThresholds:
    acceptance: float = 0.95
    fmr: float = 0.02
    latency_p50_ms: float = 200.0


@dataclass(frozen=True)
class PSZSummary:
    acceptance_rate: float
    fmr: float
    latency_p50_ms: float
    inside_psz: bool


def summarize_accept_latency(samples: Iterable[Dict[str, Any]]) -> PSZSummary:
    accepted = []
    latencies = []
    for s in samples:
        if "accepted" in s:
            accepted.append(bool(s["accepted"]))
        if "latency_ms" in s:
            latencies.append(float(s["latency_ms"]))

    acc = float(np.mean(accepted)) if accepted else 0.0
    fmr = 1.0 - acc if accepted else 0.0
    p50 = float(np.percentile(latencies, 50)) if latencies else 0.0
    inside = (acc >= PSZThresholds.acceptance) and (fmr <= PSZThresholds.fmr) and (p50 <= PSZThresholds.latency_p50_ms)
    return PSZSummary(acceptance_rate=acc, fmr=fmr, latency_p50_ms=p50, inside_psz=inside)


def inside_psz(summary: PSZSummary, thresholds: PSZThresholds = PSZThresholds()) -> bool:
    return (
        summary.acceptance_rate >= thresholds.acceptance
        and summary.fmr <= thresholds.fmr
        and summary.latency_p50_ms <= thresholds.latency_p50_ms
    )


__all__ = ["PSZThresholds", "PSZSummary", "summarize_accept_latency", "inside_psz"]

