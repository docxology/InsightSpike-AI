"""
AG/DG gate decider utilities (shared).

- AG (Attention Gate): triggers when g0 > theta_ag
- DG (Decision Gate): triggers when min(g0, gmin) <= theta_dg
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GateDecision:
    g0: float
    gmin: float
    ag: bool
    dg: bool
    b_value: float  # b(t) = min(g0, gmin)


def decide_gates(g0: float, gmin: float, theta_ag: float, theta_dg: float) -> GateDecision:
    b_val = g0 if g0 < gmin else gmin
    ag_fire = bool(g0 > theta_ag)
    dg_fire = bool(b_val <= theta_dg)
    return GateDecision(g0=float(g0), gmin=float(gmin), ag=ag_fire, dg=dg_fire, b_value=float(b_val))


__all__ = ["GateDecision", "decide_gates"]

