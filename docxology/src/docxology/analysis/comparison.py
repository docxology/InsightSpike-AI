"""Results comparison for comparing multiple execution runs.

Provides utilities for comparing results across different configurations,
versions, or time periods.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ResultsComparison:
    """Compares results across multiple runs.
    
    Provides methods for computing differences, improvements, and
    statistical significance of changes.
    
    Example:
        >>> comparison = ResultsComparison()
        >>> diff = comparison.compare(baseline, experiment)
        >>> print(diff["improvements"])
    """
    
    def __init__(self, method: str = "relative") -> None:
        """Initialize the comparison.
        
        Args:
            method: Comparison method ('relative', 'absolute', 'percentage').
        """
        self.method = method
    
    def compare(
        self,
        baseline: dict[str, Any],
        experiment: dict[str, Any],
        keys: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare two result sets.
        
        Args:
            baseline: Baseline results.
            experiment: Experiment results.
            keys: Optional specific keys to compare.
            
        Returns:
            Comparison results.
        """
        # Extract comparable metrics
        baseline_metrics = self._extract_metrics(baseline)
        experiment_metrics = self._extract_metrics(experiment)
        
        if keys:
            baseline_metrics = {k: v for k, v in baseline_metrics.items() if k in keys}
            experiment_metrics = {k: v for k, v in experiment_metrics.items() if k in keys}
        
        # Find common keys
        common_keys = set(baseline_metrics.keys()) & set(experiment_metrics.keys())
        
        comparisons = {}
        improvements = []
        regressions = []
        
        for key in common_keys:
            b_val = baseline_metrics[key]
            e_val = experiment_metrics[key]
            
            diff = self._compute_difference(b_val, e_val)
            comparisons[key] = diff
            
            if diff["change"] > 0:
                improvements.append(key)
            elif diff["change"] < 0:
                regressions.append(key)
        
        return {
            "method": self.method,
            "baseline_keys": len(baseline_metrics),
            "experiment_keys": len(experiment_metrics),
            "common_keys": len(common_keys),
            "comparisons": comparisons,
            "improvements": improvements,
            "regressions": regressions,
            "summary": {
                "improved": len(improvements),
                "regressed": len(regressions),
                "unchanged": len(common_keys) - len(improvements) - len(regressions),
            },
        }
    
    def _compute_difference(self, baseline: float, experiment: float) -> dict[str, Any]:
        """Compute difference between two values.
        
        Args:
            baseline: Baseline value.
            experiment: Experiment value.
            
        Returns:
            Difference details.
        """
        absolute = experiment - baseline
        
        if baseline != 0:
            relative = (experiment - baseline) / abs(baseline)
            percentage = relative * 100
        else:
            relative = float("inf") if experiment != 0 else 0
            percentage = float("inf") if experiment != 0 else 0
        
        return {
            "baseline": baseline,
            "experiment": experiment,
            "absolute": absolute,
            "relative": relative,
            "percentage": percentage,
            "change": absolute if self.method == "absolute" else relative,
        }
    
    def _extract_metrics(self, data: dict[str, Any]) -> dict[str, float]:
        """Extract numeric metrics from data.
        
        Args:
            data: Data dictionary.
            
        Returns:
            Dictionary of metric name to value.
        """
        metrics = {}
        
        def extract(obj: Any, prefix: str = "") -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    extract(value, new_prefix)
            elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
                metrics[prefix] = float(obj)
        
        extract(data)
        return metrics
    
    def compare_multiple(
        self,
        results: list[dict[str, Any]],
        labels: list[str] | None = None,
    ) -> dict[str, Any]:
        """Compare multiple result sets.
        
        Args:
            results: List of result dictionaries.
            labels: Optional labels for each result set.
            
        Returns:
            Multi-way comparison results.
        """
        if len(results) < 2:
            return {"error": "Need at least 2 result sets"}
        
        if not labels:
            labels = [f"run_{i}" for i in range(len(results))]
        
        # Extract metrics from all
        all_metrics = [self._extract_metrics(r) for r in results]
        
        # Find common keys
        common_keys = set(all_metrics[0].keys())
        for m in all_metrics[1:]:
            common_keys &= set(m.keys())
        
        # Build comparison table
        table = {}
        for key in common_keys:
            values = [m[key] for m in all_metrics]
            table[key] = {
                "values": dict(zip(labels, values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "range": float(np.max(values) - np.min(values)),
            }
        
        return {
            "labels": labels,
            "common_keys": len(common_keys),
            "table": table,
        }
