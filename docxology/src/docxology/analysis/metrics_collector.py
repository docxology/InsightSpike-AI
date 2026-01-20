"""Metrics collector for aggregating execution metrics.

Provides utilities for collecting, computing, and storing metrics
from multiple execution runs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MetricSample:
    """A single metric sample.
    
    Attributes:
        name: Metric name.
        value: Metric value.
        timestamp: When the sample was recorded.
        tags: Optional tags for categorization.
    """
    name: str
    value: float
    timestamp: float = field(default_factory=time.time)
    tags: dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collects and aggregates metrics from executions.
    
    Provides methods for recording individual samples, computing
    aggregations, and exporting metrics data.
    
    Example:
        >>> collector = MetricsCollector()
        >>> collector.record("duration", 1.5, tags={"script": "maze"})
        >>> collector.record("duration", 2.3, tags={"script": "maze"})
        >>> stats = collector.get_stats("duration")
    """
    
    def __init__(self) -> None:
        """Initialize the collector."""
        self._samples: dict[str, list[MetricSample]] = {}
    
    def record(
        self,
        name: str,
        value: float,
        tags: dict[str, str] | None = None,
    ) -> None:
        """Record a metric sample.
        
        Args:
            name: Metric name.
            value: Metric value.
            tags: Optional tags.
        """
        sample = MetricSample(name=name, value=value, tags=tags or {})
        
        if name not in self._samples:
            self._samples[name] = []
        self._samples[name].append(sample)
    
    def record_batch(
        self,
        metrics: dict[str, float],
        tags: dict[str, str] | None = None,
    ) -> None:
        """Record multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric name to value.
            tags: Optional tags for all metrics.
        """
        for name, value in metrics.items():
            self.record(name, value, tags)
    
    def get_stats(self, name: str, tags: dict[str, str] | None = None) -> dict[str, Any]:
        """Get statistics for a metric.
        
        Args:
            name: Metric name.
            tags: Optional tag filter.
            
        Returns:
            Statistics dictionary.
        """
        samples = self._samples.get(name, [])
        
        if tags:
            samples = [
                s for s in samples
                if all(s.tags.get(k) == v for k, v in tags.items())
            ]
        
        if not samples:
            return {"count": 0}
        
        values = [s.value for s in samples]
        
        return {
            "name": name,
            "count": len(values),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "sum": float(np.sum(values)),
            "median": float(np.median(values)),
        }
    
    def get_all_stats(self) -> dict[str, dict[str, Any]]:
        """Get statistics for all metrics.
        
        Returns:
            Dictionary of metric name to statistics.
        """
        return {name: self.get_stats(name) for name in self._samples}
    
    def get_timeseries(
        self,
        name: str,
        tags: dict[str, str] | None = None,
    ) -> list[tuple[float, float]]:
        """Get time series data for a metric.
        
        Args:
            name: Metric name.
            tags: Optional tag filter.
            
        Returns:
            List of (timestamp, value) tuples.
        """
        samples = self._samples.get(name, [])
        
        if tags:
            samples = [
                s for s in samples
                if all(s.tags.get(k) == v for k, v in tags.items())
            ]
        
        return [(s.timestamp, s.value) for s in sorted(samples, key=lambda s: s.timestamp)]
    
    def list_metrics(self) -> list[str]:
        """List all metric names.
        
        Returns:
            List of metric names.
        """
        return list(self._samples.keys())
    
    def count(self, name: str | None = None) -> int:
        """Get sample count.
        
        Args:
            name: Optional metric name. If None, return total count.
            
        Returns:
            Sample count.
        """
        if name:
            return len(self._samples.get(name, []))
        return sum(len(samples) for samples in self._samples.values())
    
    def clear(self, name: str | None = None) -> None:
        """Clear samples.
        
        Args:
            name: Optional metric name. If None, clear all.
        """
        if name:
            self._samples.pop(name, None)
        else:
            self._samples.clear()
    
    def to_dict(self) -> dict[str, Any]:
        """Export all data as dictionary.
        
        Returns:
            Dictionary with all samples and stats.
        """
        return {
            "metrics": self.list_metrics(),
            "stats": self.get_all_stats(),
            "total_samples": self.count(),
        }
