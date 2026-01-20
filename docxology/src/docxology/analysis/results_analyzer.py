"""Results analyzer for processing execution outputs.

Provides utilities for parsing, summarizing, and analyzing results
from script and module executions.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class ResultsAnalyzer:
    """Analyzes execution results and generates summaries.
    
    Parses JSON/log outputs, computes statistics, identifies patterns,
    and generates structured reports.
    
    Example:
        >>> analyzer = ResultsAnalyzer()
        >>> stats = analyzer.analyze_directory(Path("output/results"))
        >>> print(stats["summary"])
    """
    
    def __init__(self) -> None:
        """Initialize the analyzer."""
        pass
    
    def analyze_directory(
        self,
        directory: Path | str,
        pattern: str = "*.json",
    ) -> dict[str, Any]:
        """Analyze all result files in a directory.
        
        Args:
            directory: Directory containing result files.
            pattern: Glob pattern for result files.
            
        Returns:
            Aggregated analysis results.
        """
        directory = Path(directory)
        
        if not directory.exists():
            logger.warning(f"Directory does not exist: {directory}")
            return {"error": "Directory not found", "path": str(directory)}
        
        results = []
        files = list(directory.glob(pattern))
        
        logger.info(f"Analyzing {len(files)} files in {directory}")
        
        for file_path in files:
            try:
                result = self.analyze_file(file_path)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to analyze {file_path}: {e}")
        
        return self._aggregate_results(results)
    
    def analyze_file(self, file_path: Path | str) -> dict[str, Any]:
        """Analyze a single result file.
        
        Args:
            file_path: Path to the result file.
            
        Returns:
            Analysis of the file.
        """
        file_path = Path(file_path)
        
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        analysis = {
            "file": file_path.name,
            "type": self._detect_result_type(data),
            "metrics": self._extract_metrics(data),
        }
        
        return analysis
    
    def _detect_result_type(self, data: dict[str, Any]) -> str:
        """Detect the type of result data.
        
        Args:
            data: Result data dictionary.
            
        Returns:
            Result type string.
        """
        if "maze" in str(data.get("config", {})):
            return "maze"
        if "success_rate" in data:
            return "benchmark"
        if "gedig" in str(data).lower():
            return "gedig"
        return "unknown"
    
    def _extract_metrics(self, data: dict[str, Any]) -> dict[str, Any]:
        """Extract numeric metrics from result data.
        
        Args:
            data: Result data dictionary.
            
        Returns:
            Dictionary of extracted metrics.
        """
        metrics = {}
        
        # Recursively find numeric values
        def extract_numeric(obj: Any, prefix: str = "") -> None:
            if isinstance(obj, dict):
                for key, value in obj.items():
                    new_prefix = f"{prefix}.{key}" if prefix else key
                    extract_numeric(value, new_prefix)
            elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
                metrics[prefix] = obj
            elif isinstance(obj, list) and obj and isinstance(obj[0], (int, float)):
                metrics[f"{prefix}_mean"] = np.mean(obj)
                metrics[f"{prefix}_std"] = np.std(obj)
                metrics[f"{prefix}_min"] = np.min(obj)
                metrics[f"{prefix}_max"] = np.max(obj)
        
        extract_numeric(data)
        return metrics
    
    def _aggregate_results(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate multiple result analyses.
        
        Args:
            results: List of analysis results.
            
        Returns:
            Aggregated statistics.
        """
        if not results:
            return {"summary": {}, "count": 0}
        
        # Collect all metrics
        all_metrics: dict[str, list[float]] = {}
        
        for result in results:
            for key, value in result.get("metrics", {}).items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
        
        # Compute aggregate statistics
        summary = {}
        for key, values in all_metrics.items():
            if values:
                summary[key] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "count": len(values),
                }
        
        # Count by type
        type_counts = {}
        for result in results:
            t = result.get("type", "unknown")
            type_counts[t] = type_counts.get(t, 0) + 1
        
        return {
            "summary": summary,
            "count": len(results),
            "by_type": type_counts,
            "files": [r.get("file") for r in results],
        }
    
    def compute_summary(self, data: dict[str, Any]) -> dict[str, Any]:
        """Compute a summary of result data.
        
        Args:
            data: Result data dictionary.
            
        Returns:
            Summary statistics.
        """
        metrics = self._extract_metrics(data)
        
        return {
            "total_metrics": len(metrics),
            "metrics": metrics,
            "type": self._detect_result_type(data),
        }
