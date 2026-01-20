"""Reporter for generating comprehensive reports.

Combines analysis results, visualizations, and metadata into
comprehensive report documents.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from .plotters import Plotter
from .exporters import Exporter

logger = logging.getLogger(__name__)


class Reporter:
    """Generates comprehensive analysis reports.
    
    Combines analysis results, visualizations, and metadata into
    complete report packages with multiple output formats.
    
    Example:
        >>> reporter = Reporter(output_dir="output/reports")
        >>> reporter.generate("experiment_001", results)
    """
    
    def __init__(
        self,
        output_dir: Path | str = "output/reports",
        generate_plots: bool = True,
    ) -> None:
        """Initialize the reporter.
        
        Args:
            output_dir: Directory for report output.
            generate_plots: Whether to generate plot images.
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.generate_plots = generate_plots
        self.plotter = Plotter()
        self.exporter = Exporter(timestamp_files=False)
    
    def generate(
        self,
        name: str,
        data: dict[str, Any],
        formats: list[str] | None = None,
    ) -> dict[str, Path]:
        """Generate a complete report.
        
        Args:
            name: Report name (used for filenames).
            data: Analysis data to include.
            formats: Output formats (default: json, html, markdown).
            
        Returns:
            Dictionary of format to output path.
        """
        formats = formats or ["json", "html", "markdown"]
        
        # Create report directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_dir = self.output_dir / f"{name}_{timestamp}"
        report_dir.mkdir(parents=True, exist_ok=True)
        
        outputs: dict[str, Path] = {}
        
        # Add report metadata
        report_data = {
            "report": {
                "name": name,
                "generated": datetime.now().isoformat(),
                "version": "1.0",
            },
            **data,
        }
        
        # Generate plots if enabled
        if self.generate_plots and "summary" in data:
            figures_dir = report_dir / "figures"
            figures_dir.mkdir(exist_ok=True)
            
            try:
                fig_path = figures_dir / "performance.png"
                self.plotter.performance_chart(data, output=fig_path)
                outputs["performance_chart"] = fig_path
                report_data["report"]["figures"] = [str(fig_path)]
            except Exception as e:
                logger.warning(f"Failed to generate performance chart: {e}")
        
        # Export in each format
        for fmt in formats:
            try:
                if fmt == "json":
                    path = report_dir / f"{name}.json"
                    self.exporter.to_json(report_data, path)
                    outputs["json"] = path
                elif fmt == "html":
                    path = report_dir / f"{name}.html"
                    self.exporter.to_html(report_data, path, title=f"Report: {name}")
                    outputs["html"] = path
                elif fmt == "markdown":
                    path = report_dir / f"{name}.md"
                    self.exporter.to_markdown(report_data, path, title=f"Report: {name}")
                    outputs["markdown"] = path
                elif fmt == "csv":
                    path = report_dir / f"{name}.csv"
                    self.exporter.to_csv(report_data, path)
                    outputs["csv"] = path
            except Exception as e:
                logger.error(f"Failed to export {fmt}: {e}")
        
        logger.info(f"Generated report: {report_dir}")
        return outputs
    
    def generate_summary(
        self,
        results: list[dict[str, Any]],
        output: Path | str,
    ) -> Path:
        """Generate a summary report from multiple results.
        
        Args:
            results: List of result dictionaries.
            output: Output file path.
            
        Returns:
            Path to output file.
        """
        output = Path(output)
        
        summary = {
            "total_results": len(results),
            "timestamp": datetime.now().isoformat(),
            "results": results,
        }
        
        # Compute aggregate statistics
        if results:
            success_count = sum(1 for r in results if r.get("success", False))
            summary["success_rate"] = success_count / len(results)
            
            durations = [r.get("duration", 0) for r in results if "duration" in r]
            if durations:
                import numpy as np
                summary["duration_stats"] = {
                    "mean": float(np.mean(durations)),
                    "std": float(np.std(durations)),
                    "min": float(np.min(durations)),
                    "max": float(np.max(durations)),
                    "total": float(np.sum(durations)),
                }
        
        return self.exporter.to_json(summary, output)
