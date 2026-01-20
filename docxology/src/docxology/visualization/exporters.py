"""Exporter for saving results in multiple formats.

Provides utilities for exporting analysis results to JSON, CSV, HTML,
and Markdown formats.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class Exporter:
    """Exports results to multiple formats.
    
    Provides methods for exporting to JSON, CSV, HTML, and Markdown
    with configurable formatting options.
    
    Example:
        >>> exporter = Exporter()
        >>> exporter.to_json(stats, "report.json")
        >>> exporter.to_html(stats, "report.html")
    """
    
    def __init__(
        self,
        timestamp_files: bool = True,
        json_indent: int = 2,
    ) -> None:
        """Initialize the exporter.
        
        Args:
            timestamp_files: Add timestamp to filenames.
            json_indent: Indentation for JSON output.
        """
        self.timestamp_files = timestamp_files
        self.json_indent = json_indent
    
    def to_json(
        self,
        data: dict[str, Any],
        output: Path | str,
        include_metadata: bool = True,
    ) -> Path:
        """Export to JSON format.
        
        Args:
            data: Data to export.
            output: Output file path.
            include_metadata: Include export metadata.
            
        Returns:
            Path to exported file.
        """
        output = self._prepare_output(output, ".json")
        
        export_data = data.copy()
        if include_metadata:
            export_data["_export_metadata"] = {
                "format": "json",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0",
            }
        
        with open(output, "w", encoding="utf-8") as f:
            json.dump(export_data, f, indent=self.json_indent, default=str)
        
        logger.info(f"Exported JSON: {output}")
        return output
    
    def to_csv(
        self,
        data: dict[str, Any],
        output: Path | str,
    ) -> Path:
        """Export to CSV format.
        
        Args:
            data: Data to export (will be flattened).
            output: Output file path.
            
        Returns:
            Path to exported file.
        """
        output = self._prepare_output(output, ".csv")
        
        # Flatten data
        rows = self._flatten_for_csv(data)
        
        if not rows:
            rows = [{"key": "empty", "value": "No data"}]
        
        with open(output, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"Exported CSV: {output}")
        return output
    
    def to_html(
        self,
        data: dict[str, Any],
        output: Path | str,
        title: str = "Analysis Report",
    ) -> Path:
        """Export to HTML format.
        
        Args:
            data: Data to export.
            output: Output file path.
            title: HTML page title.
            
        Returns:
            Path to exported file.
        """
        output = self._prepare_output(output, ".html")
        
        html = self._generate_html(data, title)
        
        with open(output, "w", encoding="utf-8") as f:
            f.write(html)
        
        logger.info(f"Exported HTML: {output}")
        return output
    
    def to_markdown(
        self,
        data: dict[str, Any],
        output: Path | str,
        title: str = "Analysis Report",
    ) -> Path:
        """Export to Markdown format.
        
        Args:
            data: Data to export.
            output: Output file path.
            title: Document title.
            
        Returns:
            Path to exported file.
        """
        output = self._prepare_output(output, ".md")
        
        md = self._generate_markdown(data, title)
        
        with open(output, "w", encoding="utf-8") as f:
            f.write(md)
        
        logger.info(f"Exported Markdown: {output}")
        return output
    
    def _prepare_output(self, output: Path | str, extension: str) -> Path:
        """Prepare output path.
        
        Args:
            output: Base output path.
            extension: File extension.
            
        Returns:
            Prepared output path.
        """
        output = Path(output)
        
        if self.timestamp_files:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            stem = output.stem
            output = output.with_name(f"{stem}_{timestamp}{extension}")
        
        output.parent.mkdir(parents=True, exist_ok=True)
        return output
    
    def _flatten_for_csv(self, data: dict[str, Any], prefix: str = "") -> list[dict[str, str]]:
        """Flatten nested data for CSV export.
        
        Args:
            data: Nested data dictionary.
            prefix: Key prefix for flattening.
            
        Returns:
            List of flat rows.
        """
        rows = []
        
        for key, value in data.items():
            full_key = f"{prefix}.{key}" if prefix else key
            
            if isinstance(value, dict):
                rows.extend(self._flatten_for_csv(value, full_key))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        rows.extend(self._flatten_for_csv(item, f"{full_key}[{i}]"))
                    else:
                        rows.append({"key": f"{full_key}[{i}]", "value": str(item)})
            else:
                rows.append({"key": full_key, "value": str(value)})
        
        return rows
    
    def _generate_html(self, data: dict[str, Any], title: str) -> str:
        """Generate HTML content.
        
        Args:
            data: Data to include.
            title: Page title.
            
        Returns:
            HTML string.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; }}
        h1 {{ color: #333; }}
        h2 {{ color: #666; margin-top: 30px; }}
        table {{ border-collapse: collapse; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px 12px; text-align: left; }}
        th {{ background: #f5f5f5; }}
        .timestamp {{ color: #999; font-size: 0.875rem; }}
        pre {{ background: #f5f5f5; padding: 15px; overflow-x: auto; border-radius: 4px; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p class="timestamp">Generated: {timestamp}</p>
"""
        
        # Add summary table if present
        if "summary" in data and isinstance(data["summary"], dict):
            html += "<h2>Summary</h2>\n<table>\n<tr><th>Metric</th><th>Value</th></tr>\n"
            for key, value in data["summary"].items():
                if isinstance(value, dict):
                    value_str = ", ".join(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k, v in value.items())
                else:
                    value_str = f"{value:.3f}" if isinstance(value, float) else str(value)
                html += f"<tr><td>{key}</td><td>{value_str}</td></tr>\n"
            html += "</table>\n"
        
        # Add raw data
        html += "<h2>Raw Data</h2>\n<pre>"
        html += json.dumps(data, indent=2, default=str)
        html += "</pre>\n"
        
        html += "</body>\n</html>"
        return html
    
    def _generate_markdown(self, data: dict[str, Any], title: str) -> str:
        """Generate Markdown content.
        
        Args:
            data: Data to include.
            title: Document title.
            
        Returns:
            Markdown string.
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        md = f"# {title}\n\n"
        md += f"*Generated: {timestamp}*\n\n"
        
        # Add summary table if present
        if "summary" in data and isinstance(data["summary"], dict):
            md += "## Summary\n\n"
            md += "| Metric | Value |\n|--------|-------|\n"
            for key, value in data["summary"].items():
                if isinstance(value, dict):
                    value_str = ", ".join(f"{k}: {v:.3f}" if isinstance(v, float) else f"{k}: {v}" for k, v in value.items())
                else:
                    value_str = f"{value:.3f}" if isinstance(value, float) else str(value)
                md += f"| {key} | {value_str} |\n"
            md += "\n"
        
        # Add statistics if present
        if "count" in data:
            md += f"## Statistics\n\n- Total items: {data['count']}\n"
            if "by_type" in data:
                md += "- By type:\n"
                for t, count in data["by_type"].items():
                    md += f"  - {t}: {count}\n"
            md += "\n"
        
        # Add raw data
        md += "## Raw Data\n\n```json\n"
        md += json.dumps(data, indent=2, default=str)
        md += "\n```\n"
        
        return md
