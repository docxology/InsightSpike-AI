#!/usr/bin/env python
"""Analyze results from execution runs.

This example demonstrates using the analysis and visualization systems
to process and visualize execution results.

Usage:
    python examples/analyze_results.py [results_dir]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Add paths
DOCXOLOGY_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(DOCXOLOGY_ROOT / "src"))

from rich.console import Console
from rich.table import Table

from docxology.analysis import ResultsAnalyzer, MetricsCollector, ResultsComparison
from docxology.visualization import Plotter, Exporter


def create_sample_results(output_dir: Path) -> None:
    """Create sample results for demonstration."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = [
        {"run": 1, "success": True, "accuracy": 0.92, "duration": 1.5, "maze_size": 15},
        {"run": 2, "success": True, "accuracy": 0.95, "duration": 1.3, "maze_size": 15},
        {"run": 3, "success": False, "accuracy": 0.88, "duration": 2.1, "maze_size": 15},
        {"run": 4, "success": True, "accuracy": 0.96, "duration": 1.2, "maze_size": 15},
        {"run": 5, "success": True, "accuracy": 0.94, "duration": 1.4, "maze_size": 15},
    ]
    
    for result in results:
        path = output_dir / f"result_{result['run']}.json"
        path.write_text(json.dumps(result, indent=2))
    
    return results


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Analyze execution results")
    parser.add_argument(
        "results_dir",
        nargs="?",
        default=None,
        help="Directory containing result files",
    )
    parser.add_argument(
        "--export",
        "-e",
        action="store_true",
        help="Export analysis to files",
    )
    args = parser.parse_args()
    
    console = Console()
    console.print("\n[bold blue]Docxology — Results Analysis[/bold blue]\n")
    
    # Use provided directory or create sample data
    if args.results_dir:
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            console.print(f"[red]Directory not found: {results_dir}[/red]")
            return
    else:
        console.print("[yellow]No results directory provided. Creating sample data...[/yellow]\n")
        results_dir = DOCXOLOGY_ROOT / "output" / "sample_results"
        create_sample_results(results_dir)
    
    # 1. Analyze directory
    console.print(f"[cyan]Analyzing: {results_dir}[/cyan]\n")
    
    analyzer = ResultsAnalyzer()
    stats = analyzer.analyze_directory(results_dir)
    
    # 2. Display summary
    table = Table(title=f"Analysis Summary ({stats.get('count', 0)} files)")
    table.add_column("Metric", style="cyan")
    table.add_column("Mean", style="green")
    table.add_column("Std", style="yellow")
    table.add_column("Min", style="dim")
    table.add_column("Max", style="dim")
    
    for key, value in stats.get("summary", {}).items():
        if isinstance(value, dict) and "mean" in value:
            table.add_row(
                key,
                f"{value['mean']:.4f}",
                f"{value['std']:.4f}",
                f"{value['min']:.4f}",
                f"{value['max']:.4f}",
            )
    
    console.print(table)
    
    # 3. Collect metrics
    console.print("\n[bold]Metrics Collection:[/bold]\n")
    
    collector = MetricsCollector()
    
    for file in results_dir.glob("*.json"):
        try:
            data = json.loads(file.read_text())
            for key, value in data.items():
                if isinstance(value, (int, float)):
                    collector.record(key, float(value), tags={"file": file.name})
        except Exception:
            pass
    
    for metric in collector.list_metrics()[:5]:
        metric_stats = collector.get_stats(metric)
        console.print(f"  {metric}: mean={metric_stats['mean']:.4f}, count={metric_stats['count']}")
    
    # 4. Export if requested
    if args.export:
        console.print("\n[bold]Exporting results...[/bold]\n")
        
        output_dir = DOCXOLOGY_ROOT / "output" / "analysis"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        exporter = Exporter(timestamp_files=False)
        
        json_path = exporter.to_json(stats, output_dir / "analysis.json")
        console.print(f"  [green]✓[/green] JSON: {json_path}")
        
        html_path = exporter.to_html(stats, output_dir / "analysis.html")
        console.print(f"  [green]✓[/green] HTML: {html_path}")
        
        md_path = exporter.to_markdown(stats, output_dir / "analysis.md")
        console.print(f"  [green]✓[/green] Markdown: {md_path}")
        
        # Generate chart
        try:
            plotter = Plotter()
            chart_path = output_dir / "performance.png"
            plotter.performance_chart(stats, output=chart_path)
            console.print(f"  [green]✓[/green] Chart: {chart_path}")
        except Exception as e:
            console.print(f"  [yellow]⚠ Chart generation failed: {e}[/yellow]")
    
    console.print("\n[bold green]Analysis complete![/bold green]\n")


if __name__ == "__main__":
    main()
