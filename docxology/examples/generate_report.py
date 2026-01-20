#!/usr/bin/env python
"""Generate a comprehensive report from execution results.

This example demonstrates using the full docxology pipeline to
discover methods, execute scripts, analyze results, and generate reports.

Usage:
    python examples/generate_report.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add paths
DOCXOLOGY_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(DOCXOLOGY_ROOT / "src"))
sys.path.insert(0, str(DOCXOLOGY_ROOT.parent / "src"))

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from docxology.orchestrator import Pipeline
from docxology.discovery import ModuleScanner
from docxology.runners import ScriptRunner
from docxology.analysis import MetricsCollector
from docxology.visualization import Reporter


def discovery_stage(ctx: dict) -> dict:
    """Discover methods from insightspike."""
    scanner = ModuleScanner(max_depth=3)
    try:
        methods = scanner.scan_package("insightspike")
    except ImportError:
        methods = []
    
    return {
        "methods": methods,
        "count": len(methods),
        "stats": scanner.get_registry().get_stats(),
    }


def scripts_stage(ctx: dict) -> dict:
    """List available scripts."""
    runner = ScriptRunner()
    scripts = runner.list_scripts()
    
    examples_runner = ScriptRunner(scripts_dir=DOCXOLOGY_ROOT.parent / "examples")
    examples = examples_runner.list_scripts()
    scripts.extend(examples)
    
    script_info = []
    for script in scripts[:10]:
        info = runner.get_script_info(str(script))
        script_info.append(info)
    
    return {
        "scripts": [s.name for s in scripts],
        "count": len(scripts),
        "info": script_info,
    }


def metrics_stage(ctx: dict) -> dict:
    """Collect and compute metrics."""
    collector = MetricsCollector()
    
    # Record discovery metrics
    discovery = ctx.get("stage_discovery", {})
    collector.record("discovered_methods", discovery.get("count", 0))
    collector.record("discovered_modules", discovery.get("stats", {}).get("modules", 0))
    
    # Record scripts metrics
    scripts = ctx.get("stage_scripts", {})
    collector.record("available_scripts", scripts.get("count", 0))
    
    return {
        "collected": collector.count(),
        "stats": collector.get_all_stats(),
    }


def report_stage(ctx: dict) -> dict:
    """Generate final report."""
    # Collect all stage results
    report_data = {
        "discovery": ctx.get("stage_discovery", {}),
        "scripts": ctx.get("stage_scripts", {}),
        "metrics": ctx.get("stage_metrics", {}),
    }
    
    # Add summary
    report_data["summary"] = {
        "methods_discovered": report_data["discovery"].get("count", 0),
        "scripts_available": report_data["scripts"].get("count", 0),
        "metrics_collected": report_data["metrics"].get("collected", 0),
    }
    
    # Generate report
    output_dir = DOCXOLOGY_ROOT / "output" / "reports"
    reporter = Reporter(output_dir=output_dir, generate_plots=False)
    
    outputs = reporter.generate("comprehensive_report", report_data)
    
    return {
        "outputs": {k: str(v) for k, v in outputs.items()},
        "summary": report_data["summary"],
    }


def main() -> None:
    """Main entry point."""
    console = Console()
    
    console.print("\n[bold blue]Docxology — Comprehensive Report Generation[/bold blue]\n")
    
    # Create pipeline
    pipeline = Pipeline("comprehensive_report")
    pipeline.add_stage("discovery", discovery_stage)
    pipeline.add_stage("scripts", scripts_stage)
    pipeline.add_stage("metrics", metrics_stage, depends_on=["discovery", "scripts"])
    pipeline.add_stage("report", report_stage, depends_on=["metrics"])
    
    # Execute with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Running pipeline...", total=None)
        
        def on_progress(name: str, status):
            progress.update(task, description=f"Stage: {name} ({status.value})")
        
        result = pipeline.run(progress_callback=on_progress)
    
    # Display results
    if result["success"]:
        console.print("\n[bold green]✓ Pipeline completed successfully![/bold green]\n")
        
        # Show stage summary
        for stage_name, stage_result in result["stages"].items():
            status = stage_result["status"]
            duration = stage_result["duration"]
            
            if status == "completed":
                console.print(f"  [green]✓[/green] {stage_name}: {duration:.2f}s")
            else:
                console.print(f"  [red]✗[/red] {stage_name}: {status}")
        
        # Show summary
        report_result = pipeline.get_result("report")
        if report_result:
            console.print("\n[bold]Summary:[/bold]")
            for key, value in report_result.get("summary", {}).items():
                console.print(f"  • {key}: {value}")
            
            console.print("\n[bold]Generated Files:[/bold]")
            for fmt, path in report_result.get("outputs", {}).items():
                console.print(f"  • {fmt}: {path}")
    else:
        console.print("\n[bold red]✗ Pipeline failed[/bold red]")
        for stage_name, stage_result in result["stages"].items():
            if stage_result.get("error"):
                console.print(f"  [red]{stage_name}[/red]: {stage_result['error']}")
    
    console.print(f"\n[dim]Total duration: {result['duration']:.2f}s[/dim]\n")


if __name__ == "__main__":
    main()
