"""CLI entry point for docxology.

Provides command-line interface for discovery, execution, analysis, and visualization.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.table import Table

# Ensure parent package is importable
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "src"))

app = typer.Typer(
    name="docx",
    help="Docxology — InsightSpike-AI Sidecar Framework",
    add_completion=False,
)
console = Console()


def setup_logging(level: str = "INFO") -> None:
    """Configure logging with rich handler."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@app.command()
def discover(
    package: str = typer.Option("insightspike", help="Package to scan"),
    output: Optional[Path] = typer.Option(None, help="Output file for results"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Discover all methods in the target package."""
    setup_logging("DEBUG" if verbose else "INFO")
    
    from .discovery import ModuleScanner
    
    console.print(f"\n[bold blue]🔍 Discovering methods in '{package}'[/bold blue]\n")
    
    scanner = ModuleScanner()
    methods = scanner.scan_package(package)
    
    # Display results
    table = Table(title=f"Discovered Methods ({len(methods)} total)")
    table.add_column("Module", style="cyan")
    table.add_column("Type", style="magenta")
    table.add_column("Name", style="green")
    table.add_column("Signature", style="dim")
    
    for method in methods[:50]:  # Show first 50
        table.add_row(
            method.get("module", ""),
            method.get("type", ""),
            method.get("name", ""),
            method.get("signature", "")[:40] + "..." if len(method.get("signature", "")) > 40 else method.get("signature", ""),
        )
    
    console.print(table)
    
    if len(methods) > 50:
        console.print(f"\n[dim]... and {len(methods) - 50} more[/dim]")
    
    if output:
        import json
        output.write_text(json.dumps(methods, indent=2))
        console.print(f"\n[green]✓ Results saved to {output}[/green]")


@app.command()
def run(
    script: str = typer.Argument(..., help="Script to execute"),
    env: Optional[list[str]] = typer.Option(None, "--env", "-e", help="Environment variable (KEY=VALUE)"),
    timeout: int = typer.Option(300, help="Execution timeout in seconds"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Execute a script from the scripts directory."""
    setup_logging("DEBUG" if verbose else "INFO")
    
    from .runners import ScriptRunner
    
    console.print(f"\n[bold blue]🚀 Executing: {script}[/bold blue]\n")
    
    # Parse environment variables
    env_dict = {}
    if env:
        for e in env:
            if "=" in e:
                k, v = e.split("=", 1)
                env_dict[k] = v
    
    runner = ScriptRunner()
    result = runner.execute(script, env=env_dict, timeout=timeout)
    
    if result["success"]:
        console.print(f"[green]✓ Script completed successfully[/green]")
        console.print(f"  Duration: {result['duration']:.2f}s")
    else:
        console.print(f"[red]✗ Script failed[/red]")
        console.print(f"  Return code: {result['returncode']}")
        if result.get("error"):
            console.print(f"  Error: {result['error']}")


@app.command()
def analyze(
    results_dir: Path = typer.Argument(..., help="Directory containing results"),
    output: Optional[Path] = typer.Option(None, help="Output file for analysis"),
    format: str = typer.Option("json", help="Output format (json, csv, html)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Analyze results from previous runs."""
    setup_logging("DEBUG" if verbose else "INFO")
    
    from .analysis import ResultsAnalyzer
    
    console.print(f"\n[bold blue]📊 Analyzing results in: {results_dir}[/bold blue]\n")
    
    analyzer = ResultsAnalyzer()
    stats = analyzer.analyze_directory(results_dir)
    
    # Display summary
    table = Table(title="Analysis Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    
    for key, value in stats.get("summary", {}).items():
        table.add_row(key, str(value))
    
    console.print(table)
    
    if output:
        from .visualization import Exporter
        exporter = Exporter()
        if format == "json":
            exporter.to_json(stats, output)
        elif format == "csv":
            exporter.to_csv(stats, output)
        elif format == "html":
            exporter.to_html(stats, output)
        console.print(f"\n[green]✓ Analysis saved to {output}[/green]")


@app.command()
def version() -> None:
    """Show version information."""
    from . import __version__
    console.print(f"[bold]docxology[/bold] v{__version__}")
    
    # Also show parent package version
    try:
        from insightspike import About
        console.print(f"[dim]InsightSpike-AI[/dim] v{About.VERSION}")
    except ImportError:
        console.print("[dim]InsightSpike-AI not installed[/dim]")


@app.command()
def info() -> None:
    """Show configuration and environment information."""
    from config import load_config, get_project_root, get_repo_root
    
    config = load_config()
    
    console.print("\n[bold]Docxology Information[/bold]\n")
    
    table = Table()
    table.add_column("Setting", style="cyan")
    table.add_column("Value", style="green")
    
    table.add_row("Project Root", str(get_project_root()))
    table.add_row("Repo Root", str(get_repo_root()))
    table.add_row("Config Loaded", "Yes" if config else "No")
    table.add_row("Parallel Execution", str(config.get("execution", {}).get("parallel", False)))
    table.add_row("Max Workers", str(config.get("execution", {}).get("max_workers", 1)))
    table.add_row("Log Level", config.get("logging", {}).get("level", "INFO"))
    
    console.print(table)


def main() -> None:
    """Main CLI entry point."""
    app()


if __name__ == "__main__":
    main()
