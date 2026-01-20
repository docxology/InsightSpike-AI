#!/usr/bin/env python
"""Discover and run all methods from the insightspike package.

This example demonstrates using the discovery and runner systems
to automatically find and execute methods from the parent package.

Usage:
    python examples/run_all_methods.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add paths
DOCXOLOGY_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(DOCXOLOGY_ROOT / "src"))
sys.path.insert(0, str(DOCXOLOGY_ROOT.parent / "src"))

from rich.console import Console
from rich.table import Table

from docxology.discovery import ModuleScanner
from docxology.runners import ScriptRunner


def main() -> None:
    """Main entry point."""
    console = Console()
    
    console.print("\n[bold blue]Docxology — Method Discovery[/bold blue]\n")
    
    # 1. Discover methods
    console.print("[cyan]Scanning insightspike package...[/cyan]")
    
    scanner = ModuleScanner(
        exclude_patterns=["deprecated", "experimental", "tests", "__pycache__"],
        max_depth=4,
    )
    
    try:
        methods = scanner.scan_package("insightspike")
    except ImportError as e:
        console.print(f"[red]Failed to import insightspike: {e}[/red]")
        console.print("[dim]Make sure the parent package is installed.[/dim]")
        return
    
    console.print(f"[green]✓ Discovered {len(methods)} methods[/green]\n")
    
    # 2. Show summary by type
    stats = scanner.get_registry().get_stats()
    
    table = Table(title="Discovery Summary")
    table.add_column("Type", style="cyan")
    table.add_column("Count", style="green")
    
    for type_name, count in stats.get("by_type", {}).items():
        table.add_row(type_name, str(count))
    
    console.print(table)
    
    # 3. Show sample methods
    console.print("\n[bold]Sample Methods:[/bold]\n")
    
    sample_table = Table()
    sample_table.add_column("Module", style="dim")
    sample_table.add_column("Name", style="cyan")
    sample_table.add_column("Type", style="magenta")
    
    for method in methods[:15]:
        sample_table.add_row(
            method.get("module", "").split(".")[-1],
            method.get("name", ""),
            method.get("type", ""),
        )
    
    console.print(sample_table)
    
    if len(methods) > 15:
        console.print(f"\n[dim]... and {len(methods) - 15} more[/dim]")
    
    # 4. List available scripts
    console.print("\n[bold blue]Available Scripts[/bold blue]\n")
    
    runner = ScriptRunner()
    scripts = runner.list_scripts()
    
    if scripts:
        script_table = Table()
        script_table.add_column("Script", style="green")
        script_table.add_column("Size", style="dim")
        
        for script in scripts[:10]:
            size = script.stat().st_size
            script_table.add_row(script.name, f"{size:,} bytes")
        
        console.print(script_table)
        
        if len(scripts) > 10:
            console.print(f"\n[dim]... and {len(scripts) - 10} more[/dim]")
    else:
        console.print("[dim]No scripts found[/dim]")
    
    console.print("\n[bold green]Discovery complete![/bold green]\n")


if __name__ == "__main__":
    main()
