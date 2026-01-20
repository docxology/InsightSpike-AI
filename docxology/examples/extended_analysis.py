#!/usr/bin/env python
"""Extended comprehensive analysis of all InsightSpike methods.

This script performs deep analysis of the entire insightspike package,
including:
- Full method discovery with signatures and documentation
- Module-level statistics and dependency mapping
- Type distribution analysis
- Complexity metrics
- Cross-module relationship analysis
- Comprehensive visualization suite
"""

from __future__ import annotations

import json
import logging
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

# Add paths
DOCXOLOGY_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(DOCXOLOGY_ROOT / "src"))
sys.path.insert(0, str(DOCXOLOGY_ROOT.parent / "src"))

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from docxology.discovery import ModuleScanner, MethodRegistry
from docxology.runners import ScriptRunner
from docxology.analysis import MetricsCollector, ResultsAnalyzer, ResultsComparison
from docxology.visualization import Plotter, Exporter, Reporter
from docxology.orchestrator import Pipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(DOCXOLOGY_ROOT / 'output/logs/extended_analysis.log')
    ]
)
logger = logging.getLogger('docxology.extended_analysis')
console = Console()


def analyze_module_structure(methods: list[dict]) -> dict:
    """Analyze module-level structure and relationships."""
    modules = defaultdict(list)
    for m in methods:
        module = m.get('module', '')
        modules[module].append(m)
    
    # Compute module statistics
    module_stats = {}
    for mod, items in modules.items():
        classes = [i for i in items if i.get('type') == 'class']
        functions = [i for i in items if i.get('type') == 'function']
        module_stats[mod] = {
            'total': len(items),
            'classes': len(classes),
            'functions': len(functions),
            'class_names': [c['name'] for c in classes],
            'function_names': [f['name'] for f in functions],
        }
    
    return {
        'module_count': len(modules),
        'modules': module_stats,
        'top_modules': sorted(module_stats.items(), key=lambda x: x[1]['total'], reverse=True)[:20],
    }


def analyze_naming_patterns(methods: list[dict]) -> dict:
    """Analyze naming conventions and patterns."""
    prefixes = Counter()
    suffixes = Counter()
    name_lengths = []
    
    for m in methods:
        name = m.get('name', '')
        name_lengths.append(len(name))
        
        # Common prefixes
        for prefix in ['get_', 'set_', 'is_', 'has_', 'create_', 'build_', 'compute_', 'calculate_', 'process_', 'validate_']:
            if name.lower().startswith(prefix):
                prefixes[prefix] += 1
                break
        
        # Common suffixes
        for suffix in ['_manager', '_factory', '_builder', '_handler', '_processor', '_config', '_result', '_error']:
            if name.lower().endswith(suffix):
                suffixes[suffix] += 1
                break
    
    return {
        'prefix_distribution': dict(prefixes.most_common(15)),
        'suffix_distribution': dict(suffixes.most_common(15)),
        'name_length_stats': {
            'mean': float(np.mean(name_lengths)),
            'std': float(np.std(name_lengths)),
            'min': int(np.min(name_lengths)),
            'max': int(np.max(name_lengths)),
            'median': float(np.median(name_lengths)),
        },
    }


def analyze_documentation_coverage(methods: list[dict]) -> dict:
    """Analyze documentation coverage."""
    documented = 0
    undocumented = 0
    doc_lengths = []
    
    for m in methods:
        doc = m.get('docstring', '')
        if doc and len(doc.strip()) > 0:
            documented += 1
            doc_lengths.append(len(doc))
        else:
            undocumented += 1
    
    return {
        'documented_count': documented,
        'undocumented_count': undocumented,
        'coverage_percentage': 100 * documented / len(methods) if methods else 0,
        'doc_length_stats': {
            'mean': float(np.mean(doc_lengths)) if doc_lengths else 0,
            'max': int(np.max(doc_lengths)) if doc_lengths else 0,
        },
    }


def analyze_signature_complexity(methods: list[dict]) -> dict:
    """Analyze method signature complexity."""
    param_counts = []
    simple_signatures = 0
    complex_signatures = 0
    
    for m in methods:
        sig = m.get('signature', '()')
        # Count parameters (rough estimation)
        params = sig.count(',') + (1 if sig not in ['()', ''] else 0)
        param_counts.append(params)
        
        if params <= 3:
            simple_signatures += 1
        else:
            complex_signatures += 1
    
    return {
        'param_count_stats': {
            'mean': float(np.mean(param_counts)),
            'std': float(np.std(param_counts)),
            'max': int(np.max(param_counts)),
        },
        'simple_signatures': simple_signatures,
        'complex_signatures': complex_signatures,
        'complexity_ratio': complex_signatures / len(methods) if methods else 0,
    }


def analyze_type_hierarchy(methods: list[dict]) -> dict:
    """Analyze class/type hierarchy patterns."""
    base_classes = Counter()
    mixin_count = 0
    abstract_count = 0
    factory_count = 0
    
    for m in methods:
        if m.get('type') != 'class':
            continue
        
        name = m.get('name', '')
        
        if 'Mixin' in name:
            mixin_count += 1
        if 'Abstract' in name or 'Base' in name:
            abstract_count += 1
        if 'Factory' in name:
            factory_count += 1
        
        # Detect common patterns
        for pattern in ['Agent', 'Manager', 'Handler', 'Provider', 'Processor', 'Builder', 'Config', 'Result', 'Error', 'Interface']:
            if pattern in name:
                base_classes[pattern] += 1
    
    return {
        'pattern_distribution': dict(base_classes.most_common(15)),
        'mixin_count': mixin_count,
        'abstract_count': abstract_count,
        'factory_count': factory_count,
    }


def create_comprehensive_visualizations(analysis_data: dict, output_dir: Path) -> list[Path]:
    """Generate comprehensive visualization suite."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plotter = Plotter(dpi=150)
    generated_files = []
    
    # 1. Module size distribution
    module_data = analysis_data.get('module_structure', {})
    top_modules = module_data.get('top_modules', [])[:15]
    if top_modules:
        module_sizes = {m[0].split('.')[-1][:20]: m[1]['total'] for m in top_modules}
        fig = plotter.bar_chart(
            module_sizes,
            title='Top 15 Modules by Method Count',
            xlabel='Module',
            ylabel='Method Count',
            output=output_dir / 'module_distribution.png'
        )
        generated_files.append(output_dir / 'module_distribution.png')
    
    # 2. Type distribution pie-style bar
    type_dist = analysis_data.get('stats', {}).get('by_type', {})
    if type_dist:
        fig = plotter.bar_chart(
            type_dist,
            title='Method Type Distribution',
            ylabel='Count',
            output=output_dir / 'type_distribution.png',
            color='coral'
        )
        generated_files.append(output_dir / 'type_distribution.png')
    
    # 3. Documentation coverage
    doc_data = analysis_data.get('documentation', {})
    if doc_data:
        coverage = {
            'Documented': doc_data.get('documented_count', 0),
            'Undocumented': doc_data.get('undocumented_count', 0),
        }
        fig = plotter.bar_chart(
            coverage,
            title=f"Documentation Coverage ({doc_data.get('coverage_percentage', 0):.1f}%)",
            ylabel='Count',
            output=output_dir / 'documentation_coverage.png',
            color='seagreen'
        )
        generated_files.append(output_dir / 'documentation_coverage.png')
    
    # 4. Naming pattern distribution
    naming = analysis_data.get('naming_patterns', {})
    prefixes = naming.get('prefix_distribution', {})
    if prefixes:
        fig = plotter.bar_chart(
            prefixes,
            title='Method Name Prefixes',
            xlabel='Prefix',
            ylabel='Count',
            output=output_dir / 'naming_prefixes.png',
            color='mediumpurple'
        )
        generated_files.append(output_dir / 'naming_prefixes.png')
    
    # 5. Class pattern distribution
    hierarchy = analysis_data.get('type_hierarchy', {})
    patterns = hierarchy.get('pattern_distribution', {})
    if patterns:
        fig = plotter.bar_chart(
            patterns,
            title='Class Naming Patterns',
            xlabel='Pattern',
            ylabel='Count',
            output=output_dir / 'class_patterns.png',
            color='steelblue'
        )
        generated_files.append(output_dir / 'class_patterns.png')
    
    # 6. Signature complexity comparison
    sig_data = analysis_data.get('signature_complexity', {})
    if sig_data:
        complexity = {
            'Simple (≤3 params)': sig_data.get('simple_signatures', 0),
            'Complex (>3 params)': sig_data.get('complex_signatures', 0),
        }
        fig = plotter.bar_chart(
            complexity,
            title='Signature Complexity Distribution',
            ylabel='Count',
            output=output_dir / 'signature_complexity.png',
            color='darkorange'
        )
        generated_files.append(output_dir / 'signature_complexity.png')
    
    return generated_files


def main():
    """Main analysis entry point."""
    console.print("\n[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]")
    console.print("[bold blue]       DOCXOLOGY EXTENDED COMPREHENSIVE ANALYSIS              [/bold blue]")
    console.print("[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]")
    console.print(f"Started: {datetime.now().isoformat()}\n")
    
    output_dir = DOCXOLOGY_ROOT / 'output'
    
    # Phase 1: Discovery
    console.print("[bold cyan]PHASE 1: Deep Method Discovery[/bold cyan]")
    console.print("─" * 60)
    
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console) as progress:
        task = progress.add_task("Scanning insightspike package...", total=None)
        
        scanner = ModuleScanner(max_depth=10, include_private=False)
        methods = scanner.scan_package('insightspike')
        registry = scanner.get_registry()
        stats = registry.get_stats()
    
    console.print(f"  ✓ Discovered [green]{len(methods)}[/green] methods")
    console.print(f"  ✓ Classes: [cyan]{stats['by_type'].get('class', 0)}[/cyan]")
    console.print(f"  ✓ Functions: [cyan]{stats['by_type'].get('function', 0)}[/cyan]")
    console.print(f"  ✓ Modules: [cyan]{stats['modules']}[/cyan]\n")
    
    # Phase 2: Deep Analysis
    console.print("[bold cyan]PHASE 2: Deep Structural Analysis[/bold cyan]")
    console.print("─" * 60)
    
    analysis_data = {
        'timestamp': datetime.now().isoformat(),
        'total_methods': len(methods),
        'stats': stats,
    }
    
    # Module structure analysis
    console.print("  Analyzing module structure...")
    analysis_data['module_structure'] = analyze_module_structure(methods)
    console.print(f"    ✓ Analyzed [green]{analysis_data['module_structure']['module_count']}[/green] modules")
    
    # Naming patterns
    console.print("  Analyzing naming patterns...")
    analysis_data['naming_patterns'] = analyze_naming_patterns(methods)
    console.print(f"    ✓ Mean name length: [cyan]{analysis_data['naming_patterns']['name_length_stats']['mean']:.1f}[/cyan] chars")
    
    # Documentation coverage
    console.print("  Analyzing documentation coverage...")
    analysis_data['documentation'] = analyze_documentation_coverage(methods)
    console.print(f"    ✓ Coverage: [green]{analysis_data['documentation']['coverage_percentage']:.1f}%[/green]")
    
    # Signature complexity
    console.print("  Analyzing signature complexity...")
    analysis_data['signature_complexity'] = analyze_signature_complexity(methods)
    console.print(f"    ✓ Mean params: [cyan]{analysis_data['signature_complexity']['param_count_stats']['mean']:.1f}[/cyan]")
    
    # Type hierarchy
    console.print("  Analyzing type hierarchy patterns...")
    analysis_data['type_hierarchy'] = analyze_type_hierarchy(methods)
    console.print(f"    ✓ Factory classes: [cyan]{analysis_data['type_hierarchy']['factory_count']}[/cyan]")
    console.print()
    
    # Phase 3: Script Analysis
    console.print("[bold cyan]PHASE 3: Script Inventory & Analysis[/bold cyan]")
    console.print("─" * 60)
    
    runner = ScriptRunner()
    scripts = runner.list_scripts()
    
    script_analysis = {
        'count': len(scripts),
        'scripts': [],
        'total_size': 0,
    }
    
    for script in scripts:
        info = runner.get_script_info(script.name)
        script_analysis['scripts'].append(info)
        script_analysis['total_size'] += info.get('size', 0)
    
    console.print(f"  ✓ Scripts found: [green]{len(scripts)}[/green]")
    console.print(f"  ✓ Total size: [cyan]{script_analysis['total_size']:,}[/cyan] bytes")
    
    # Show script categories
    analysis_scripts = [s for s in scripts if 'analyze' in s.name or 'aggregate' in s.name]
    run_scripts = [s for s in scripts if 'run_' in s.name]
    console.print(f"  ✓ Analysis scripts: [cyan]{len(analysis_scripts)}[/cyan]")
    console.print(f"  ✓ Runner scripts: [cyan]{len(run_scripts)}[/cyan]")
    
    analysis_data['scripts'] = script_analysis
    console.print()
    
    # Phase 4: Visualization
    console.print("[bold cyan]PHASE 4: Comprehensive Visualization[/bold cyan]")
    console.print("─" * 60)
    
    figures_dir = output_dir / 'figures' / 'extended'
    generated_figures = create_comprehensive_visualizations(analysis_data, figures_dir)
    
    for fig in generated_figures:
        console.print(f"  ✓ Generated: [green]{fig.name}[/green]")
    
    console.print()
    
    # Phase 5: Export
    console.print("[bold cyan]PHASE 5: Multi-Format Export[/bold cyan]")
    console.print("─" * 60)
    
    exporter = Exporter(timestamp_files=False)
    reports_dir = output_dir / 'reports' / 'extended'
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Full analysis export
    json_path = exporter.to_json(analysis_data, reports_dir / 'extended_analysis.json')
    console.print(f"  ✓ JSON: [green]{json_path.name}[/green]")
    
    html_path = exporter.to_html(analysis_data, reports_dir / 'extended_analysis.html', title='InsightSpike Extended Analysis')
    console.print(f"  ✓ HTML: [green]{html_path.name}[/green]")
    
    md_path = exporter.to_markdown(analysis_data, reports_dir / 'extended_analysis.md', title='InsightSpike Extended Analysis')
    console.print(f"  ✓ Markdown: [green]{md_path.name}[/green]")
    
    csv_path = exporter.to_csv(analysis_data, reports_dir / 'extended_analysis.csv')
    console.print(f"  ✓ CSV: [green]{csv_path.name}[/green]")
    
    # Method inventory export
    method_inventory = {
        'timestamp': datetime.now().isoformat(),
        'total': len(methods),
        'methods': methods,
    }
    inventory_path = exporter.to_json(method_inventory, reports_dir / 'method_inventory.json')
    console.print(f"  ✓ Method Inventory: [green]{inventory_path.name}[/green]")
    
    console.print()
    
    # Summary
    console.print("[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]")
    console.print("[bold blue]                     EXECUTION SUMMARY                         [/bold blue]")
    console.print("[bold blue]═══════════════════════════════════════════════════════════════[/bold blue]")
    
    summary_table = Table(show_header=True, header_style="bold cyan")
    summary_table.add_column("Metric", style="dim")
    summary_table.add_column("Value", justify="right", style="green")
    
    summary_table.add_row("Methods Discovered", str(len(methods)))
    summary_table.add_row("Classes", str(stats['by_type'].get('class', 0)))
    summary_table.add_row("Functions", str(stats['by_type'].get('function', 0)))
    summary_table.add_row("Modules Analyzed", str(analysis_data['module_structure']['module_count']))
    summary_table.add_row("Documentation Coverage", f"{analysis_data['documentation']['coverage_percentage']:.1f}%")
    summary_table.add_row("Scripts Inventoried", str(len(scripts)))
    summary_table.add_row("Figures Generated", str(len(generated_figures)))
    summary_table.add_row("Reports Exported", "5")
    
    console.print(summary_table)
    console.print(f"\nCompleted: {datetime.now().isoformat()}")
    console.print("[bold green]✓ Extended analysis complete![/bold green]\n")
    
    logger.info(f"Extended analysis complete: {len(methods)} methods, {len(generated_figures)} figures")
    
    return analysis_data


if __name__ == "__main__":
    main()
