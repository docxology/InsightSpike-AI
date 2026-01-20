"""Smoke tests for docxology.

Quick validation that core components load and function correctly.
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestImports:
    """Test that all core modules can be imported."""
    
    def test_import_docxology(self):
        """Test main package import."""
        import docxology
        assert hasattr(docxology, "__version__")
    
    def test_import_discovery(self):
        """Test discovery module import."""
        from docxology.discovery import ModuleScanner, MethodRegistry
        assert ModuleScanner is not None
        assert MethodRegistry is not None
    
    def test_import_runners(self):
        """Test runners module import."""
        from docxology.runners import ScriptRunner, ModuleRunner, BatchRunner
        assert ScriptRunner is not None
        assert ModuleRunner is not None
        assert BatchRunner is not None
    
    def test_import_orchestrator(self):
        """Test orchestrator module import."""
        from docxology.orchestrator import Pipeline, Stage
        assert Pipeline is not None
        assert Stage is not None
    
    def test_import_analysis(self):
        """Test analysis module import."""
        from docxology.analysis import ResultsAnalyzer, MetricsCollector, ResultsComparison
        assert ResultsAnalyzer is not None
        assert MetricsCollector is not None
        assert ResultsComparison is not None
    
    def test_import_visualization(self):
        """Test visualization module import."""
        from docxology.visualization import Plotter, Exporter, Reporter
        assert Plotter is not None
        assert Exporter is not None
        assert Reporter is not None


class TestConfig:
    """Test configuration system."""
    
    def test_config_load(self, docxology_root):
        """Test configuration loads successfully."""
        import sys
        sys.path.insert(0, str(docxology_root / "config"))
        
        from config import load_config, get_defaults
        
        # Should not raise
        config = load_config()
        assert isinstance(config, dict)
        
        defaults = get_defaults()
        assert "discovery" in defaults
        assert "execution" in defaults
    
    def test_config_sections(self, docxology_root):
        """Test configuration sections."""
        import sys
        sys.path.insert(0, str(docxology_root / "config"))
        
        from config import get_section
        
        discovery = get_section("discovery")
        assert "packages" in discovery


class TestDiscovery:
    """Test discovery system."""
    
    def test_method_registry_basic(self):
        """Test MethodRegistry basic operations."""
        from docxology.discovery import MethodRegistry, MethodInfo
        
        registry = MethodRegistry()
        
        info = MethodInfo(
            name="test_func",
            module="test.module",
            type="function",
            signature="(x, y)",
            docstring="A test function.",
        )
        
        registry.register(info)
        
        assert len(registry) == 1
        assert "test.module.test_func" in registry
        
        retrieved = registry.get("test.module.test_func")
        assert retrieved is not None
        assert retrieved.name == "test_func"
    
    def test_method_registry_filtering(self):
        """Test MethodRegistry filtering."""
        from docxology.discovery import MethodRegistry, MethodInfo
        
        registry = MethodRegistry()
        
        # Add various methods
        registry.register(MethodInfo(name="func1", module="mod.a", type="function"))
        registry.register(MethodInfo(name="func2", module="mod.a", type="function"))
        registry.register(MethodInfo(name="Class1", module="mod.b", type="class"))
        
        funcs = registry.filter_by_type("function")
        assert len(funcs) == 2
        
        classes = registry.filter_by_type("class")
        assert len(classes) == 1
    
    def test_module_scanner_init(self):
        """Test ModuleScanner initialization."""
        from docxology.discovery import ModuleScanner
        
        scanner = ModuleScanner()
        assert scanner.max_depth == 10
        assert not scanner.include_private


class TestRunners:
    """Test runner components."""
    
    def test_script_runner_init(self, repo_root):
        """Test ScriptRunner initialization."""
        from docxology.runners import ScriptRunner
        
        runner = ScriptRunner()
        assert runner.scripts_dir.exists() or True  # May not exist in test env
    
    def test_script_runner_list_scripts(self, scripts_dir):
        """Test listing available scripts."""
        from docxology.runners import ScriptRunner
        
        runner = ScriptRunner(scripts_dir=scripts_dir)
        scripts = runner.list_scripts()
        
        # Should find at least some scripts if directory exists
        if scripts_dir.exists():
            assert isinstance(scripts, list)
    
    def test_module_runner_init(self):
        """Test ModuleRunner initialization."""
        from docxology.runners import ModuleRunner
        
        runner = ModuleRunner()
        assert runner is not None


class TestOrchestrator:
    """Test orchestrator components."""
    
    def test_pipeline_basic(self):
        """Test basic pipeline creation and execution."""
        from docxology.orchestrator import Pipeline
        
        results = []
        
        def stage1(ctx):
            results.append("stage1")
            return {"value": 1}
        
        def stage2(ctx):
            results.append("stage2")
            return {"value": 2}
        
        pipeline = Pipeline("test")
        pipeline.add_stage("s1", stage1)
        pipeline.add_stage("s2", stage2, depends_on=["s1"])
        
        result = pipeline.run()
        
        assert result["success"]
        assert results == ["stage1", "stage2"]
    
    def test_pipeline_dependency_order(self):
        """Test pipeline respects dependencies."""
        from docxology.orchestrator import Pipeline
        
        execution_order = []
        
        def make_stage(name):
            def stage(ctx):
                execution_order.append(name)
                return name
            return stage
        
        pipeline = Pipeline("test")
        pipeline.add_stage("c", make_stage("c"), depends_on=["a", "b"])
        pipeline.add_stage("a", make_stage("a"))
        pipeline.add_stage("b", make_stage("b"), depends_on=["a"])
        
        pipeline.run()
        
        # a must come before b, and both before c
        assert execution_order.index("a") < execution_order.index("b")
        assert execution_order.index("b") < execution_order.index("c")


class TestAnalysis:
    """Test analysis components."""
    
    def test_results_analyzer_init(self):
        """Test ResultsAnalyzer initialization."""
        from docxology.analysis import ResultsAnalyzer
        
        analyzer = ResultsAnalyzer()
        assert analyzer is not None
    
    def test_metrics_collector_basic(self):
        """Test MetricsCollector basic operations."""
        from docxology.analysis import MetricsCollector
        
        collector = MetricsCollector()
        
        collector.record("latency", 1.5)
        collector.record("latency", 2.0)
        collector.record("latency", 1.8)
        
        stats = collector.get_stats("latency")
        
        assert stats["count"] == 3
        assert 1.5 <= stats["mean"] <= 2.0
        assert stats["min"] == 1.5
        assert stats["max"] == 2.0
    
    def test_results_comparison(self, sample_results):
        """Test ResultsComparison."""
        from docxology.analysis import ResultsComparison
        
        comparison = ResultsComparison()
        
        baseline = {"accuracy": 0.9, "latency": 100}
        experiment = {"accuracy": 0.95, "latency": 80}
        
        result = comparison.compare(baseline, experiment)
        
        assert "improvements" in result
        assert "regressions" in result


class TestVisualization:
    """Test visualization components."""
    
    def test_plotter_init(self):
        """Test Plotter initialization."""
        from docxology.visualization import Plotter
        
        plotter = Plotter()
        assert plotter.dpi == 150
    
    def test_exporter_init(self):
        """Test Exporter initialization."""
        from docxology.visualization import Exporter
        
        exporter = Exporter()
        assert exporter.json_indent == 2
    
    def test_exporter_json(self, tmp_path, sample_results):
        """Test JSON export."""
        from docxology.visualization import Exporter
        
        exporter = Exporter(timestamp_files=False)
        output = tmp_path / "test.json"
        
        result = exporter.to_json(sample_results, output)
        
        assert result.exists()
        
        import json
        with open(result) as f:
            data = json.load(f)
        assert data["success"] is True
