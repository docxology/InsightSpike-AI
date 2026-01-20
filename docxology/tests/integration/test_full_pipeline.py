"""Integration tests for full pipeline execution.

Tests end-to-end workflows using real methods from insightspike.
"""

from __future__ import annotations

from pathlib import Path

import pytest


class TestDiscoveryIntegration:
    """Integration tests for method discovery."""
    
    @pytest.mark.slow
    def test_scan_insightspike_package(self, repo_root):
        """Test scanning the real insightspike package."""
        import sys
        sys.path.insert(0, str(repo_root / "src"))
        
        from docxology.discovery import ModuleScanner
        
        scanner = ModuleScanner(
            exclude_patterns=["deprecated", "experimental", "tests", "__pycache__"],
            max_depth=5,
        )
        
        methods = scanner.scan_package("insightspike")
        
        # Should discover many methods
        assert len(methods) > 10, f"Expected >10 methods, found {len(methods)}"
        
        # Should have functions and classes
        types = {m["type"] for m in methods}
        assert "function" in types or "class" in types


class TestScriptRunnerIntegration:
    """Integration tests for script runner."""
    
    def test_list_real_scripts(self, scripts_dir):
        """Test listing real scripts from repository."""
        if not scripts_dir.exists():
            pytest.skip("Scripts directory not found")
        
        from docxology.runners import ScriptRunner
        
        runner = ScriptRunner(scripts_dir=scripts_dir)
        scripts = runner.list_scripts()
        
        assert len(scripts) > 0, "Expected to find scripts"
        
        # Verify they are Python files
        for script in scripts[:5]:
            assert script.suffix == ".py"
    
    def test_get_script_info(self, scripts_dir):
        """Test getting script info."""
        if not scripts_dir.exists():
            pytest.skip("Scripts directory not found")
        
        from docxology.runners import ScriptRunner
        
        runner = ScriptRunner(scripts_dir=scripts_dir)
        scripts = runner.list_scripts()
        
        if scripts:
            info = runner.get_script_info(scripts[0].name)
            assert info["exists"]
            assert "path" in info


class TestPipelineIntegration:
    """Integration tests for pipeline execution."""
    
    def test_discovery_analysis_pipeline(self, repo_root):
        """Test a pipeline that discovers methods and analyzes them."""
        import sys
        sys.path.insert(0, str(repo_root / "src"))
        
        from docxology.orchestrator import Pipeline
        from docxology.discovery import ModuleScanner
        
        def discover_stage(ctx):
            scanner = ModuleScanner(max_depth=3)
            # Just scan a small subset for speed
            try:
                methods = scanner.scan_package("insightspike.algorithms")
            except ImportError:
                methods = []
            return {"methods": methods, "count": len(methods)}
        
        def analyze_stage(ctx):
            discovery_result = ctx.get("stage_discover", {})
            count = discovery_result.get("count", 0)
            return {
                "analysis": "complete",
                "method_count": count,
                "has_methods": count > 0,
            }
        
        pipeline = Pipeline("discovery_analysis")
        pipeline.add_stage("discover", discover_stage)
        pipeline.add_stage("analyze", analyze_stage, depends_on=["discover"])
        
        result = pipeline.run()
        
        assert result["success"]
        assert "discover" in result["stages"]
        assert "analyze" in result["stages"]


class TestAnalysisVisualizationIntegration:
    """Integration tests for analysis and visualization."""
    
    def test_analyze_and_export(self, tmp_path, sample_results):
        """Test analyzing results and exporting."""
        from docxology.analysis import ResultsAnalyzer
        from docxology.visualization import Exporter, Plotter
        
        # Create sample result files
        import json
        for i in range(3):
            result = {
                "run": i,
                "success": True,
                "accuracy": 0.9 + i * 0.02,
                "duration": 1.0 + i * 0.5,
            }
            (tmp_path / f"result_{i}.json").write_text(json.dumps(result))
        
        # Analyze
        analyzer = ResultsAnalyzer()
        stats = analyzer.analyze_directory(tmp_path)
        
        assert stats["count"] == 3
        
        # Export
        exporter = Exporter(timestamp_files=False)
        
        json_path = exporter.to_json(stats, tmp_path / "analysis.json")
        html_path = exporter.to_html(stats, tmp_path / "analysis.html")
        md_path = exporter.to_markdown(stats, tmp_path / "analysis.md")
        
        assert json_path.exists()
        assert html_path.exists()
        assert md_path.exists()
    
    def test_full_report_generation(self, tmp_path, sample_results):
        """Test full report generation."""
        from docxology.visualization import Reporter
        
        reporter = Reporter(output_dir=tmp_path, generate_plots=False)
        
        outputs = reporter.generate("test_report", sample_results)
        
        assert "json" in outputs
        assert outputs["json"].exists()
