#!/usr/bin/env python3
"""Run-All Script for Docxology.

Comprehensive execution of all tests and methods in the docxology repository.
Runs docxology tests, method discovery, docxology examples, and parent repository examples.

Usage:
    python run_all.py                   # Run everything
    python run_all.py --tests           # Run tests only
    python run_all.py --discovery       # Run discovery only
    python run_all.py --docx-examples   # Run docxology examples only
    python run_all.py --parent-examples # Run parent repo examples only
    python run_all.py --quick           # Quick smoke tests only
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Setup paths
DOCXOLOGY_ROOT = Path(__file__).parent.absolute()
PARENT_ROOT = DOCXOLOGY_ROOT.parent
SRC_PATH = DOCXOLOGY_ROOT / "src"
DOCX_EXAMPLES_PATH = DOCXOLOGY_ROOT / "examples"
PARENT_EXAMPLES_PATH = PARENT_ROOT / "examples"
TESTS_PATH = DOCXOLOGY_ROOT / "tests"
OUTPUT_PATH = DOCXOLOGY_ROOT / "output"

sys.path.insert(0, str(SRC_PATH))
sys.path.insert(0, str(PARENT_ROOT / "src"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

# Parent examples in logical execution order
# 1. Basic/beginner examples first
# 2. Config and infrastructure
# 3. Feature demonstrations
# 4. Advanced simulations
# 5. Tests and LLM integration last
PARENT_EXAMPLES_ORDER = [
    "public_quick_start.py",     # Minimal public API usage
    "hello_insight.py",          # Tiny geDIG gauge demo
    "hello_gating.py",           # Minimal AG/DG gating behavior
    "config_examples.py",        # Configuration examples
    "playground.py",             # Interactive playground
    "test_analogy_spike.py",     # Analogy spike testing
    "simulate_econophysics.py",  # Econophysics simulation
    "simulate_eureka.py",        # Eureka simulation
    "test_local_llm.py",         # Local LLM testing (last - may timeout)
]


def run_command(
    cmd: list[str],
    cwd: Path = DOCXOLOGY_ROOT,
    timeout: int = 300,
    env: dict | None = None
) -> tuple[int, str]:
    """Run a command and return exit code and output."""
    logger.info(f"Running: {' '.join(cmd)}")
    import os
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=run_env
        )
        output = result.stdout + result.stderr
        return result.returncode, output
    except subprocess.TimeoutExpired:
        return 1, "Command timed out"
    except Exception as e:
        return 1, str(e)


def run_tests(quick: bool = False) -> dict:
    """Run pytest test suite."""
    logger.info("=" * 60)
    logger.info("RUNNING DOCXOLOGY TESTS")
    logger.info("=" * 60)
    
    if quick:
        cmd = ["python", "-m", "pytest", "tests/test_smoke.py", "-v", "--tb=short"]
    else:
        cmd = ["python", "-m", "pytest", "tests/", "-v", "--tb=short"]
    
    start = time.time()
    exit_code, output = run_command(cmd)
    duration = time.time() - start
    
    # Parse test results
    passed = output.count(" PASSED")
    failed = output.count(" FAILED")
    
    result = {
        "component": "docxology-tests",
        "passed": passed,
        "failed": failed,
        "exit_code": exit_code,
        "duration": duration,
        "success": exit_code == 0
    }
    
    if exit_code == 0:
        logger.info(f"✅ Tests: {passed} passed in {duration:.2f}s")
    else:
        logger.error(f"❌ Tests: {failed} failed, {passed} passed")
        print(output)
    
    return result


def run_discovery() -> dict:
    """Run method discovery on insightspike package."""
    logger.info("=" * 60)
    logger.info("RUNNING DISCOVERY")
    logger.info("=" * 60)
    
    start = time.time()
    
    try:
        from docxology.discovery import ModuleScanner
        
        scanner = ModuleScanner(
            exclude_patterns=["deprecated", "experimental", "tests", "__pycache__"],
            max_depth=5
        )
        methods = scanner.scan_package("insightspike")
        duration = time.time() - start
        
        result = {
            "component": "discovery",
            "methods_found": len(methods),
            "duration": duration,
            "success": True
        }
        
        logger.info(f"✅ Discovery: Found {len(methods)} methods in {duration:.2f}s")
        
        # Summary by type
        types = {}
        for m in methods:
            t = m.get("type", "unknown")
            types[t] = types.get(t, 0) + 1
        
        for t, count in types.items():
            logger.info(f"  • {t}: {count}")
        
    except Exception as e:
        duration = time.time() - start
        result = {
            "component": "discovery",
            "error": str(e),
            "duration": duration,
            "success": False
        }
        logger.error(f"❌ Discovery failed: {e}")
    
    return result


def run_docxology_examples() -> list[dict]:
    """Run docxology example scripts."""
    logger.info("=" * 60)
    logger.info("RUNNING DOCXOLOGY EXAMPLES")
    logger.info("=" * 60)
    
    results = []
    examples = sorted(DOCX_EXAMPLES_PATH.glob("*.py"))
    
    for script in examples:
        logger.info(f"Running: {script.name}")
        start = time.time()
        exit_code, output = run_command(["python", str(script)], timeout=120)
        duration = time.time() - start
        
        result = {
            "component": f"docx-examples/{script.name}",
            "exit_code": exit_code,
            "duration": duration,
            "success": exit_code == 0
        }
        results.append(result)
        
        if exit_code == 0:
            logger.info(f"  ✅ {script.name} completed in {duration:.2f}s")
        else:
            logger.warning(f"  ⚠️ {script.name} failed (exit code {exit_code})")
    
    return results


def run_parent_examples() -> list[dict]:
    """Run parent repository examples in logical order."""
    logger.info("=" * 60)
    logger.info("RUNNING PARENT REPOSITORY EXAMPLES")
    logger.info("=" * 60)
    
    if not PARENT_EXAMPLES_PATH.exists():
        logger.warning(f"Parent examples directory not found: {PARENT_EXAMPLES_PATH}")
        return []
    
    results = []
    
    for script_name in PARENT_EXAMPLES_ORDER:
        script_path = PARENT_EXAMPLES_PATH / script_name
        
        if not script_path.exists():
            logger.warning(f"  ⚠️ {script_name} not found, skipping")
            continue
        
        logger.info(f"Running: {script_name}")
        start = time.time()
        
        # Use longer timeout for simulation scripts
        timeout = 180 if "simulate" in script_name or "llm" in script_name else 120
        
        # Set PYTHONPATH to include parent src directory
        parent_env = {"PYTHONPATH": str(PARENT_ROOT / "src")}
        
        exit_code, output = run_command(
            ["python", str(script_path)],
            cwd=PARENT_ROOT,
            timeout=timeout,
            env=parent_env
        )
        duration = time.time() - start
        
        result = {
            "component": f"parent-examples/{script_name}",
            "exit_code": exit_code,
            "duration": duration,
            "success": exit_code == 0
        }
        results.append(result)
        
        if exit_code == 0:
            logger.info(f"  ✅ {script_name} completed in {duration:.2f}s")
        else:
            logger.warning(f"  ⚠️ {script_name} failed (exit code {exit_code})")
            # Show brief error for debugging
            if output:
                error_lines = [l for l in output.split('\n') if 'error' in l.lower() or 'Error' in l][:3]
                for line in error_lines:
                    logger.warning(f"      {line[:100]}")
    
    return results


def print_summary(results: list[dict]) -> bool:
    """Print execution summary and return overall success."""
    logger.info("")
    logger.info("=" * 60)
    logger.info("EXECUTION SUMMARY")
    logger.info("=" * 60)
    
    total_duration = sum(r.get("duration", 0) for r in results)
    successes = sum(1 for r in results if r.get("success", False))
    failures = len(results) - successes
    
    for r in results:
        status = "✅" if r.get("success") else "❌"
        component = r.get("component", "unknown")
        duration = r.get("duration", 0)
        
        extra = ""
        if "passed" in r:
            extra = f" ({r['passed']} passed)"
        elif "methods_found" in r:
            extra = f" ({r['methods_found']} methods)"
        
        logger.info(f"  {status} {component}: {duration:.2f}s{extra}")
    
    logger.info("-" * 60)
    logger.info(f"Total: {successes}/{len(results)} succeeded in {total_duration:.2f}s")
    
    if failures == 0:
        logger.info("🎉 All components passed!")
        return True
    else:
        logger.warning(f"⚠️ {failures} component(s) failed")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run all docxology tests and methods"
    )
    parser.add_argument("--tests", action="store_true", help="Run docxology tests only")
    parser.add_argument("--discovery", action="store_true", help="Run discovery only")
    parser.add_argument("--docx-examples", action="store_true", help="Run docxology examples only")
    parser.add_argument("--parent-examples", action="store_true", help="Run parent repo examples only")
    parser.add_argument("--quick", action="store_true", help="Quick smoke tests only (no examples)")
    args = parser.parse_args()
    
    start_time = datetime.now()
    logger.info(f"Docxology Run-All started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Docxology Root: {DOCXOLOGY_ROOT}")
    logger.info(f"Parent Root: {PARENT_ROOT}")
    
    results = []
    
    # Determine what to run
    run_all = not (args.tests or args.discovery or args.docx_examples or args.parent_examples)
    
    # Run tests
    if args.tests or run_all:
        results.append(run_tests(quick=args.quick))
    
    # Run discovery
    if args.discovery or run_all:
        results.append(run_discovery())
    
    # Run docxology examples (skip in quick mode)
    if (args.docx_examples or run_all) and not args.quick:
        results.extend(run_docxology_examples())
    
    # Run parent examples (skip in quick mode)
    if (args.parent_examples or run_all) and not args.quick:
        results.extend(run_parent_examples())
    
    # Print summary
    success = print_summary(results)
    
    end_time = datetime.now()
    logger.info(f"Finished at {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
