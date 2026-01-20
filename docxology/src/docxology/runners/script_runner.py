"""Script runner for executing scripts from the scripts directory.

Provides subprocess-based execution with environment configuration,
output capture, and error handling.
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ScriptRunner:
    """Executes scripts from the parent repository's scripts directory.
    
    Provides subprocess-based execution with configurable environment,
    timeout handling, and output capture.
    
    Example:
        >>> runner = ScriptRunner()
        >>> result = runner.execute("run_fixed_mazes.py", env={"MAZE_SIZE": "15"})
        >>> print(result["success"])
    """
    
    def __init__(
        self,
        scripts_dir: Path | str | None = None,
        python_executable: str = "python",
        default_env: dict[str, str] | None = None,
        working_dir: Path | str | None = None,
    ) -> None:
        """Initialize the script runner.
        
        Args:
            scripts_dir: Path to scripts directory. Defaults to parent repo's scripts/.
            python_executable: Python executable to use.
            default_env: Default environment variables for all executions.
            working_dir: Working directory for execution. Defaults to repo root.
        """
        # Compute paths relative to this file
        self._docxology_root = Path(__file__).parent.parent.parent.parent
        self._repo_root = self._docxology_root.parent
        
        if scripts_dir:
            self.scripts_dir = Path(scripts_dir)
        else:
            self.scripts_dir = self._repo_root / "scripts"
        
        self.python_executable = python_executable
        self.default_env = default_env or {}
        
        if working_dir:
            self.working_dir = Path(working_dir)
        else:
            self.working_dir = self._repo_root
        
        logger.debug(f"ScriptRunner initialized: scripts_dir={self.scripts_dir}")
    
    def execute(
        self,
        script: str,
        args: list[str] | None = None,
        env: dict[str, str] | None = None,
        timeout: int = 300,
        capture_output: bool = True,
    ) -> dict[str, Any]:
        """Execute a script.
        
        Args:
            script: Script name or path (relative to scripts_dir or absolute).
            args: Optional command-line arguments.
            env: Additional environment variables.
            timeout: Execution timeout in seconds.
            capture_output: If True, capture stdout/stderr.
            
        Returns:
            Dictionary with execution results:
                - success: bool
                - returncode: int
                - stdout: str (if captured)
                - stderr: str (if captured)
                - duration: float (seconds)
                - error: str (if failed)
        """
        # Resolve script path
        script_path = self._resolve_script_path(script)
        if not script_path.exists():
            return {
                "success": False,
                "returncode": -1,
                "error": f"Script not found: {script}",
                "duration": 0,
            }
        
        # Build command
        cmd = [self.python_executable, str(script_path)]
        if args:
            cmd.extend(args)
        
        # Build environment
        run_env = os.environ.copy()
        run_env.update(self.default_env)
        if env:
            run_env.update(env)
        
        # Ensure PYTHONPATH includes src
        pythonpath = run_env.get("PYTHONPATH", "")
        src_path = str(self._repo_root / "src")
        if src_path not in pythonpath:
            run_env["PYTHONPATH"] = f"{src_path}:{pythonpath}" if pythonpath else src_path
        
        logger.info(f"Executing: {' '.join(cmd)}")
        logger.debug(f"Working dir: {self.working_dir}")
        
        start_time = time.time()
        
        try:
            if capture_output:
                result = subprocess.run(
                    cmd,
                    cwd=str(self.working_dir),
                    env=run_env,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                )
                stdout = result.stdout
                stderr = result.stderr
            else:
                result = subprocess.run(
                    cmd,
                    cwd=str(self.working_dir),
                    env=run_env,
                    timeout=timeout,
                )
                stdout = ""
                stderr = ""
            
            duration = time.time() - start_time
            
            return {
                "success": result.returncode == 0,
                "returncode": result.returncode,
                "stdout": stdout,
                "stderr": stderr,
                "duration": duration,
            }
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            logger.error(f"Script timed out after {timeout}s")
            return {
                "success": False,
                "returncode": -1,
                "error": f"Timeout after {timeout}s",
                "duration": duration,
            }
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Script execution failed: {e}")
            return {
                "success": False,
                "returncode": -1,
                "error": str(e),
                "duration": duration,
            }
    
    def _resolve_script_path(self, script: str) -> Path:
        """Resolve script path.
        
        Args:
            script: Script name or path.
            
        Returns:
            Resolved Path object.
        """
        script_path = Path(script)
        
        # If absolute, use as-is
        if script_path.is_absolute():
            return script_path
        
        # Check in scripts directory
        in_scripts = self.scripts_dir / script
        if in_scripts.exists():
            return in_scripts
        
        # Check relative to working dir
        relative = self.working_dir / script
        if relative.exists():
            return relative
        
        # Return path in scripts dir (even if doesn't exist)
        return in_scripts
    
    def list_scripts(self, pattern: str = "*.py") -> list[Path]:
        """List available scripts.
        
        Args:
            pattern: Glob pattern for matching scripts.
            
        Returns:
            List of script paths.
        """
        if not self.scripts_dir.exists():
            return []
        return sorted(self.scripts_dir.glob(pattern))
    
    def get_script_info(self, script: str) -> dict[str, Any]:
        """Get information about a script.
        
        Args:
            script: Script name or path.
            
        Returns:
            Dictionary with script information.
        """
        script_path = self._resolve_script_path(script)
        
        if not script_path.exists():
            return {"exists": False, "path": str(script_path)}
        
        # Read first docstring
        docstring = ""
        try:
            content = script_path.read_text()
            lines = content.split("\n")
            in_docstring = False
            for line in lines:
                if line.strip().startswith('"""') or line.strip().startswith("'''"):
                    if in_docstring:
                        break
                    in_docstring = True
                    docstring = line.strip().strip('"""').strip("'''")
                elif in_docstring:
                    if '"""' in line or "'''" in line:
                        docstring += " " + line.strip().strip('"""').strip("'''")
                        break
                    docstring += " " + line.strip()
        except Exception:
            pass
        
        return {
            "exists": True,
            "path": str(script_path),
            "name": script_path.name,
            "size": script_path.stat().st_size,
            "docstring": docstring.strip(),
        }
