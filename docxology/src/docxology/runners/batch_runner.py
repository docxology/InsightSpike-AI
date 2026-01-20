"""Batch runner for executing multiple scripts or methods.

Provides batch execution with parallel processing, progress tracking,
and aggregated results.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable

from .script_runner import ScriptRunner
from .module_runner import ModuleRunner

logger = logging.getLogger(__name__)


@dataclass
class BatchTask:
    """Represents a single task in a batch.
    
    Attributes:
        name: Task identifier.
        type: Task type ('script' or 'module').
        target: Script path or module.method.
        args: Positional arguments.
        kwargs: Keyword arguments.
        env: Environment variables (for scripts).
    """
    name: str
    type: str  # 'script' or 'module'
    target: str
    args: list[Any] = field(default_factory=list)
    kwargs: dict[str, Any] = field(default_factory=dict)
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class BatchResult:
    """Result of a batch execution.
    
    Attributes:
        name: Task identifier.
        success: Whether the task succeeded.
        result: Task result or error.
        duration: Execution time in seconds.
    """
    name: str
    success: bool
    result: Any
    duration: float


class BatchRunner:
    """Executes multiple tasks in batch with optional parallelism.
    
    Provides batch execution for scripts and module methods with
    parallel processing, progress tracking, and result aggregation.
    
    Example:
        >>> runner = BatchRunner(max_workers=4)
        >>> tasks = [
        ...     BatchTask(name="maze", type="script", target="run_fixed_mazes.py"),
        ...     BatchTask(name="analysis", type="script", target="analyze_step_logs.py"),
        ... ]
        >>> results = runner.run(tasks)
    """
    
    def __init__(
        self,
        max_workers: int = 4,
        parallel: bool = True,
        timeout: int = 300,
    ) -> None:
        """Initialize the batch runner.
        
        Args:
            max_workers: Maximum parallel workers.
            parallel: Enable parallel execution.
            timeout: Default timeout per task.
        """
        self.max_workers = max_workers
        self.parallel = parallel
        self.timeout = timeout
        
        self.script_runner = ScriptRunner()
        self.module_runner = ModuleRunner()
    
    def run(
        self,
        tasks: list[BatchTask],
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> list[BatchResult]:
        """Execute a batch of tasks.
        
        Args:
            tasks: List of tasks to execute.
            progress_callback: Optional callback(current, total, task_name).
            
        Returns:
            List of BatchResult objects.
        """
        if not tasks:
            return []
        
        logger.info(f"Starting batch execution: {len(tasks)} tasks, parallel={self.parallel}")
        results: list[BatchResult] = []
        
        if self.parallel and len(tasks) > 1:
            results = self._run_parallel(tasks, progress_callback)
        else:
            results = self._run_sequential(tasks, progress_callback)
        
        # Log summary
        success_count = sum(1 for r in results if r.success)
        total_duration = sum(r.duration for r in results)
        logger.info(
            f"Batch complete: {success_count}/{len(tasks)} succeeded, "
            f"total duration: {total_duration:.2f}s"
        )
        
        return results
    
    def _run_sequential(
        self,
        tasks: list[BatchTask],
        progress_callback: Callable[[int, int, str], None] | None,
    ) -> list[BatchResult]:
        """Execute tasks sequentially.
        
        Args:
            tasks: Tasks to execute.
            progress_callback: Optional progress callback.
            
        Returns:
            List of results.
        """
        results = []
        
        for i, task in enumerate(tasks):
            if progress_callback:
                progress_callback(i + 1, len(tasks), task.name)
            
            result = self._execute_task(task)
            results.append(result)
        
        return results
    
    def _run_parallel(
        self,
        tasks: list[BatchTask],
        progress_callback: Callable[[int, int, str], None] | None,
    ) -> list[BatchResult]:
        """Execute tasks in parallel.
        
        Args:
            tasks: Tasks to execute.
            progress_callback: Optional progress callback.
            
        Returns:
            List of results.
        """
        results: dict[str, BatchResult] = {}
        completed = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._execute_task, task): task
                for task in tasks
            }
            
            for future in as_completed(futures):
                task = futures[future]
                completed += 1
                
                if progress_callback:
                    progress_callback(completed, len(tasks), task.name)
                
                try:
                    result = future.result()
                    results[task.name] = result
                except Exception as e:
                    logger.error(f"Task {task.name} failed: {e}")
                    results[task.name] = BatchResult(
                        name=task.name,
                        success=False,
                        result=str(e),
                        duration=0,
                    )
        
        # Return in original order
        return [results[task.name] for task in tasks if task.name in results]
    
    def _execute_task(self, task: BatchTask) -> BatchResult:
        """Execute a single task.
        
        Args:
            task: Task to execute.
            
        Returns:
            BatchResult.
        """
        start_time = time.time()
        
        try:
            if task.type == "script":
                result = self.script_runner.execute(
                    task.target,
                    args=task.args,
                    env=task.env,
                    timeout=self.timeout,
                )
            elif task.type == "module":
                # Parse module.method format
                if "." in task.target:
                    parts = task.target.rsplit(".", 1)
                    module_path, method_name = parts[0], parts[1]
                else:
                    return BatchResult(
                        name=task.name,
                        success=False,
                        result="Invalid module target format (expected module.method)",
                        duration=time.time() - start_time,
                    )
                
                result = self.module_runner.call(
                    module_path,
                    method_name,
                    args=tuple(task.args),
                    kwargs=task.kwargs,
                )
            else:
                return BatchResult(
                    name=task.name,
                    success=False,
                    result=f"Unknown task type: {task.type}",
                    duration=time.time() - start_time,
                )
            
            return BatchResult(
                name=task.name,
                success=result.get("success", False),
                result=result,
                duration=result.get("duration", time.time() - start_time),
            )
            
        except Exception as e:
            return BatchResult(
                name=task.name,
                success=False,
                result=str(e),
                duration=time.time() - start_time,
            )
    
    def run_scripts(
        self,
        scripts: list[str],
        env: dict[str, str] | None = None,
    ) -> list[BatchResult]:
        """Convenience method to run multiple scripts.
        
        Args:
            scripts: List of script paths.
            env: Environment variables for all scripts.
            
        Returns:
            List of results.
        """
        tasks = [
            BatchTask(
                name=script,
                type="script",
                target=script,
                env=env or {},
            )
            for script in scripts
        ]
        return self.run(tasks)
