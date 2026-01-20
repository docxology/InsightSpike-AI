"""Pipeline orchestrator for multi-stage execution.

Provides pipeline definition and execution with dependency management,
progress tracking, and error handling.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)


class StageStatus(Enum):
    """Status of a pipeline stage."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class Stage:
    """Represents a stage in a pipeline.
    
    Attributes:
        name: Stage identifier.
        func: Callable to execute.
        depends_on: List of stage names this stage depends on.
        status: Current status.
        result: Execution result.
        duration: Execution time in seconds.
    """
    name: str
    func: Callable[..., Any]
    depends_on: list[str] = field(default_factory=list)
    status: StageStatus = StageStatus.PENDING
    result: Any = None
    error: str | None = None
    duration: float = 0.0


class Pipeline:
    """Multi-stage pipeline with dependency management.
    
    Executes stages in dependency order, passing results between stages,
    with progress tracking and error handling.
    
    Example:
        >>> pipeline = Pipeline("analysis")
        >>> pipeline.add_stage("load", load_data)
        >>> pipeline.add_stage("process", process_data, depends_on=["load"])
        >>> pipeline.add_stage("report", generate_report, depends_on=["process"])
        >>> results = pipeline.run()
    """
    
    def __init__(self, name: str) -> None:
        """Initialize a pipeline.
        
        Args:
            name: Pipeline identifier.
        """
        self.name = name
        self.stages: dict[str, Stage] = {}
        self._execution_order: list[str] = []
        self._context: dict[str, Any] = {}
    
    def add_stage(
        self,
        name: str,
        func: Callable[..., Any],
        depends_on: list[str] | None = None,
    ) -> "Pipeline":
        """Add a stage to the pipeline.
        
        Args:
            name: Stage identifier.
            func: Callable to execute. Receives (context, **kwargs) and returns result.
            depends_on: List of stage names this stage depends on.
            
        Returns:
            Self for chaining.
        """
        if name in self.stages:
            raise ValueError(f"Stage '{name}' already exists")
        
        self.stages[name] = Stage(
            name=name,
            func=func,
            depends_on=depends_on or [],
        )
        
        self._execution_order = []  # Reset, will be computed on run
        return self
    
    def run(
        self,
        initial_context: dict[str, Any] | None = None,
        progress_callback: Callable[[str, StageStatus], None] | None = None,
    ) -> dict[str, Any]:
        """Execute the pipeline.
        
        Args:
            initial_context: Initial context data available to all stages.
            progress_callback: Optional callback(stage_name, status).
            
        Returns:
            Dictionary with execution results:
                - success: bool
                - stages: dict of stage results
                - duration: total duration
                - context: final context
        """
        logger.info(f"Starting pipeline: {self.name}")
        start_time = time.time()
        
        # Initialize context
        self._context = initial_context.copy() if initial_context else {}
        
        # Compute execution order
        self._execution_order = self._topological_sort()
        logger.debug(f"Execution order: {self._execution_order}")
        
        # Reset all stages
        for stage in self.stages.values():
            stage.status = StageStatus.PENDING
            stage.result = None
            stage.error = None
            stage.duration = 0.0
        
        # Execute stages
        all_success = True
        for stage_name in self._execution_order:
            stage = self.stages[stage_name]
            
            # Check dependencies
            deps_ok = all(
                self.stages[dep].status == StageStatus.COMPLETED
                for dep in stage.depends_on
            )
            
            if not deps_ok:
                stage.status = StageStatus.SKIPPED
                stage.error = "Dependencies not satisfied"
                if progress_callback:
                    progress_callback(stage_name, StageStatus.SKIPPED)
                continue
            
            # Execute stage
            stage.status = StageStatus.RUNNING
            if progress_callback:
                progress_callback(stage_name, StageStatus.RUNNING)
            
            stage_start = time.time()
            try:
                logger.info(f"Executing stage: {stage_name}")
                result = stage.func(self._context)
                stage.result = result
                stage.status = StageStatus.COMPLETED
                stage.duration = time.time() - stage_start
                
                # Store result in context
                self._context[f"stage_{stage_name}"] = result
                
                if progress_callback:
                    progress_callback(stage_name, StageStatus.COMPLETED)
                    
            except Exception as e:
                logger.error(f"Stage {stage_name} failed: {e}")
                stage.status = StageStatus.FAILED
                stage.error = str(e)
                stage.duration = time.time() - stage_start
                all_success = False
                
                if progress_callback:
                    progress_callback(stage_name, StageStatus.FAILED)
        
        total_duration = time.time() - start_time
        
        return {
            "success": all_success,
            "pipeline": self.name,
            "stages": {
                name: {
                    "status": stage.status.value,
                    "duration": stage.duration,
                    "error": stage.error,
                }
                for name, stage in self.stages.items()
            },
            "duration": total_duration,
            "context": self._context,
        }
    
    def _topological_sort(self) -> list[str]:
        """Compute execution order based on dependencies.
        
        Returns:
            List of stage names in execution order.
            
        Raises:
            ValueError: If circular dependency detected.
        """
        visited: set[str] = set()
        result: list[str] = []
        temp_mark: set[str] = set()
        
        def visit(name: str) -> None:
            if name in temp_mark:
                raise ValueError(f"Circular dependency detected at stage '{name}'")
            if name in visited:
                return
            
            temp_mark.add(name)
            
            stage = self.stages.get(name)
            if stage:
                for dep in stage.depends_on:
                    if dep not in self.stages:
                        raise ValueError(f"Stage '{name}' depends on unknown stage '{dep}'")
                    visit(dep)
            
            temp_mark.remove(name)
            visited.add(name)
            result.append(name)
        
        for name in self.stages:
            if name not in visited:
                visit(name)
        
        return result
    
    def get_stage(self, name: str) -> Stage | None:
        """Get a stage by name.
        
        Args:
            name: Stage name.
            
        Returns:
            Stage if found, None otherwise.
        """
        return self.stages.get(name)
    
    def get_result(self, stage_name: str) -> Any:
        """Get the result of a stage.
        
        Args:
            stage_name: Name of the stage.
            
        Returns:
            Stage result or None.
        """
        stage = self.stages.get(stage_name)
        return stage.result if stage else None
    
    def reset(self) -> None:
        """Reset all stages to pending state."""
        for stage in self.stages.values():
            stage.status = StageStatus.PENDING
            stage.result = None
            stage.error = None
            stage.duration = 0.0
        self._context.clear()
    
    def __len__(self) -> int:
        """Return number of stages."""
        return len(self.stages)
