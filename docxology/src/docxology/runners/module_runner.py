"""Module runner for directly invoking module methods.

Provides direct Python function/method invocation with parameter injection,
result serialization, and exception handling.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ModuleRunner:
    """Directly invokes module methods with parameter injection.
    
    Imports and calls Python functions/methods, handling parameter
    injection from configuration, result serialization, and exceptions.
    
    Example:
        >>> runner = ModuleRunner()
        >>> result = runner.call(
        ...     "insightspike.algorithms.gedig_calculator",
        ...     "compute_gedig",
        ...     kwargs={"graph1": g1, "graph2": g2}
        ... )
    """
    
    def __init__(self, add_src_to_path: bool = True) -> None:
        """Initialize the module runner.
        
        Args:
            add_src_to_path: If True, add parent repo's src/ to sys.path.
        """
        if add_src_to_path:
            repo_root = Path(__file__).parent.parent.parent.parent.parent
            src_path = str(repo_root / "src")
            if src_path not in sys.path:
                sys.path.insert(0, src_path)
                logger.debug(f"Added to sys.path: {src_path}")
    
    def call(
        self,
        module_path: str,
        method_name: str,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call a method from a module.
        
        Args:
            module_path: Fully qualified module path (e.g., "insightspike.algorithms.gedig").
            method_name: Name of the function/method to call.
            args: Positional arguments for the method.
            kwargs: Keyword arguments for the method.
            
        Returns:
            Dictionary with execution results:
                - success: bool
                - result: Any (return value if successful)
                - error: str (if failed)
                - duration: float (seconds)
                - traceback: str (if failed)
        """
        kwargs = kwargs or {}
        start_time = time.time()
        
        try:
            # Import module
            module = importlib.import_module(module_path)
            
            # Get method
            if not hasattr(module, method_name):
                return {
                    "success": False,
                    "error": f"Method '{method_name}' not found in module '{module_path}'",
                    "duration": time.time() - start_time,
                }
            
            method = getattr(module, method_name)
            
            if not callable(method):
                return {
                    "success": False,
                    "error": f"'{method_name}' is not callable",
                    "duration": time.time() - start_time,
                }
            
            logger.info(f"Calling: {module_path}.{method_name}")
            
            # Call method
            result = method(*args, **kwargs)
            
            duration = time.time() - start_time
            
            return {
                "success": True,
                "result": result,
                "duration": duration,
            }
            
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"Method call failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "duration": duration,
            }
    
    def call_callable(
        self,
        callable_ref: Callable,
        args: tuple = (),
        kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Call a callable directly.
        
        Args:
            callable_ref: The callable to invoke.
            args: Positional arguments.
            kwargs: Keyword arguments.
            
        Returns:
            Dictionary with execution results.
        """
        kwargs = kwargs or {}
        start_time = time.time()
        
        try:
            name = getattr(callable_ref, "__name__", str(callable_ref))
            logger.info(f"Calling: {name}")
            
            result = callable_ref(*args, **kwargs)
            
            return {
                "success": True,
                "result": result,
                "duration": time.time() - start_time,
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "duration": time.time() - start_time,
            }
    
    def instantiate_and_call(
        self,
        module_path: str,
        class_name: str,
        method_name: str,
        init_args: tuple = (),
        init_kwargs: dict[str, Any] | None = None,
        method_args: tuple = (),
        method_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Instantiate a class and call a method on it.
        
        Args:
            module_path: Fully qualified module path.
            class_name: Name of the class to instantiate.
            method_name: Name of the method to call on the instance.
            init_args: Positional arguments for __init__.
            init_kwargs: Keyword arguments for __init__.
            method_args: Positional arguments for the method.
            method_kwargs: Keyword arguments for the method.
            
        Returns:
            Dictionary with execution results.
        """
        init_kwargs = init_kwargs or {}
        method_kwargs = method_kwargs or {}
        start_time = time.time()
        
        try:
            # Import module and get class
            module = importlib.import_module(module_path)
            cls = getattr(module, class_name)
            
            if not inspect.isclass(cls):
                return {
                    "success": False,
                    "error": f"'{class_name}' is not a class",
                    "duration": time.time() - start_time,
                }
            
            # Instantiate
            logger.info(f"Instantiating: {module_path}.{class_name}")
            instance = cls(*init_args, **init_kwargs)
            
            # Get and call method
            if not hasattr(instance, method_name):
                return {
                    "success": False,
                    "error": f"Instance has no method '{method_name}'",
                    "duration": time.time() - start_time,
                }
            
            method = getattr(instance, method_name)
            logger.info(f"Calling: {class_name}.{method_name}")
            
            result = method(*method_args, **method_kwargs)
            
            return {
                "success": True,
                "result": result,
                "instance": instance,
                "duration": time.time() - start_time,
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "duration": time.time() - start_time,
            }
    
    def get_method_signature(
        self,
        module_path: str,
        method_name: str,
    ) -> dict[str, Any]:
        """Get the signature of a method.
        
        Args:
            module_path: Fully qualified module path.
            method_name: Name of the method.
            
        Returns:
            Dictionary with method signature information.
        """
        try:
            module = importlib.import_module(module_path)
            method = getattr(module, method_name)
            sig = inspect.signature(method)
            
            params = []
            for name, param in sig.parameters.items():
                params.append({
                    "name": name,
                    "kind": str(param.kind),
                    "default": str(param.default) if param.default is not inspect.Parameter.empty else None,
                    "annotation": str(param.annotation) if param.annotation is not inspect.Parameter.empty else None,
                })
            
            return {
                "success": True,
                "signature": str(sig),
                "parameters": params,
                "docstring": inspect.getdoc(method) or "",
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }
