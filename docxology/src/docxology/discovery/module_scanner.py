"""Module scanner for discovering methods in Python packages.

Scans a package hierarchy to discover all public classes, functions, and methods.
Uses introspection to extract signatures and documentation.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil
import sys
from pathlib import Path
from typing import Any, Callable, Iterator

from .method_registry import MethodInfo, MethodRegistry

logger = logging.getLogger(__name__)


class ModuleScanner:
    """Scans Python packages to discover methods and classes.
    
    Uses introspection to find all public callable objects in a package
    hierarchy, extracting signatures, docstrings, and metadata.
    
    Example:
        >>> scanner = ModuleScanner()
        >>> methods = scanner.scan_package("insightspike")
        >>> print(f"Found {len(methods)} methods")
    """
    
    def __init__(
        self,
        exclude_patterns: list[str] | None = None,
        include_private: bool = False,
        max_depth: int = 10,
    ) -> None:
        """Initialize the scanner.
        
        Args:
            exclude_patterns: Module name patterns to exclude.
            include_private: If True, include private methods (starting with _).
            max_depth: Maximum recursion depth for submodules.
        """
        self.exclude_patterns = exclude_patterns or [
            "deprecated",
            "experimental",
            "__pycache__",
            "tests",
            "test_",
        ]
        self.include_private = include_private
        self.max_depth = max_depth
        self.registry = MethodRegistry()
        self._scanned_modules: set[str] = set()
    
    def scan_package(self, package_name: str) -> list[dict[str, Any]]:
        """Scan a package and return discovered methods.
        
        Args:
            package_name: Name of the package to scan (e.g., "insightspike").
            
        Returns:
            List of method information dictionaries.
        """
        logger.info(f"Scanning package: {package_name}")
        self._scanned_modules.clear()
        self.registry.clear()
        
        try:
            package = importlib.import_module(package_name)
        except ImportError as e:
            logger.error(f"Failed to import package {package_name}: {e}")
            return []
        
        # Scan the root package
        self._scan_module(package, depth=0)
        
        # If package has __path__, scan submodules
        if hasattr(package, "__path__"):
            self._scan_submodules(package, package_name, depth=0)
        
        logger.info(f"Discovery complete: found {len(self.registry)} methods")
        return self.registry.to_list()
    
    def _scan_module(self, module: Any, depth: int) -> None:
        """Scan a single module for callable objects.
        
        Args:
            module: The module object to scan.
            depth: Current recursion depth.
        """
        module_name = getattr(module, "__name__", str(module))
        
        if module_name in self._scanned_modules:
            return
        self._scanned_modules.add(module_name)
        
        if self._should_exclude(module_name):
            logger.debug(f"Skipping excluded module: {module_name}")
            return
        
        logger.debug(f"Scanning module: {module_name}")
        
        try:
            members = inspect.getmembers(module)
        except Exception as e:
            logger.warning(f"Failed to get members of {module_name}: {e}")
            return
        
        for name, obj in members:
            if self._should_skip_member(name, obj, module_name):
                continue
            
            try:
                self._register_member(name, obj, module_name)
            except Exception as e:
                logger.debug(f"Failed to register {name}: {e}")
    
    def _scan_submodules(self, package: Any, package_name: str, depth: int) -> None:
        """Recursively scan submodules of a package.
        
        Args:
            package: The package object.
            package_name: Fully qualified package name.
            depth: Current recursion depth.
        """
        if depth >= self.max_depth:
            logger.debug(f"Max depth reached at {package_name}")
            return
        
        try:
            submodules = list(pkgutil.iter_modules(package.__path__, prefix=f"{package_name}."))
        except Exception as e:
            logger.debug(f"Failed to iterate submodules of {package_name}: {e}")
            return
        
        for importer, modname, is_pkg in submodules:
            if self._should_exclude(modname):
                continue
            
            try:
                submodule = importlib.import_module(modname)
                self._scan_module(submodule, depth + 1)
                
                if is_pkg:
                    self._scan_submodules(submodule, modname, depth + 1)
            except Exception as e:
                logger.debug(f"Failed to import {modname}: {e}")
    
    def _should_exclude(self, name: str) -> bool:
        """Check if a module should be excluded from scanning.
        
        Args:
            name: Module name to check.
            
        Returns:
            True if module should be excluded.
        """
        for pattern in self.exclude_patterns:
            if pattern in name:
                return True
        return False
    
    def _should_skip_member(self, name: str, obj: Any, module_name: str) -> bool:
        """Check if a member should be skipped.
        
        Args:
            name: Member name.
            obj: Member object.
            module_name: Parent module name.
            
        Returns:
            True if member should be skipped.
        """
        # Skip private members unless configured
        if name.startswith("_") and not self.include_private:
            return True
        
        # Skip non-callables (except classes)
        if not (callable(obj) or inspect.isclass(obj)):
            return True
        
        # Skip built-ins and members from other modules
        obj_module = getattr(obj, "__module__", None)
        if obj_module and not obj_module.startswith(module_name.split(".")[0]):
            return True
        
        return False
    
    def _register_member(self, name: str, obj: Any, module_name: str) -> None:
        """Register a member in the registry.
        
        Args:
            name: Member name.
            obj: Member object.
            module_name: Parent module name.
        """
        # Determine type
        if inspect.isclass(obj):
            obj_type = "class"
        elif inspect.isfunction(obj):
            obj_type = "function"
        elif inspect.ismethod(obj):
            obj_type = "method"
        else:
            obj_type = "callable"
        
        # Get signature
        try:
            sig = str(inspect.signature(obj))
        except (ValueError, TypeError):
            sig = "()"
        
        # Get docstring
        doc = inspect.getdoc(obj) or ""
        
        # Create method info
        info = MethodInfo(
            name=name,
            module=module_name,
            type=obj_type,
            signature=sig,
            docstring=doc,
            callable_ref=obj,
        )
        
        self.registry.register(info)
    
    def get_registry(self) -> MethodRegistry:
        """Get the method registry.
        
        Returns:
            The MethodRegistry containing discovered methods.
        """
        return self.registry
