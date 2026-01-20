"""Method registry for storing discovered methods and their metadata.

Provides a centralized registry for all discovered methods, supporting
lookup, filtering, and serialization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)


@dataclass
class MethodInfo:
    """Information about a discovered method or class.
    
    Attributes:
        name: Method or class name.
        module: Fully qualified module path.
        type: Type of the member (function, class, method, callable).
        signature: Method signature as string.
        docstring: First line of documentation.
        callable_ref: Reference to the actual callable (may be None).
        tags: Optional tags for categorization.
    """
    name: str
    module: str
    type: str
    signature: str = ""
    docstring: str = ""
    callable_ref: Callable | None = None
    tags: list[str] = field(default_factory=list)
    
    @property
    def full_name(self) -> str:
        """Return fully qualified name."""
        return f"{self.module}.{self.name}"
    
    @property
    def short_doc(self) -> str:
        """Return first line of docstring."""
        if not self.docstring:
            return ""
        return self.docstring.split("\n")[0].strip()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (serializable)."""
        return {
            "name": self.name,
            "module": self.module,
            "type": self.type,
            "signature": self.signature,
            "docstring": self.short_doc,
            "full_name": self.full_name,
            "tags": self.tags,
        }
    
    def is_callable(self) -> bool:
        """Check if the method has a callable reference."""
        return self.callable_ref is not None and callable(self.callable_ref)


class MethodRegistry:
    """Registry for storing and querying discovered methods.
    
    Provides methods for registering, looking up, filtering, and
    exporting method information.
    
    Example:
        >>> registry = MethodRegistry()
        >>> registry.register(MethodInfo(name="my_func", module="mymodule", type="function"))
        >>> methods = registry.filter_by_type("function")
    """
    
    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._methods: dict[str, MethodInfo] = {}
        self._by_module: dict[str, list[str]] = {}
        self._by_type: dict[str, list[str]] = {}
    
    def register(self, info: MethodInfo) -> None:
        """Register a method in the registry.
        
        Args:
            info: MethodInfo to register.
        """
        key = info.full_name
        
        if key in self._methods:
            logger.debug(f"Overwriting existing entry: {key}")
        
        self._methods[key] = info
        
        # Index by module
        if info.module not in self._by_module:
            self._by_module[info.module] = []
        if key not in self._by_module[info.module]:
            self._by_module[info.module].append(key)
        
        # Index by type
        if info.type not in self._by_type:
            self._by_type[info.type] = []
        if key not in self._by_type[info.type]:
            self._by_type[info.type].append(key)
    
    def get(self, full_name: str) -> MethodInfo | None:
        """Get a method by its full name.
        
        Args:
            full_name: Fully qualified name (module.name).
            
        Returns:
            MethodInfo if found, None otherwise.
        """
        return self._methods.get(full_name)
    
    def filter_by_module(self, module: str, prefix_match: bool = True) -> list[MethodInfo]:
        """Filter methods by module name.
        
        Args:
            module: Module name to filter by.
            prefix_match: If True, match modules starting with the given name.
            
        Returns:
            List of matching MethodInfo objects.
        """
        results = []
        for mod, keys in self._by_module.items():
            if prefix_match and mod.startswith(module) or mod == module:
                results.extend(self._methods[k] for k in keys)
        return results
    
    def filter_by_type(self, type_name: str) -> list[MethodInfo]:
        """Filter methods by type.
        
        Args:
            type_name: Type to filter by (function, class, method, callable).
            
        Returns:
            List of matching MethodInfo objects.
        """
        keys = self._by_type.get(type_name, [])
        return [self._methods[k] for k in keys]
    
    def filter_by_tag(self, tag: str) -> list[MethodInfo]:
        """Filter methods by tag.
        
        Args:
            tag: Tag to filter by.
            
        Returns:
            List of matching MethodInfo objects.
        """
        return [m for m in self._methods.values() if tag in m.tags]
    
    def search(self, query: str) -> list[MethodInfo]:
        """Search methods by name or docstring.
        
        Args:
            query: Search query (case-insensitive substring match).
            
        Returns:
            List of matching MethodInfo objects.
        """
        query_lower = query.lower()
        return [
            m for m in self._methods.values()
            if query_lower in m.name.lower() or query_lower in m.docstring.lower()
        ]
    
    def clear(self) -> None:
        """Clear all registered methods."""
        self._methods.clear()
        self._by_module.clear()
        self._by_type.clear()
    
    def to_list(self) -> list[dict[str, Any]]:
        """Convert registry to list of dictionaries.
        
        Returns:
            List of method dictionaries (serializable).
        """
        return [m.to_dict() for m in self._methods.values()]
    
    def get_stats(self) -> dict[str, Any]:
        """Get registry statistics.
        
        Returns:
            Dictionary with statistics about registered methods.
        """
        return {
            "total": len(self._methods),
            "by_type": {t: len(keys) for t, keys in self._by_type.items()},
            "modules": len(self._by_module),
        }
    
    def __len__(self) -> int:
        """Return number of registered methods."""
        return len(self._methods)
    
    def __iter__(self) -> Iterator[MethodInfo]:
        """Iterate over registered methods."""
        return iter(self._methods.values())
    
    def __contains__(self, full_name: str) -> bool:
        """Check if a method is registered."""
        return full_name in self._methods
