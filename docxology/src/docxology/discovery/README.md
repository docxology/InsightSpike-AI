# Discovery Module

Method discovery system using Python introspection.

## Classes

| Class | Purpose |
|-------|---------|
| `ModuleScanner` | Scan packages to discover methods |
| `MethodRegistry` | Store and query discovered methods |
| `MethodInfo` | Data class for method metadata |

## Usage

```python
from docxology.discovery import ModuleScanner, MethodRegistry

scanner = ModuleScanner(
    exclude_patterns=["deprecated", "tests"],
    max_depth=5
)
methods = scanner.scan_package("insightspike")
print(f"Found {len(methods)} methods")
```

## Files

- `module_scanner.py` — `ModuleScanner` class
- `method_registry.py` — `MethodRegistry`, `MethodInfo`
