# Source Package

Contains the installable `docxology` Python package.

## Structure

```
src/
└── docxology/          # Main package
    ├── __init__.py     # Package init with version
    ├── cli.py          # CLI entry point
    ├── analysis/       # Results analysis
    ├── discovery/      # Method discovery
    ├── orchestrator/   # Pipeline execution
    ├── runners/        # Script/module runners
    └── visualization/  # Plotting/export
```

## Installation

```bash
cd docxology
pip install -e .
```

## See Also

- [docxology Package](./docxology/README.md)
- [Root README](../README.md)
