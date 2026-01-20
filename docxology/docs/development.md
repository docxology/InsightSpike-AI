# Development Guide

Summary of development documentation from `/docs/development/` and contributing guides.

## Contributing

See: [CONTRIBUTING.md](../../docs/CONTRIBUTING.md)

### Code Standards

- **Zero-Mock Policy**: All methods must be real implementations
- **Configuration-Driven**: No hardcoded values
- **Comprehensive Logging**: All operations logged

### Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src/insightspike

# Quick smoke tests
pytest tests/test_smoke.py -v
```

## Code of Conduct

Community guidelines for contributors.

See: [CODE_OF_CONDUCT.md](../../docs/CODE_OF_CONDUCT.md)

## Version History

### Current: v0.8.0

Key features:
- 4-layer agent architecture
- geDIG metrics (ΔGED, ΔIG)
- Torch Geometric integration
- Zero-mock verification

See: [CHANGELOG.md](../../docs/CHANGELOG.md)

## Development Tools

### docxology Framework

Testing and validation framework:

```bash
cd docxology

# Run all tests and discovery
python run_all.py

# Quick verification
python run_all.py --quick
```

See: [../README.md](../README.md)

---

## Source Documents

| Document | Path |
|----------|------|
| Contributing | [CONTRIBUTING.md](../../docs/CONTRIBUTING.md) |
| Code of Conduct | [CODE_OF_CONDUCT.md](../../docs/CODE_OF_CONDUCT.md) |
| Changelog | [CHANGELOG.md](../../docs/CHANGELOG.md) |
| Version Matrix | [version_matrix.md](../../docs/version_matrix.md) |
| Development Dir | [development/](../../docs/development/) |
