# Test Suite

Real functional tests for docxology (Zero-Mock policy).

## Structure

```
tests/
├── conftest.py          # Pytest fixtures
├── test_smoke.py        # Quick smoke tests (28 tests)
└── integration/         # End-to-end tests
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Smoke tests only
pytest tests/test_smoke.py -v

# Integration tests
pytest tests/integration/ -v

# With coverage
pytest tests/ --cov=docxology --cov-report=html
```

## Test Status

**28/28 tests PASSED** ✅

## Fixtures

| Fixture | Purpose |
|---------|---------|
| `docxology_root` | Path to docxology directory |
| `repo_root` | Path to parent repository |
| `scripts_dir` | Path to scripts directory |
| `sample_results` | Sample result data for testing |
