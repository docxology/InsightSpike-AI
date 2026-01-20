# Integration Tests

End-to-end tests verifying full docxology workflows.

## Test Classes

| Class | Tests |
|-------|-------|
| `TestDiscoveryIntegration` | Scan real insightspike package |
| `TestScriptRunnerIntegration` | List and run real scripts |
| `TestPipelineIntegration` | Discovery→Analysis pipeline |
| `TestAnalysisVisualizationIntegration` | Analyze and export |

## Running

```bash
pytest tests/integration/ -v
```
