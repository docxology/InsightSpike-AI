# Runners Module

Script and module execution system.

## Classes

| Class | Purpose |
|-------|---------|
| `ScriptRunner` | Execute Python scripts |
| `ModuleRunner` | Execute module methods directly |
| `BatchRunner` | Batch execution with parallelization |

## Usage

```python
from docxology.runners import ScriptRunner, ModuleRunner, BatchRunner

# Run a script
runner = ScriptRunner()
result = runner.execute("run_fixed_mazes.py")

# Run a module method
mod_runner = ModuleRunner()
result = mod_runner.call("insightspike.algorithms", "compute_gedig", args)

# Batch execution
batch = BatchRunner(max_workers=4)
results = batch.run_all(script_list)
```

## Files

- `script_runner.py` — `ScriptRunner` class
- `module_runner.py` — `ModuleRunner` class
- `batch_runner.py` — `BatchRunner` class
