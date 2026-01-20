# Training Module

Model training and quantization utilities.

## Components

| File | Purpose |
|------|---------|
| `train.py` | Model training functions |
| `predict.py` | Prediction utilities |
| `quantizer.py` | Model quantization |

## Usage

```python
from insightspike.training import train, predict, quantizer

# Train a model
model = train.fit(data)

# Make predictions
predictions = predict.run(model, new_data)

# Quantize for deployment
quantized = quantizer.quantize(model)
```
