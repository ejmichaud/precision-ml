# precision-ml

Tools for training neural networks to very high precision.

## Installation
```
pip install -e .
```

## Usage
```python
from precisionml.optimizers import ConjugateGradients

model = nn.Sequential(...)
input, target = 
optimizer = ConjugateGradients(model.parameters())
optimizer.zero_grad()
loss_fn(model(input), target).backward()
optimizer.step()
```

## Testing
I've written some tests in the `tests/` directory. Run them with:
```
pytest tests/
```
