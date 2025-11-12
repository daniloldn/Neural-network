## Purpose

These short instructions help an AI coding assistant become immediately productive in this repository by describing the project's architecture, developer workflows, project-specific patterns, and a few concrete examples.

## Quick start (how devs run things)

- This project targets Python 3.12 (see `pyproject.toml`). The repository includes a virtualenv at `vneural/` that already has common tools (pytest, numpy).
- Typical quick commands a developer uses from the project root:

```
# activate the provided venv
source vneural/bin/activate

# run tests
pytest -q
```

Note: test modules add the parent directory to `sys.path` (see `test/*.py`) rather than installing the package. Mirror that when running quick scripts.

## Big-picture architecture (what to read first)

- `neural_network/network.py` — top-level model object `Network`. Responsible for assembling:
  - `init_layer` (first layer, `Layer`)
  - `hidden_layer` (a Python list of `Layer` objects)
  - `output_layer` (final `Layer`)
  - `feedforward(x)` which does: init_layer.feed_forward -> recurse through `hidden_layer` -> output_layer.feed_forward

- `neural_network/layer.py` — container of `Neuron` objects. `Layer.feed_forward(inputs)` returns a Python list of neuron outputs and also stores `self.output`.

- `neural_network/neuron.py` — single neuron implementation. Important details:
  - `weights` are initialized as a transposed 2D numpy array and `bias` as a scalar.
  - `feed_forward(inputs)` converts inputs to a numpy array and computes `np.dot(x, self.weights) + bias` then applies `activation`.
  - Activation defaults to `sigmoid` from `activation.py`.

- `neural_network/activation.py` — `sigmoid` and `deriv_sigmoid`. Activation functions are implemented using numpy and accept scalars or arrays.

- `neural_network/training.py` — currently contains `mse_loss`. There is no training loop implemented in the repo; add training routines that accept/return numpy arrays compatible with `mse_loss`.

## Project-specific conventions & gotchas (do not assume defaults)

- Tests use an import-time `sys.path.insert(0, str(Path(__file__).resolve().parents[1]))` pattern to import `neural_network` directly from source. When adding new tests or quick scripts, either use the same pattern or install the package into the venv.

- `Network.network` stores `[init_layer, hidden_layer, output_layer]` where `hidden_layer` itself is a Python list. Tests access nested structures (e.g. `(net.network[1])[0].neurons`). When modifying network internals, keep compatibility with this nested-list shape unless you update tests.

- `Neuron.weights` shape is a transposed array which can make broadcasting/dot-product behavior different than a plain 1-D weights vector. When changing Neuron internals, verify shapes in `neurons_test.py` and across `Layer.feed_forward`.

- Type hints are used, and `pyproject.toml` sets `mypy.strict = true` but also `ignore_missing_imports = true`. Aim to keep type hints accurate; small local changes are acceptable, but large refactors should preserve hints or update pyproject config.

## Examples (concrete lines to reference)

- Feedforward flow example: `neural_network/network.py` `feedforward()` — call order is `init_layer.feed_forward(x)` → recursive hidden feedforward → `output_layer.feed_forward(...)`.

- Creating a small network (used in tests):
  - `Network(x, y, depth=1, width=3)` where `x` is a 2D numpy array and `y` a target array.

## Tests, debugging and validation

- Run tests with `pytest -q` from the project root. Tests are small and use deterministic instantiation but rely on random weights; when writing deterministic tests, seed numpy's RNG with `np.random.seed(...)` at top of test.

- Use `vneural/bin/python -m pytest` if you prefer an explicit interpreter.

- For quick debugging, tests print structures (see `test/*.py`) and import the package by modifying `sys.path` instead of requiring installation.

## When changing or extending the project

- Keep the public API of `Network`, `Layer`, and `Neuron` stable: many tests reference internal attributes (e.g., `.neurons`, `.weights`, `.bias`). If you change shapes or storage types, update tests together with implementation.

- If you add training loops, follow the `mse_loss(y_true, y_pred)` signature and use numpy arrays for inputs/outputs. Place new code into `neural_network/training.py` and add tests under `test/`.

## Files to read first when making changes

- `neural_network/network.py` — model assembly and feedforward
- `neural_network/layer.py` — layer/neurons orchestration
- `neural_network/neuron.py` — compute kernel for a neuron
- `neural_network/activation.py` — activation functions
- `test/` — small, illustrative tests that show how the API is used

## Feedback

If any of these sections are unclear or you'd like more examples (training loop skeleton, shapes matrix, deterministic tests), tell me which area to expand and I will iterate.
