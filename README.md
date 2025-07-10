# Learnable Product-Price Attribution for Retail Shelf Images

This is the official implementation of "Learnable Product-Price Attribution for Retail Shelf Images", a novel approach to solving retail price attribution using neural networks.

## Getting Started

### Install Project Dependencies

`price-net` is managed via the `uv` package manager ([installation instructions](https://docs.astral.sh/uv/getting-started/installation/)). To install the dependencies, simply run `uv sync` from the root directory of the repository after cloning.

### Install Pre-Commit Hook

To install this repo's pre-commit hook with automatic linting and code quality checks, simply execute the following command:

```bash
pre-commit install
```

When you commit new code, the pre-commit hook will run a series of scripts to standardize formatting and run code quality checks. Any issues must be resolved for the commit to go through. If you need to bypass the linters for a specific commit, add the `--no-verify` flag to your git commit command.

## Viewing a Dataset

To view a dataset, simply use the [Data Viewer](notebooks/data_viewer.ipynb). This file is a Jupyter notebook that provides an interactive interface for visualizing individual price scenes.

## Training a Product-Price Associator

To train an association model, first fill out a `TrainingConfig` (following the [specified schema](src/price_net/configs.py)). Then, run the [training script](src/price_net/training/train_associator.py):

```bash
uv run train_associator --config path/to/your/config.yaml
```

The training script will save trained weights (both the best in terms of validation loss and the most recent copy) to the checkpoint directory specified in the config, and metrics will be logged in Weights and Biases (if indicated in the config) or locally (to the log directory specified in the config). The train config will also be saved in this log directory.

## Evaluation

### Evaluating a Trained Product-Price Associator

To evaluate a product-price associator, first fill out an `EvaluationConfig` (see the [specifications](src/price_net/configs.py) for details). Then, run the [associator evaluation script](src/price_net/association/evaluate.py) via:

```bash
uv run evaluate_associator --config path/to/your/eval/config.yaml
```

This script will follow the logging settings specified in the config (WandB vs. local). It will also save evaluation metrics to a YAML file in the specified results directory.

To get a qualitative sense of how well a model performs for price attribution, use the [Predictions Viewer](notebooks/predictions_viewer.ipynb). This file is a Jupyter notebook that provides an interactive interface for visualizing individual predicted price associations (and comparing them to the ground truth).

### Evaluating a Heuristic Product-Price Associator

All heuristic methods for product-price association should implement the `Heuristic` protocol in [this file](src/price_net/association/heuristics.py). Then, to evaluate a specific method, run the [heuristic associator evaluation script](src/price_net/association/evaluate_heuristic.py) via:

```bash
uv run evaluate_heuristic_associator \
    --dataset-dir path/to/dataset \
    --heuristic name-of-heuristic \
    --results-dir dir/for/results
```

## Development

### Managing Dependencies

To add a new dependency to the project, run `uv add <package-name>`. This will install the dependency into uv's managed .venv and automatically update the `pyproject.toml` file and the `uv.lock` file, ensuring that the dependency is available for all users of the project who run `uv sync`.

To remove a dependency, run `uv remove <package-name>`. This will perform the reverse of `uv add` (including updating the `pyproject.toml` and `uv.lock` files).

See [uv's documentation](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) for more details.
