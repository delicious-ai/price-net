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

To train an association model, first fill out an `AssociatorTrainingConfig` (following the [specified schema](src/price_net/configs.py)). Then, run the [training script](src/price_net/training/train_associator.py):

```bash
uv run train_associator --config path/to/your/config.yaml
```

The training script will save trained weights (both the best in terms of validation loss and the most recent copy) to the checkpoint directory specified in the config, and metrics will be logged in Weights and Biases (if indicated in the config) or locally (to the log directory specified in the config). The train config will also be saved in this log directory.

**Note**: If training on a GPU, our enforcement of deterministic operations may mean you need to set `CUBLAS_WORKSPACE_CONFIG=:4096:8` in your environment before running the above script.

## Evaluation

### Evaluating a Trained Product-Price Associator

To evaluate a product-price associator, first fill out an `AssociatorEvaluationConfig` (see the [specifications](src/price_net/association/configs.py) for details). Then, run the [associator evaluation script](src/price_net/association/evaluate.py) via:

```bash
uv run evaluate_associator --config path/to/your/eval/config.yaml
```

This script will follow the logging settings specified in the config (WandB vs. local). It will also save evaluation metrics to a `association_metrics.yaml` file in the specified results directory.

To get a qualitative sense of how well a model performs for price attribution, use the [Predictions Viewer](notebooks/predictions_viewer.ipynb). This file is a Jupyter notebook that provides an interactive interface for visualizing individual predicted price associations (and comparing them to the ground truth).

### Evaluating a Heuristic Product-Price Associator

All heuristic methods for product-price association should implement the `Heuristic` protocol in [this file](src/price_net/association/heuristics.py). Then, to evaluate a specific method, run the [heuristic associator evaluation script](src/price_net/association/evaluate_heuristic.py) via:

```bash
uv run evaluate_heuristic_associator \
    --dataset-dir path/to/dataset \
    --heuristic name-of-heuristic \
    --results-dir dir/for/results
```

Evaluation metrics will be saved in a `association_metrics.yaml` file in the specified results directory.

### Evaluating a Price Attribution System

To evaluate an end-to-end price attribution system, run the [attribution system evaluation script](src/price_net/scripts/evaluate_attributions.py):

```bash
uv run evaluate_attribution_system --config path/to/attribution/config.yaml
```

Running this script requires exactly one of the following (described via the `AttributionEvaluationConfig`):

1. A pre-computed set of attributions (a JSON file with a list of json-ified `PriceAttribution` objects).
2. A set of pre-computed price extractions and specifications for running price association.

The specifications mentioned in (2) include either a heuristic method (such as `nearest_per_group`) or a learned associator model (as described by an `AssociatorEvaluationConfig`). Here is what the config would look like for evaluating the price attribution performance of a system that uses heuristic association:

```yaml
dataset_dir: path/to/dataset/dir
results_dir: path/to/results/dir
# The price file is just a list of price bbox IDs and their extracted price
extracted_prices_path: path/to/prices.json
heuristic: nearest_below_per_group
```

Here is what the config would look like for evaluating the price attribution performance of a system that uses a learned association model:

```yaml
dataset_dir: path/to/dataset/dir
results_dir: path/to/results/dir
extracted_prices_path: path/to/prices.json
associator_eval_config_path: path/to/associator/eval/config.yaml
# Used to determine what is considered a valid "association"
threshold: 0.5
```

Evaluation metrics will be saved in an `attribution_metrics.yaml` file in the results directory listed in your attribution config.

## Extraction

To run a gemini-based extraction model, set up your `.env` with the following environment variables:

```dotenv
GOOGLE_APPLICATION_CREDENTIALS={path-to-your-gcloud-auth-json}
GOOGLE_CLOUD_PROJECT={gcloud-project}
GOOGLE_CLOUD_LOCATION={gcloud-region}
```

## Development

### Managing Dependencies

To add a new dependency to the project, run `uv add <package-name>`. This will install the dependency into uv's managed .venv and automatically update the `pyproject.toml` file and the `uv.lock` file, ensuring that the dependency is available for all users of the project who run `uv sync`.

To remove a dependency, run `uv remove <package-name>`. This will perform the reverse of `uv add` (including updating the `pyproject.toml` and `uv.lock` files).

See [uv's documentation](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) for more details.
