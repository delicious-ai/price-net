# Learning to Attribute Products to Price Tags in Retail Shelf Images

This is the official implementation of "Learning to Attribute Products to Price Tags in Retail Shelf Images", which proposes the first learnable system for end-to-end retail price attribution.

## Getting Started

### Install Project Dependencies

`price-net` is managed via the `uv` package manager ([installation instructions](https://docs.astral.sh/uv/getting-started/installation/)). To install the dependencies, simply run `uv sync` from the root directory of the repository after cloning.

### Install Pre-Commit Hook

To install this repo's pre-commit hook with automatic linting and code quality checks, simply execute the following command:

```bash
pre-commit install
```

When you commit new code, the pre-commit hook will run a series of scripts to standardize formatting and run code quality checks. Any issues must be resolved for the commit to go through. If you need to bypass the linters for a specific commit, add the `--no-verify` flag to your git commit command.

## Viewing Instances from BRePS

To view instances from the **B**everage **Re**tail Price Scenes (BRePS) dataset, simply use the [Data Viewer](notebooks/data_viewer.ipynb). This file is a Jupyter notebook that provides an interactive interface for visualizing individual price scenes. Note that you will need `BRePS` downloaded locally to view it.

## Training a Product-Price Associator

To train an association model, first fill out an `AssociatorTrainingConfig` (following the [specified schema](src/price_net/configs.py)). Then, run the [training script](src/price_net/association/train.py):

```bash
uv run train_associator --config path/to/your/config.yaml
```

The training script will save trained weights (both the best in terms of validation loss and the most recent copy) to the checkpoint directory specified in the config, and metrics will be logged in Weights and Biases (if indicated in the config) or locally (to the log directory specified in the config). The train config will also be saved in this log directory.

**Note**: If training on a GPU, our enforcement of deterministic operations may mean you need to set `CUBLAS_WORKSPACE_CONFIG=:4096:8` in your environment before running the above script.

## Running a Price Extractor

To run a gemini-based extraction model, set up your `.env` with the following environment variables:

```dotenv
GOOGLE_APPLICATION_CREDENTIALS={path-to-your-gcloud-auth-json}
GOOGLE_CLOUD_PROJECT={gcloud-project}
GOOGLE_CLOUD_LOCATION={gcloud-region}
```

## Evaluation

### Evaluating a Trained Product-Price Associator

To evaluate a product-price associator, first fill out an `AssociatorEvaluationConfig` (see the [specifications](src/price_net/configs.py) for details). Then, run the [associator evaluation script](src/price_net/association/evaluate.py) via:

```bash
uv run evaluate_associator --config path/to/your/eval/config.yaml
```

This script will follow the logging settings specified in the config (WandB vs. local). It will also save evaluation metrics to an `association_metrics.yaml` file in the specified results directory.

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

### Evaluating an End-to-End Price Attribution System

To evaluate an end-to-end price attribution system, run the [end-to-end evaluation script](src/price_net/scripts/evaluate_e2e.py):

```bash
uv run evaluate_e2e --config path/to/config.yaml
```

This script can either evaluate a VLM-based system or a variant of `PriceLens` (our proposed modular pipeline for price attribution). Since VLM inference is expensive, we recommend cacheing product-price attributions from VLMs in a JSON file (i.e. as a list of json-ified `PriceAttribution` objects, see [the exact schema](src/price_net/schema.py)). Then, in your `AttributionEvaluationConfig` ([definition here](src/price_net/configs.py)), indicate where to find these attributions by including the filepath with the `cached_attributions_path` key.

If evaluating a `PriceLens` stack, the script expects the config to specify a pre-computed set of product detections (labeled by product identity) and price detections (labeled with an extracted price string). These should be stored in a JSON file, with the path indicated under the `cached_detections_path` key in the config. A path to a price associator eval config should also be provided under `associator_eval_config_path` (this sub-config gives instructions for how to load and run the associator).

Here is what an `AttributionEvaluationConfig` file would look like for evaluating a VLM:

```yaml
dataset_dir: path/to/dataset/dir
results_dir: path/to/results/dir
cached_attributions_path: path/to/attributions.json
```

Here is what the config would look like for evaluating a `PriceLens` stack:

```yaml
dataset_dir: path/to/dataset/dir
results_dir: path/to/results/dir
cached_detections_path: path/to/detections.json
associator_eval_config_path: path/to/associator/eval/config.yaml
# Used to determine what is considered a valid "association"
threshold: 0.5
```

Evaluation metrics will be saved in an `attribution_metrics.yaml` file in the results directory listed in your attribution config.

## Development

### Managing Dependencies

To add a new dependency to the project, run `uv add <package-name>`. This will install the dependency into uv's managed .venv and automatically update the `pyproject.toml` file and the `uv.lock` file, ensuring that the dependency is available for all users of the project who run `uv sync`.

To remove a dependency, run `uv remove <package-name>`. This will perform the reverse of `uv add` (including updating the `pyproject.toml` and `uv.lock` files).

See [uv's documentation](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) for more details.
