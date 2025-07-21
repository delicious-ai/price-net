from __future__ import annotations

import json
import statistics
from argparse import ArgumentParser
from collections import defaultdict
from itertools import product
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import yaml
from price_net.association.configs import AssociatorEvaluationConfig
from price_net.association.configs import AssociatorTrainingConfig
from price_net.association.datamodule import PriceAssociationDataModule
from price_net.association.dataset import PriceAssociationDataset
from price_net.association.heuristics import HEURISTIC_REGISTRY
from price_net.association.models import PriceAssociatorLightningModule
from price_net.enums import HeuristicType
from price_net.enums import PredictionStrategy
from price_net.enums import PriceType
from price_net.schema import PriceAssociationScene
from price_net.schema import PriceAttribution
from price_net.schema import PriceBuilder
from price_net.schema import PriceModelType
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator
from tqdm import tqdm


class AttributionEvaluationConfig(BaseModel):
    dataset_dir: Path
    results_dir: Path

    cached_attributions_path: Path | None = None

    extracted_prices_path: Path | None = None
    associator_eval_config_path: Path | None = None
    # Only used for probabilistic associators.
    threshold: float = Field(ge=0.0, le=1.0, default=0.5)
    heuristic: HeuristicType | None = None
    heuristic_kwargs: dict = {}

    @model_validator(mode="after")
    def check_mutually_exclusive_modes(self) -> AttributionEvaluationConfig:
        mode_flags = [
            bool(self.cached_attributions_path),
            bool(self.heuristic),
            bool(self.associator_eval_config_path),
        ]
        if sum(mode_flags) != 1:
            raise ValueError(
                "You must specify exactly one of: "
                "`cached_attributions_path`, `heuristic`, or `associator_eval_config_path`."
            )

        if (
            self.heuristic or self.associator_eval_config_path
        ) and not self.extracted_prices_path:
            raise ValueError(
                "`extracted_prices_path` is required when using a heuristic or associator."
            )

        return self


IGNORED_PRICE_TYPES = (PriceType.UNKNOWN, PriceType.MISC)
PRICE_TYPES_WITH_UNIT_PRICE = (PriceType.STANDARD, PriceType.BULK_OFFER)


def _get_heuristic_attributions(
    heuristic_type: HeuristicType,
    heuristic_kwargs: dict,
    scenes: list[PriceAssociationScene],
    prices: dict[str, PriceModelType],
) -> set[PriceAttribution]:
    heuristic = HEURISTIC_REGISTRY[heuristic_type](**heuristic_kwargs)
    attributions = {
        PriceAttribution(
            scene_id=scene.scene_id,
            upc=scene.products[prod_id],
            price=prices[price_id],
        )
        for scene in tqdm(scenes)
        for group in heuristic(scene)
        for prod_id, price_id in product(group.product_bbox_ids, group.price_bbox_ids)
    }
    return attributions


@torch.inference_mode()
def _get_learned_attributions(
    associator_eval_config: AssociatorEvaluationConfig,
    prices: dict[str, PriceModelType],
    threshold: float,
) -> set[PriceAttribution]:
    model = PriceAssociatorLightningModule.load_from_checkpoint(
        associator_eval_config.ckpt_path
    ).eval()
    device = model.device
    with open(associator_eval_config.trn_config_path, "r") as f:
        training_config = AssociatorTrainingConfig(**yaml.safe_load(f))

    datamodule = PriceAssociationDataModule(
        data_dir=training_config.dataset_dir,
        aggregation=training_config.model.aggregation,
        prediction_strategy=training_config.model.prediction_strategy,
        featurization_config=training_config.model.featurization,
    )
    datamodule.setup("test")
    dataset = datamodule.test

    attributions = set()
    for i in tqdm(range(len(dataset))):
        X, _, scene_id = dataset[i]
        X = X.to(device)
        if training_config.model.prediction_strategy == PredictionStrategy.JOINT:
            X = X.unsqueeze(0)
        scene_indices = dataset.scene_id_to_indices[scene_id]
        group_ids = dataset.instances[scene_indices]["group_id"].to_list()
        price_ids = dataset.instances[scene_indices]["price_id"].to_list()

        pred_logits: torch.Tensor = model(X)
        pred_probs = pred_logits.sigmoid().flatten()

        for j in range(len(X)):
            if pred_probs[j] > threshold:
                upc = group_ids[j]
                price = prices[price_ids[j]]
                attributions.add(
                    PriceAttribution(scene_id=scene_id, upc=upc, price=price)
                )
    return attributions


def evaluate(config: AttributionEvaluationConfig):
    dataset_dir = config.dataset_dir
    results_dir = config.results_dir

    raw_scenes_path = (
        dataset_dir / "test" / PriceAssociationDataset.RAW_PRICE_SCENES_FNAME
    )
    with open(raw_scenes_path, "r") as f:
        scenes = [PriceAssociationScene(**x) for x in json.load(f)]

    actual_attributions = {
        PriceAttribution(
            scene_id=scene.scene_id,
            upc=scene.products[product_id],
            price=scene.prices[price_id],
        )
        for scene in scenes
        for group in scene.price_groups
        for product_id, price_id in product(
            group.product_bbox_ids, group.price_bbox_ids
        )
        if scene.prices[price_id].price_type not in IGNORED_PRICE_TYPES
    }

    # Path 1: cached attributions were provided.
    if config.cached_attributions_path is not None:
        with open(config.cached_attributions_path, "r") as f:
            pred_attributions = {PriceAttribution(**x) for x in json.load(f)}
            pred_attributions = {
                x
                for x in tqdm(pred_attributions)
                if x.price.price_type not in IGNORED_PRICE_TYPES
            }

    else:
        with open(config.extracted_prices_path, "r") as f:
            extracted_prices: dict[str, PriceModelType] = {
                x["price_id"]: PriceBuilder(price=x["price"]).price
                for x in json.load(f)
            }
        if config.heuristic is not None:
            # Path 2: Use a heuristic associator for getting attributions.
            pred_attributions = _get_heuristic_attributions(
                heuristic_type=config.heuristic,
                heuristic_kwargs=config.heuristic_kwargs,
                scenes=scenes,
                prices=extracted_prices,
            )
        else:
            # Path 3: Use a learned associator for getting attributions.
            assert config.associator_eval_config_path is not None
            with open(config.associator_eval_config_path, "r") as f:
                associator_eval_config = AssociatorEvaluationConfig(**yaml.safe_load(f))
            pred_attributions = _get_learned_attributions(
                associator_eval_config=associator_eval_config,
                prices=extracted_prices,
            )

    tp = pred_attributions & actual_attributions
    fp = pred_attributions - actual_attributions
    fn = actual_attributions - pred_attributions

    precision = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 0.0
    recall = len(tp) / (len(tp) + len(fn)) if (tp or fn) else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    )

    scene_upc_type_to_actual_price = defaultdict(list)
    scene_upc_type_to_pred_price = defaultdict(list)

    for x in actual_attributions:
        price_type = x.price.price_type
        if price_type in PRICE_TYPES_WITH_UNIT_PRICE:
            assert x.price.unit_price is not None
            scene_upc_type_to_actual_price[(x.scene_id, x.upc, price_type)].append(
                x.price.unit_price
            )
    for x in pred_attributions:
        price_type = x.price.price_type
        if price_type in PRICE_TYPES_WITH_UNIT_PRICE:
            assert x.price.unit_price is not None
            scene_upc_type_to_pred_price[(x.scene_id, x.upc, price_type)].append(
                x.price.unit_price
            )

    abs_errors = []
    for key in set(scene_upc_type_to_actual_price) & set(scene_upc_type_to_pred_price):
        actuals = scene_upc_type_to_actual_price[key]
        preds = scene_upc_type_to_pred_price[key]

        for pred in preds:
            abs_errors.append(min(np.abs(np.array(actuals) - pred)))

    mae = np.mean(abs_errors).item() if abs_errors else None

    metrics = {"precision": precision, "recall": recall, "f1": f1, "mae": mae}

    results_dir.mkdir(parents=True, exist_ok=True)
    result_file = Path(results_dir / "attribution_metrics.yaml")
    if result_file.exists():
        exp = yaml.safe_load(open(result_file, "r"))
        run_id = max(int(k) for k in exp["runs"].keys()) + 1
        exp["runs"][run_id] = metrics
        exp["overall"] = {
            "mean": {
                "precision": statistics.mean(
                    [exp["runs"][k]["precision"] for k in exp["runs"].keys()]
                ),
                "recall": statistics.mean(
                    [exp["runs"][k]["recall"] for k in exp["runs"].keys()]
                ),
                "f1": statistics.mean(
                    [exp["runs"][k]["f1"] for k in exp["runs"].keys()]
                ),
                "mae": statistics.mean(
                    [exp["runs"][k]["mae"] for k in exp["runs"].keys()]
                ),
            },
            "std": {
                "precision": statistics.stdev(
                    [exp["runs"][k]["precision"] for k in exp["runs"].keys()]
                ),
                "recall": statistics.stdev(
                    [exp["runs"][k]["recall"] for k in exp["runs"].keys()]
                ),
                "f1": statistics.stdev(
                    [exp["runs"][k]["f1"] for k in exp["runs"].keys()]
                ),
                "mae": statistics.stdev(
                    [exp["runs"][k]["mae"] for k in exp["runs"].keys()]
                ),
            },
        }
    else:
        exp = {"runs": {1: metrics}}
    with open(result_file, "w") as f:
        yaml.safe_dump(exp, f)
    pprint(exp)


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = AttributionEvaluationConfig(**yaml.safe_load(f))
    evaluate(config=config)


if __name__ == "__main__":
    main()
