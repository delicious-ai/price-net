from __future__ import annotations

import json
import re
from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import polars as pl
import torch
import yaml
from price_net.association.datamodule import PriceAssociationDataModule
from price_net.association.dataset import PriceAssociationDataset
from price_net.association.models import PriceAssociatorLightningModule
from price_net.configs import AssociatorEvaluationConfig
from price_net.configs import AssociatorTrainingConfig
from price_net.configs import AttributionEvaluationConfig
from price_net.enums import Aggregation
from price_net.enums import PredictionStrategy
from price_net.enums import PriceType
from price_net.schema import PriceAttribution
from price_net.schema import PriceScene
from price_net.schema import SceneDetections
from tqdm import tqdm


IGNORED_PRICE_TYPES = (PriceType.UNKNOWN, PriceType.MISC)


def get_price_type(price: str) -> PriceType:
    """Infer the price type from a price string."""
    standard_pattern = r"^\$\d+\.\d{2}$"
    bulk_offer_pattern = r"^\d+\s*/\s*\$\d+\.\d{2}$"
    bogo_pattern = r"^Buy\s+\d+,\s*get\s+\d+\s+for\s+\$\d+\.\d{2}$"
    if bool(re.match(standard_pattern, price)):
        return PriceType.STANDARD
    elif bool(re.match(bulk_offer_pattern, price)):
        return PriceType.BULK_OFFER
    elif bool(re.match(bogo_pattern, price)):
        return PriceType.BUY_X_GET_Y_FOR_Z
    elif price == "Unknown":
        return PriceType.UNKNOWN
    else:
        return PriceType.MISC


@torch.inference_mode()
def _get_learned_attributions(
    associator_eval_config: AssociatorEvaluationConfig,
    detections: list[SceneDetections],
    threshold: float,
) -> set[tuple[str, str, str]]:
    model = PriceAssociatorLightningModule.load_from_checkpoint(
        associator_eval_config.ckpt_path
    ).eval()
    device = model.device
    with open(associator_eval_config.trn_config_path, "r") as f:
        training_config = AssociatorTrainingConfig(**yaml.safe_load(f))

    aggregation = training_config.model.aggregation
    use_depth = training_config.model.featurization.use_depth
    transform = PriceAssociationDataModule._get_transform(
        config=training_config.model.featurization
    )
    attributions = set()
    for scene_detections in tqdm(detections):
        scene_id = scene_detections.scene_id

        # We form all possible product-price pairs.
        product_bboxes, product_labels = [], []
        price_bboxes, price_labels, price_ids = [], [], []
        for prod_bbox in scene_detections.product_bboxes:
            assert prod_bbox.label is not None
            for price_bbox in scene_detections.price_bboxes:
                assert price_bbox.label is not None
                product_bboxes.append(prod_bbox.to_tensor().tolist())
                product_labels.append(prod_bbox.label)
                price_bboxes.append(price_bbox.to_tensor().tolist())
                price_labels.append(price_bbox.label)
                price_ids.append(price_bbox.id)

        if not product_bboxes or not price_bboxes:
            continue
        else:
            # We assemble the potential pairs into a structured format.
            df = pl.DataFrame(
                {
                    "product_bbox": product_bboxes,
                    "price_bbox": price_bboxes,
                    "upc": product_labels,
                    "price": price_labels,
                    "price_id": price_ids,
                },
                schema_overrides={
                    "product_bbox": pl.Array(pl.Float32, 5),
                    "price_bbox": pl.Array(pl.Float32, 5),
                },
            )

            # We perform any necessary aggregation (e.g., closest per product group).
            if aggregation == Aggregation.CLOSEST_PER_GROUP:
                centroid_end_dim = 3 if use_depth else 2
                df = (
                    df.with_columns(
                        pl.struct("price_bbox", "product_bbox")
                        .map_elements(
                            lambda s: sum(
                                (a - b) ** 2
                                for a, b in zip(
                                    s["price_bbox"][:centroid_end_dim],
                                    s["product_bbox"][:centroid_end_dim],
                                )
                            )
                            ** 0.5,
                            return_dtype=pl.Float32,
                        )
                        .alias("centroid_dist")
                    )
                    .sort("centroid_dist", "price_id")
                    .group_by(["price_id", "upc"], maintain_order=True)
                    .agg(
                        pl.first("product_bbox"),
                        pl.first("price_bbox"),
                        pl.first("price"),
                    )
                )

        prod_bboxes_tensor = torch.tensor(df["product_bbox"], dtype=torch.float32)
        price_bboxes_tensor = torch.tensor(df["price_bbox"], dtype=torch.float32)
        upcs = df["upc"].to_list()
        prices = df["price"].to_list()
        X = transform(prod_bboxes_tensor, price_bboxes_tensor).to(device)
        if training_config.model.prediction_strategy == PredictionStrategy.JOINT:
            X = X.unsqueeze(0)

        pred_logits: torch.Tensor = model(X)
        pred_probs = pred_logits.sigmoid().flatten()
        for i in range(len(pred_probs)):
            if pred_probs[i] > threshold:
                upc = upcs[i]
                price = prices[i]
                price_type = get_price_type(price)
                if price_type not in IGNORED_PRICE_TYPES:
                    attributions.add((scene_id, upc, price))
    return attributions


def evaluate(config: AttributionEvaluationConfig, threshold: float = 0.5):
    dataset_dir = config.dataset_dir
    results_dir = config.results_dir

    raw_scenes_path = (
        dataset_dir / config.split / PriceAssociationDataset.RAW_PRICE_SCENES_FNAME
    )
    with open(raw_scenes_path, "r") as f:
        scenes = [PriceScene(**x) for x in json.load(f)]

    actual_attributions = {
        (x.scene_id, x.upc, x.price.price_text)
        for x in (
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
        )
        if x.price.price_type not in IGNORED_PRICE_TYPES
    }

    # Path 1: cached attributions were provided.
    if config.cached_attributions_path is not None:
        with open(config.cached_attributions_path, "r") as f:
            pred_attributions = {
                (x.scene_id, x.upc, x.price.price_text)
                for x in (PriceAttribution(**x) for x in json.load(f))
                if x.price.price_text not in IGNORED_PRICE_TYPES
            }
    # Path 2: we are evaluating a PriceLens stack (detection + extraction already run).
    else:
        with open(config.cached_detections_path, "r") as f:
            detections = [SceneDetections(**x) for x in json.load(f)]
        assert config.associator_eval_config_path is not None
        with open(config.associator_eval_config_path, "r") as f:
            associator_eval_config = AssociatorEvaluationConfig(**yaml.safe_load(f))
        pred_attributions = _get_learned_attributions(
            associator_eval_config=associator_eval_config,
            detections=detections,
            threshold=threshold,
        )

    tp = pred_attributions & actual_attributions
    fp = pred_attributions - actual_attributions
    fn = actual_attributions - pred_attributions

    precision = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 0.0
    recall = len(tp) / (len(tp) + len(fn)) if (tp or fn) else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    )

    metrics = {"precision": precision, "recall": recall, "f1": f1}

    results_dir.mkdir(parents=True, exist_ok=True)
    result_file = Path(results_dir / "attribution_metrics.yaml")
    with open(result_file, "w") as f:
        yaml.safe_dump(metrics, f)
    print(metrics)


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = AttributionEvaluationConfig(**yaml.safe_load(f))
    evaluate(config=config, threshold=args.threshold)


if __name__ == "__main__":
    main()
