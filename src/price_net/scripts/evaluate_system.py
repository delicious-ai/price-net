import json
from argparse import ArgumentParser
from itertools import product
from pathlib import Path
from pprint import pprint

import yaml
from price_net.association.dataset import PriceAssociationDataset
from price_net.enums import PriceType
from price_net.schema import PriceAssociationScene
from price_net.schema import UPCPrice


IGNORED_PRICE_TYPES = (PriceType.UNKNOWN, PriceType.MISC)
PRICE_TYPES_WITH_UNIT_PRICE = (PriceType.STANDARD, PriceType.BULK_OFFER)


def evaluate(dataset_dir: Path, test_predictions_path: Path, results_dir: Path):
    raw_scenes_path = (
        dataset_dir / "test" / PriceAssociationDataset.RAW_PRICE_SCENES_FNAME
    )
    with open(raw_scenes_path, "r") as f:
        scenes = [PriceAssociationScene(**x) for x in json.load(f)]

    actual_pairs = {
        UPCPrice(
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

    with open(test_predictions_path, "r") as f:
        pred_pairs = {UPCPrice(**x) for x in json.load(f)}
    pred_pairs = {
        x for x in pred_pairs if x.price.price_type not in IGNORED_PRICE_TYPES
    }

    tp = pred_pairs & actual_pairs
    fp = pred_pairs - actual_pairs
    fn = actual_pairs - pred_pairs

    precision = len(tp) / (len(tp) + len(fp)) if (tp or fp) else 0.0
    recall = len(tp) / (len(tp) + len(fn)) if (tp or fn) else 0.0
    f1 = (
        (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    )

    scene_id_upc_to_actual_price = {
        (x.scene_id, x.upc): x.price
        for x in actual_pairs
        if x.price.price_type in PRICE_TYPES_WITH_UNIT_PRICE
    }
    scene_id_upc_to_pred_price = {
        (x.scene_id, x.upc): x.price
        for x in pred_pairs
        if x.price.price_type in PRICE_TYPES_WITH_UNIT_PRICE
    }
    matched_keys = set(scene_id_upc_to_actual_price.keys()) & set(
        scene_id_upc_to_pred_price.keys()
    )
    abs_errors = [
        abs(
            scene_id_upc_to_actual_price[k].unit_price
            - scene_id_upc_to_pred_price[k].unit_price
        )
        for k in matched_keys
    ]
    mae = sum(abs_errors) / len(abs_errors) if abs_errors else None
    metrics = {"precision": precision, "recall": recall, "f1": f1, "mae": mae}

    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "attribution_metrics.yaml", "w") as f:
        yaml.safe_dump(metrics, f)

    pprint(metrics)


def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--test-predictions", type=Path)
    parser.add_argument("--results-dir", type=Path)
    args = parser.parse_args()
    evaluate(
        dataset_dir=args.dataset_dir,
        test_predictions_path=args.test_predictions,
        results_dir=args.results_dir,
    )


if __name__ == "__main__":
    main()
