import json
from argparse import ArgumentParser
from itertools import product
from pathlib import Path
from pprint import pprint

import yaml
from price_net.dataset import PriceAssociationDataset
from price_net.enums import HeuristicType
from price_net.heuristics import HEURISTIC_REGISTRY
from price_net.schema import PriceAssociationScene
from tqdm import tqdm


def main(dataset_dir: Path, heuristic_type: HeuristicType, results_dir: Path):
    method = HEURISTIC_REGISTRY[heuristic_type]()
    raw_scenes_path = (
        dataset_dir / "test" / PriceAssociationDataset.RAW_PRICE_SCENES_FNAME
    )
    with open(raw_scenes_path, "r") as f:
        scenes = [PriceAssociationScene(**x) for x in json.load(f)]

    pred_pairs = set()
    actual_pairs = set()
    for scene in tqdm(scenes, desc=f"Evaluating '{heuristic_type.value}' heuristic..."):
        pred_price_groups = method(scene)
        pred_prod_price_pairs = {
            (prod_id, price_id)
            for group in pred_price_groups
            for prod_id, price_id in product(
                group.product_bbox_ids, group.price_bbox_ids
            )
        }
        actual_prod_price_pairs = {
            (prod_id, price_id)
            for group in scene.price_groups
            for prod_id, price_id in product(
                group.product_bbox_ids, group.price_bbox_ids
            )
        }
        pred_pairs.update(pred_prod_price_pairs)
        actual_pairs.update(actual_prod_price_pairs)

    tp = pred_pairs & actual_pairs
    fp = pred_pairs - actual_pairs
    fn = actual_pairs - pred_pairs

    precision = len(tp) / (len(tp) + len(fp))
    recall = len(tp) / (len(tp) + len(fn))
    f1 = (2 * precision * recall) / (precision + recall)

    metrics = {"precision": precision, "recall": recall, "f1": f1}
    results_dir.mkdir(exist_ok=True, parents=True)
    with open(results_dir / "eval_metrics.yaml", "w") as f:
        yaml.safe_dump(metrics, f)
    pprint(metrics)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--heuristic", type=HeuristicType)
    parser.add_argument("--results-dir", type=Path)
    args = parser.parse_args()
    main(args.dataset_dir, args.heuristic, args.results_dir)
