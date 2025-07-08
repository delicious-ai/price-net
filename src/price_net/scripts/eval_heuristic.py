import ast
import json
from argparse import ArgumentParser
from itertools import product
from pathlib import Path
from pprint import pprint
from typing import Literal

import yaml
from price_net.dataset import PriceAssociationDataset
from price_net.enums import HeuristicType
from price_net.heuristics import HEURISTIC_REGISTRY
from price_net.schema import PriceAssociationScene
from tqdm import tqdm


def parse_unknown_args(unknown_args: list[str]):
    it = iter(unknown_args)
    kwargs = {}
    for key in it:
        if not key.startswith("--"):
            raise ValueError(f"Unexpected argument format: {key}")
        key = key[2:].replace("-", "_")
        value = next(it)
        try:
            value = ast.literal_eval(value)
        except Exception:
            pass
        kwargs[key] = value
    return kwargs


def main(
    dataset_dir: Path,
    heuristic_type: HeuristicType,
    results_dir: Path,
    split: Literal["val", "test"] = "test",
    heuristic_kwargs: dict = {},
):
    method = HEURISTIC_REGISTRY[heuristic_type](**heuristic_kwargs)
    raw_scenes_path = (
        dataset_dir / split / PriceAssociationDataset.RAW_PRICE_SCENES_FNAME
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
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    args, unknown_args = parser.parse_known_args()
    heuristic_kwargs = parse_unknown_args(unknown_args)
    main(
        dataset_dir=args.dataset_dir,
        heuristic_type=args.heuristic,
        results_dir=args.results_dir,
        split=args.split,
        heuristic_kwargs=heuristic_kwargs,
    )
