import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import yaml
from price_net.association.datamodule import PriceAssociationDataModule
from price_net.association.models import PriceAssociatorLightningModule
from price_net.configs import AssociatorEvaluationConfig
from price_net.configs import AssociatorTrainingConfig
from price_net.enums import Aggregation
from price_net.enums import PredictionStrategy
from price_net.schema import PriceScene
from sklearn.metrics import average_precision_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


@torch.inference_mode()
def evaluate(
    config: AssociatorEvaluationConfig, split: Literal["val", "test"] = "test"
):
    model = PriceAssociatorLightningModule.load_from_checkpoint(config.ckpt_path).eval()
    device = model.device
    with open(config.trn_config_path, "r") as f:
        training_config = AssociatorTrainingConfig(**yaml.safe_load(f))

    datamodule = PriceAssociationDataModule(
        data_dir=training_config.dataset_dir,
        aggregation=training_config.model.aggregation,
        prediction_strategy=training_config.model.prediction_strategy,
        featurization_config=training_config.model.featurization,
    )
    if split == "test":
        datamodule.setup("test")
        dataset = datamodule.test
    else:
        datamodule.setup("validate")
        dataset = datamodule.val

    with open(dataset.root_dir / dataset.RAW_PRICE_SCENES_FNAME, "r") as f:
        raw_scenes = [PriceScene(**scene) for scene in json.load(f)]
    raw_scenes = {scene.scene_id: scene for scene in raw_scenes}

    y_true = []
    y_score = []
    sample_weights = []
    for i in tqdm(range(len(dataset))):
        X, y, scene_id = dataset[i]
        X = X.to(device)
        y = y.to(device)
        if training_config.model.prediction_strategy == PredictionStrategy.JOINT:
            X = X.unsqueeze(0)
        group_ids = dataset.instances[dataset.scene_id_to_indices[scene_id]][
            "group_id"
        ].to_list()

        scene = raw_scenes[scene_id]
        id_to_product_group = {group.group_id: group for group in scene.product_groups}

        pred_logits: torch.Tensor = model(X)
        pred_probs = pred_logits.sigmoid().flatten()

        for j in range(len(X)):
            y_true.append(y[j].item())
            y_score.append(pred_probs[j].item())

            if training_config.model.aggregation == Aggregation.CLOSEST_PER_GROUP:
                group_id = group_ids[j]
                num_in_group = len(id_to_product_group[group_id].product_bbox_ids)
                sample_weights.append(num_in_group)
            else:
                sample_weights.append(1)

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    results = {}
    thresholds = np.linspace(0.1, 0.9, num=9)
    for t in thresholds:
        y_pred = y_score > t
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true=y_true,
            y_pred=y_pred,
            average="binary",
            sample_weight=sample_weights,
        )
        jaccard = jaccard_score(
            y_true=y_true,
            y_pred=y_pred,
            average="binary",
            sample_weight=sample_weights,
        )
        results[f"precision@{t:.1f}"] = precision
        results[f"recall@{t:.1f}"] = recall
        results[f"f1@{t:.1f}"] = f1
        results[f"jaccard@{t:.1f}"] = jaccard
    results["roc-auc"] = roc_auc_score(
        y_true=y_true,
        y_score=y_score,
        sample_weight=sample_weights,
    )
    results["aupr"] = average_precision_score(
        y_true=y_true,
        y_score=y_score,
        sample_weight=sample_weights,
    )
    config.results_dir.mkdir(parents=True, exist_ok=True)
    with open(config.results_dir / "association_metrics.yaml", "w") as f:
        yaml.safe_dump(results, f)


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        eval_config = AssociatorEvaluationConfig(**yaml.safe_load(f))
    evaluate(config=eval_config, split=args.split)


if __name__ == "__main__":
    main()
