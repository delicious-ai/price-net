import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
import yaml
from price_net.configs import EvaluationConfig
from price_net.configs import TrainingConfig
from price_net.datamodule import PriceAssociationDataModule
from price_net.models import PriceAssociatorLightningModule
from price_net.schema import PriceAssociationScene
from sklearn.metrics import average_precision_score
from sklearn.metrics import jaccard_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


@torch.inference_mode()
def evaluate(config: EvaluationConfig):
    model = PriceAssociatorLightningModule.load_from_checkpoint(config.ckpt_path).eval()
    with open(config.trn_config_path, "r") as f:
        training_config = TrainingConfig(**yaml.safe_load(f))

    datamodule = PriceAssociationDataModule(
        data_dir=training_config.dataset_dir,
        input_reduction=training_config.model.input_reduction,
        prediction_strategy=training_config.model.prediction_strategy,
        featurization_method=training_config.model.featurization_method,
        use_depth=training_config.model.use_depth,
    )
    datamodule.setup("test")

    dataset = datamodule.test
    instances_df = dataset.instances
    with open(dataset.root_dir / dataset.RAW_PRICE_SCENES_FNAME, "r") as f:
        raw_scenes = [PriceAssociationScene(**scene) for scene in json.load(f)]
        raw_scenes = {scene.scene_id: scene for scene in raw_scenes}

    y_true = []
    y_score = []
    sample_weights = []
    for i in tqdm(range(len(dataset))):
        X, y = dataset[i]

        scene_id = str(instances_df[i]["scene_id"][0]).split("__")[0]
        scene = raw_scenes[scene_id]
        id_to_product_group = {group.group_id: group for group in scene.product_groups}
        group_ids = dataset.instances[i]["group_id"].to_list()
        pred_probs = model.forward(X.unsqueeze(0)).sigmoid().flatten()

        for j in range(len(X)):
            actually_associated = bool(y[j])
            assoc_prob = pred_probs[j]
            group_id = group_ids[j]
            num_in_group = len(id_to_product_group[group_id].product_bbox_ids)

            y_true.append(actually_associated)
            y_score.append(assoc_prob)
            sample_weights.append(num_in_group)

    results = {}
    thresholds = np.linspace(0.1, 0.9, num=9)
    for t in thresholds:
        y_pred = np.array(y_score) > t
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
    with open(config.results_dir / "eval_metrics.yaml", "w") as f:
        yaml.safe_dump(results, f)


def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        eval_config = EvaluationConfig(**yaml.safe_load(f))
    evaluate(config=eval_config)


if __name__ == "__main__":
    main()
