import json
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path
from typing import Callable

import polars as pl
import torch
from price_net.enums import FeaturizationMethod
from price_net.enums import InputGranularity
from price_net.schema import PriceAttributionScene
from price_net.utils import parse_bboxes
from torch.utils.data import Dataset
from tqdm import tqdm


Transform = Callable[[torch.Tensor], torch.Tensor] | None


class PriceAttributionDataset(Dataset):
    FEATURE_DIM = 10
    INSTANCES_SCHEMA = {
        "scene_id": pl.String,
        "price_id": pl.String,
        "group_id": pl.String,
        "x": pl.Array(pl.Float32, FEATURE_DIM),
        "y": pl.Int8,
    }
    IMAGES_DIR = "images"
    DEPTH_MAPS_DIR = "depth-maps"
    RAW_PRICE_SCENES_FNAME = "raw_price_scenes.json"
    INSTANCES_FNAME = "instances.parquet"

    def __init__(
        self,
        root_dir: str | Path,
        input_transform: Transform = None,
        output_transform: Transform = None,
        input_granularity: InputGranularity = InputGranularity.PAIRWISE,
        featurization_method: FeaturizationMethod = FeaturizationMethod.CENTROID_DIFF,
    ):
        """Initialize a `PriceAttributionDataset`.

        Args:
            root_dir (str | Path): Root directory where the dataset is stored.
            input_transform (Transform | None, optional): Transform to apply to each input of the dataset when `__getitem__` is called. Defaults to None.
            output_transform (Transform | None, optional): Transform to apply to each output of the dataset when `__getitem__` is called. Defaults to None.
            input_granularity (InputGranularity, optional): Determines the shape of inputs returned by `__getitem__`. Defaults to InputType.PAIRWISE (each potential association treated independently).
            featurization_method (FeaturizationMethod, optional): The method to use when building features for a potential prod-price edge. Defaults to `FeaturizationMethod.CENTROID_DIFF` (the feature vector is the centroid delta concatenated with the price bbox).
        """
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.depth_maps_dir = self.root_dir / "depth-maps"
        self.price_scenes_file = self.root_dir / self.RAW_PRICE_SCENES_FNAME
        self._check_expected_files_exist()

        self.input_transform = input_transform
        self.output_transform = output_transform
        self.input_granularity = input_granularity
        self.featurization_method = featurization_method
        self.instances = self._get_instances()

    def __getitem__(self, idx: int):
        row = self.instances.row(idx, named=True)
        x = torch.tensor(row["x"], dtype=torch.float32)
        y = torch.tensor(row["y"], dtype=torch.float32)

        if self.input_transform:
            x = self.input_transform(x)
        if self.output_transform:
            y = self.output_transform(y)

        return x, y

    def __len__(self):
        return len(self.instances)

    def _check_expected_files_exist(self):
        if not self.price_scenes_file.exists():
            raise FileNotFoundError(f"Missing '{self.price_scenes_file}'.")
        for dir in (self.images_dir, self.depth_maps_dir):
            if not dir.exists():
                raise FileNotFoundError(f"Missing '{dir}' directory.")
            if not any(dir.iterdir()):
                raise FileNotFoundError(f"'{dir}' directory is empty.")

    def _get_instances(self) -> pl.DataFrame:
        instances_path = self.root_dir / self.INSTANCES_FNAME
        if instances_path.exists():
            instances = pl.read_parquet(instances_path)
        else:
            with open(self.price_scenes_file, "r") as f:
                raw_scenes = json.load(f)

            with ThreadPoolExecutor() as executor:
                instances = list(
                    tqdm(
                        executor.map(self._process_scene, raw_scenes),
                        total=len(raw_scenes),
                        desc="Processing scenes",
                    )
                )
            instances = pl.DataFrame(
                data=list(chain.from_iterable(instances)),
                orient="row",
                schema=self.INSTANCES_SCHEMA,
            )
            instances.write_parquet(instances_path)
        if self.input_granularity == InputGranularity.PAIRWISE:
            return instances
        elif self.input_granularity == InputGranularity.SCENE_LEVEL:
            return instances.group_by("scene_id").agg(
                pl.col("price_id"),
                pl.col("group_id"),
                pl.col("x"),
                pl.col("y"),
            )
        else:
            raise NotImplementedError(
                f"Unsupported input type: {self.input_granularity.value}"
            )

    def _process_scene(self, raw_scene: dict):
        scene = PriceAttributionScene(**raw_scene)
        scene_id = scene.scene_id

        product_bboxes, prod_ids, prod_id_to_idx = parse_bboxes(scene.product_bboxes)
        price_bboxes, price_ids, _ = parse_bboxes(scene.price_bboxes)

        # For each product group, find which centroid is nearest to each price tag.
        # Then deduce if this group goes with that price tag (0 or 1).
        # This is one instance.
        price_id_to_assoc_products = {
            price_id: group.product_bbox_ids
            for group in scene.price_groups
            for price_id in group.price_bbox_ids
        }
        scene_instances = []
        for product_group in scene.product_groups:
            group_id = product_group.group_id

            group_indices = [
                prod_id_to_idx[box_id] for box_id in product_group.product_bbox_ids
            ]
            group_bboxes = product_bboxes[group_indices]
            group_centroids = group_bboxes[:, :3]

            for idx, price_bbox in enumerate(price_bboxes):
                centroid_diffs = group_centroids - price_bbox[:3]
                idx_of_closest_prod = centroid_diffs.norm(dim=1).argmin()
                cluster_wh = group_bboxes[:, 3:].mean(dim=0)
                if self.featurization_method == FeaturizationMethod.CENTROID:
                    x = torch.cat(
                        [
                            group_centroids[idx_of_closest_prod],
                            cluster_wh,
                            price_bbox,
                        ]
                    )
                elif self.featurization_method == FeaturizationMethod.CENTROID_DIFF:
                    x = torch.cat(
                        [
                            centroid_diffs[idx_of_closest_prod],
                            cluster_wh,
                            price_bbox,
                        ]
                    )
                else:
                    raise NotImplementedError(
                        f"Unsupported featurizing strategy: {self.featurization_method.value}"
                    )

                id_of_closest_prod = prod_ids[group_indices[idx_of_closest_prod]]
                price_id = price_ids[idx]
                prod_ids_assoc_with_price = price_id_to_assoc_products.get(price_id, {})
                y = int(id_of_closest_prod in prod_ids_assoc_with_price)

                instance = {
                    "scene_id": scene_id,
                    "price_id": price_id,
                    "group_id": group_id,
                    "x": x.tolist(),
                    "y": y,
                }
                scene_instances.append(instance)
        return scene_instances
