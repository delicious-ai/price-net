import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from pathlib import Path

import polars as pl
import torch
from price_net.association.transforms import ConcatenateBoundingBoxes
from price_net.association.transforms import InputTransform
from price_net.enums import Aggregation
from price_net.schema import PriceAssociationScene
from price_net.utils import parse_bboxes
from torch.utils.data import Dataset
from tqdm import tqdm


class PriceAssociationDataset(Dataset):
    INSTANCES_SCHEMA = {
        "scene_id": pl.String,
        "price_id": pl.String,
        "group_id": pl.String,
        "price_bbox": pl.Array(pl.Float32, 5),
        "product_bbox": pl.Array(pl.Float32, 5),
        "is_associated": pl.Int8,
    }
    IMAGES_DIR = "images"
    DEPTH_MAPS_DIR = "depth-maps"
    RAW_PRICE_SCENES_FNAME = "raw_price_scenes.json"
    INSTANCES_FNAME = "instances.parquet"

    def __init__(
        self,
        root_dir: str | Path,
        input_transform: InputTransform = ConcatenateBoundingBoxes(),
        aggregation: Aggregation = Aggregation.NONE,
        use_depth: bool = True,
    ):
        """Initialize a `PriceAssociationDataset`.

        Args:
            root_dir (str | Path): Root directory where the dataset is stored.
            input_transform (InputTransform, optional): Transform to apply to each input of the dataset when `__getitem__` is called. Defaults to `ConcatenateBoundingBoxes`.
            aggregation (Aggregation, optional): Determines how we parse instances for `__getitem__`. Defaults to Aggregation.NONE (each potential product-price association pair is returned for a scene).
            use_depth (bool, optional): Whether/not to use depth if aggregating by "closest_per_group".
        """
        self.root_dir = Path(root_dir)
        self.images_dir = self.root_dir / "images"
        self.depth_maps_dir = self.root_dir / "depth-maps"
        self.price_scenes_file = self.root_dir / self.RAW_PRICE_SCENES_FNAME
        self._check_expected_files_exist()

        self.input_transform = input_transform
        self.aggregation = aggregation
        self.use_depth = use_depth
        self.instances = self._get_instances()
        self.scene_ids = []
        self.scene_id_to_indices = defaultdict(list)
        for i, scene_id in enumerate(self.instances["scene_id"]):
            if scene_id not in self.scene_id_to_indices:
                self.scene_ids.append(scene_id)
            self.scene_id_to_indices[scene_id].append(i)

    def __getitem__(self, idx: int):
        scene_id = self.scene_ids[idx]
        rows = self.instances[self.scene_id_to_indices[scene_id]]
        prod_bboxes = torch.tensor(rows["product_bbox"], dtype=torch.float32)
        price_bboxes = torch.tensor(rows["price_bbox"], dtype=torch.float32)
        y = torch.tensor(rows["is_associated"], dtype=torch.float32)
        x = self.input_transform(prod_bboxes, price_bboxes)
        return x, y, scene_id

    def __len__(self):
        return len(self.scene_ids)

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
        instances = self._prepare_instances(instances)
        return instances

    def _prepare_instances(self, instances: pl.DataFrame):
        if self.aggregation == Aggregation.CLOSEST_PER_GROUP:
            centroid_end_dim = 3 if self.use_depth else 2
            instances = instances.with_columns(
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
            instances = (
                instances.sort("centroid_dist", "price_id")
                .group_by(["scene_id", "price_id", "group_id"], maintain_order=True)
                .agg(
                    pl.first("product_bbox"),
                    pl.first("price_bbox"),
                    pl.first("is_associated"),
                )
            )
        instances = instances.sort("scene_id")
        return instances

    def _process_scene(self, raw_scene: dict):
        scene = PriceAssociationScene(**raw_scene)
        scene_id = scene.scene_id

        product_bboxes, prod_ids, prod_id_to_idx = parse_bboxes(scene.product_bboxes)
        price_bboxes, price_ids, _ = parse_bboxes(scene.price_bboxes)

        price_id_to_assoc_products = {
            price_id: group.product_bbox_ids
            for group in scene.price_groups
            for price_id in group.price_bbox_ids
        }
        scene_instances = []
        for price_idx, price_bbox in enumerate(price_bboxes):
            price_id = price_ids[price_idx]
            assoc_prod_ids = price_id_to_assoc_products.get(price_id, set())

            for product_group in scene.product_groups:
                group_id = product_group.group_id
                group_indices = [
                    prod_id_to_idx[box_id] for box_id in product_group.product_bbox_ids
                ]
                group_bboxes = product_bboxes[group_indices]

                for prod_idx_in_group, prod_bbox in enumerate(group_bboxes):
                    prod_id = prod_ids[group_indices[prod_idx_in_group]]
                    instance = {
                        "scene_id": scene_id,
                        "price_id": price_id,
                        "group_id": group_id,
                        "price_bbox": price_bbox.tolist(),
                        "product_bbox": prod_bbox.tolist(),
                        "is_associated": int(prod_id in assoc_prod_ids),
                    }
                    scene_instances.append(instance)
        return scene_instances
