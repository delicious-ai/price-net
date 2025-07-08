from pathlib import Path
from typing import Callable
from typing import Literal

import lightning as L
import torch
from price_net.dataset import PriceAssociationDataset
from price_net.enums import FeaturizationMethod
from price_net.enums import InputReduction
from price_net.enums import PredictionStrategy
from price_net.transforms import ConcatenateBoundingBoxes
from price_net.transforms import ConcatenateWithCentroidDiff
from price_net.transforms import InputTransform
from price_net.utils import scene_level_collate_fn
from torch.utils.data import DataLoader


class PriceAssociationDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 1,
        num_workers: int = 0,
        input_reduction: InputReduction = InputReduction.NONE,
        prediction_strategy: PredictionStrategy = PredictionStrategy.MARGINAL,
        featurization_method: FeaturizationMethod = FeaturizationMethod.CENTROID,
        use_depth: bool = True,
    ):
        """Initialize a `PriceAssociationDataModule`.

        Args:
            data_dir (Path): The directory where the dataset is stored.
            batch_size (int, optional): The batch size to use for dataloaders. Defaults to 1.
            num_workers (int, optional): The number of workers to use for dataloaders. Defaults to 0.
            input_reduction (InputReduction, optional): Specifies how to reduce the raw set of product-price associations. Defaults to InputReduction.NONE.
            prediction_strategy (PredictionStrategy, optional): Specifies if predictions will be made marginally (treating each association independently) or jointly (across a scene). Defaults to PredictionStrategy.MARGINAL.
            featurization_method (FeaturizationMethod, optional): Specifies how to build feature representations of a possible product-price association. Defaults to FeaturizationMethod.CENTROID (pure centroids of prod/price bboxes are used).
            use_depth (bool, optional): Whether/not to use inferred depths in feature representation. Defaults to True.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.input_reduction = input_reduction
        self.prediction_strategy = prediction_strategy
        self.featurization_method = featurization_method
        self.use_depth = use_depth
        self.transform = self._get_transform()
        self.collate_fn = self._get_collate_fn()

    def setup(self, stage: Literal["fit", "validate", "test", "predict"]):
        if stage == "fit":
            self.train = self._init_dataset_split("train")
            self.val = self._init_dataset_split("val")
            self.test = self._init_dataset_split("test")
        elif stage == "validate":
            self.val = self._init_dataset_split("val")
        else:
            self.test = self._init_dataset_split("test")

    def _init_dataset_split(
        self, split: Literal["train", "val", "test"]
    ) -> PriceAssociationDataset:
        return PriceAssociationDataset(
            root_dir=self.data_dir / split,
            input_transform=self.transform,
            input_reduction=self.input_reduction,
            prediction_strategy=self.prediction_strategy,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

    def _get_transform(self) -> InputTransform:
        if self.featurization_method == FeaturizationMethod.CENTROID:
            concatenation_op = ConcatenateBoundingBoxes()
        else:
            concatenation_op = ConcatenateWithCentroidDiff()
        mask = torch.ones(10, dtype=bool)
        if not self.use_depth:
            mask[2] = False
            mask[7] = False

        def transform(
            price_bbox: torch.Tensor, prod_bbox: torch.Tensor
        ) -> torch.Tensor:
            if price_bbox.ndim == 2:
                return concatenation_op(price_bbox, prod_bbox)[:, mask]
            else:
                return concatenation_op(price_bbox, prod_bbox)[mask]

        return transform

    def _get_collate_fn(self) -> Callable | None:
        if self.prediction_strategy == PredictionStrategy.JOINT:
            return scene_level_collate_fn
        else:
            return None
