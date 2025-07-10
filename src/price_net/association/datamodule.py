from functools import partial
from pathlib import Path
from typing import Callable
from typing import Literal

import lightning as L
import torch
from price_net.association.configs import FeaturizationConfig
from price_net.association.dataset import PriceAssociationDataset
from price_net.association.transforms import InputTransform
from price_net.enums import Aggregation
from price_net.enums import PredictionStrategy
from price_net.utils import joint_prediction_collate_fn
from price_net.utils import marginal_prediction_collate_fn
from price_net.utils import split_bboxes
from torch.utils.data import DataLoader


class PriceAssociationDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: Path,
        batch_size: int = 1,
        num_workers: int = 0,
        prediction_strategy: PredictionStrategy = PredictionStrategy.MARGINAL,
        aggregation: Aggregation = Aggregation.NONE,
        featurization_config: FeaturizationConfig = FeaturizationConfig(),
    ):
        """Initialize a `PriceAssociationDataModule`.

        Args:
            data_dir (Path): The directory where the dataset is stored.
            batch_size (int, optional): The batch size to use for dataloaders. Defaults to 1.
            num_workers (int, optional): The number of workers to use for dataloaders. Defaults to 0.
            prediction_strategy (PredictionStrategy, optional): Specifies if predictions will be made marginally (treating each association independently) or jointly (across a scene). Defaults to PredictionStrategy.MARGINAL.
            aggregation (Aggregation, optional): Specifies how to aggregate the raw set of potential product-price associations. Defaults to Aggregation.NONE.
            featurization_config (FeaturizationConfig, optional): Specifies how to build feature representations of possible product-price associations.
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.aggregation = aggregation
        self.prediction_strategy = prediction_strategy
        self.featurization_config = featurization_config
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
            aggregation=self.aggregation,
            use_depth=self.featurization_config.use_depth,
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
        config = self.featurization_config
        _split_bboxes = partial(split_bboxes, use_depth=config.use_depth)

        class _Transform(InputTransform):
            def __call__(
                self, prod_bboxes: torch.Tensor, price_bboxes: torch.Tensor
            ) -> torch.Tensor:
                features = []

                prod_centroids, prod_wh = _split_bboxes(prod_bboxes)
                price_centroids, price_wh = _split_bboxes(price_bboxes)

                if config.use_delta:
                    features.append(prod_centroids - price_centroids)
                elif config.use_prod_centroid:
                    features.append(prod_centroids)

                if config.use_prod_size:
                    features.append(prod_wh)

                if config.use_price_centroid:
                    features.append(price_centroids)

                if config.use_price_size:
                    features.append(price_wh)

                return torch.cat(features, dim=1)

        return _Transform()

    def _get_collate_fn(self) -> Callable | None:
        if self.prediction_strategy == PredictionStrategy.JOINT:
            return joint_prediction_collate_fn
        else:
            return marginal_prediction_collate_fn
