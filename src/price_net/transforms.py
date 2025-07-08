from typing import Protocol

import torch


class InputTransform(Protocol):
    def __call__(
        self, price_bboxes: torch.Tensor, prod_bboxes: torch.Tensor
    ) -> torch.Tensor: ...


class ConcatenateBoundingBoxes(InputTransform):
    def __call__(
        self, price_bboxes: torch.Tensor, prod_bboxes: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat([price_bboxes, prod_bboxes], dim=1)


class ConcatenateWithCentroidDiff(InputTransform):
    def __init__(self, centroid_dim: int = 3):
        self.centroid_dim = centroid_dim

    def __call__(
        self, price_bbox: torch.Tensor, prod_bbox: torch.Tensor
    ) -> torch.Tensor:
        prod_centroid = prod_bbox[:, : self.centroid_dim]
        price_centroid = price_bbox[:, : self.centroid_dim]
        centroid_diff = prod_centroid - price_centroid
        prod_wh = prod_bbox[:, self.centroid_dim :]
        result = torch.cat([centroid_diff, prod_wh, price_bbox], dim=1)
        return result
