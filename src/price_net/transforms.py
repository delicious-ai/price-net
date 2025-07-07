from typing import Protocol

import torch


class InputTransform(Protocol):
    def __call__(
        self, price_bbox: torch.Tensor, prod_bbox: torch.Tensor
    ) -> torch.Tensor: ...


class ConcatenateBoundingBoxes(InputTransform):
    def __call__(
        self, price_bbox: torch.Tensor, prod_bbox: torch.Tensor
    ) -> torch.Tensor:
        flat = False
        if price_bbox.ndim == 1:
            price_bbox = price_bbox.unsqueeze(0)
            flat = True
        if prod_bbox.ndim == 1:
            prod_bbox = prod_bbox.unsqueeze(0)
            flat = True
        result = torch.cat([price_bbox, prod_bbox], dim=1)
        return result.squeeze(0) if flat else result


class ConcatenateWithCentroidDiff(InputTransform):
    def __init__(self, centroid_dim: int = 3):
        self.centroid_dim = centroid_dim

    def __call__(
        self, price_bbox: torch.Tensor, prod_bbox: torch.Tensor
    ) -> torch.Tensor:
        flat = False
        if price_bbox.ndim == 1:
            price_bbox = price_bbox.unsqueeze(0)
            flat = True
        if prod_bbox.ndim == 1:
            prod_bbox = prod_bbox.unsqueeze(0)
            flat = True

        prod_centroid = prod_bbox[:, : self.centroid_dim]
        price_centroid = price_bbox[:, : self.centroid_dim]
        centroid_diff = prod_centroid - price_centroid
        prod_wh = prod_bbox[:, self.centroid_dim :]
        result = torch.cat([centroid_diff, prod_wh, price_bbox])
        return result.squeeze(0) if flat else result
