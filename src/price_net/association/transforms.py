from typing import Protocol

import torch


class InputTransform(Protocol):
    def __call__(
        self, prod_bboxes: torch.Tensor, price_bboxes: torch.Tensor
    ) -> torch.Tensor: ...


class ConcatenateBoundingBoxes(InputTransform):
    def __call__(
        self, prod_bboxes: torch.Tensor, price_bboxes: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat([prod_bboxes, price_bboxes], dim=1)
