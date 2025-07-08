from itertools import product
from typing import Protocol

import torch
from price_net.enums import HeuristicType
from price_net.schema import PriceAssociationScene
from price_net.schema import PriceGroup


class Heuristic(Protocol):
    def __call__(self, scene: PriceAssociationScene) -> list[PriceGroup]: ...


class AssignEverythingToEverything(Heuristic):
    def __call__(self, scene: PriceAssociationScene) -> list[PriceGroup]:
        prod_ids = scene.product_bboxes.keys()
        price_ids = scene.price_bboxes.keys()
        groups = [
            PriceGroup(product_bbox_ids={prod_id}, price_bbox_ids={price_id})
            for prod_id, price_id in product(prod_ids, price_ids)
        ]
        return groups


class AssignProductToNearestPrice(Heuristic):
    def __call__(self, scene: PriceAssociationScene) -> list[PriceGroup]:
        prod_ids: list[str] = []
        prod_bboxes = []
        for id_, bbox in scene.product_bboxes.items():
            prod_ids.append(id_)
            prod_bboxes.append(bbox.to_tensor())
        prod_bboxes = torch.stack(prod_bboxes)

        price_ids: list[str] = []
        price_bboxes = []
        for id_, bbox in scene.price_bboxes.items():
            price_ids.append(id_)
            price_bboxes.append(bbox.to_tensor())
        price_bboxes = torch.stack(price_bboxes)

        prod_centroids = prod_bboxes[:, :3]
        price_centroids = price_bboxes[:, :3]

        distances = torch.cdist(prod_centroids, price_centroids, p=2)
        idx_of_nearest: list[int] = torch.argmin(distances, dim=1).tolist()

        groups = []
        for prod_id, nearest_price_idx in zip(prod_ids, idx_of_nearest):
            price_id = price_ids[nearest_price_idx]
            implied_group = PriceGroup(
                product_bbox_ids={prod_id}, price_bbox_ids={price_id}
            )
            groups.append(implied_group)

        return groups


HEURISTIC_REGISTRY: dict[HeuristicType, type[Heuristic]] = {
    HeuristicType.EVERYTHING: AssignEverythingToEverything,
    HeuristicType.NEAREST: AssignProductToNearestPrice,
}
