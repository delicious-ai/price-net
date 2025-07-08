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


class AssignProductToNearestPricePerGroup(Heuristic):
    def __call__(self, scene: PriceAssociationScene) -> list[PriceGroup]:
        price_ids: list[str] = []
        price_bboxes = []
        for id_, bbox in scene.price_bboxes.items():
            price_ids.append(id_)
            price_bboxes.append(bbox.to_tensor())
        price_bboxes = torch.stack(price_bboxes)

        pred_groups = []
        for group in scene.product_groups:
            group_bboxes = torch.stack(
                [
                    scene.product_bboxes[prod_id].to_tensor()
                    for prod_id in group.product_bbox_ids
                ]
            )
            distances = torch.cdist(group_bboxes[:, :3], price_bboxes[:, :3], p=2)
            idx_of_nearest = torch.argmin(distances).item() % distances.shape[1]
            implied_group = PriceGroup(
                product_bbox_ids=group.product_bbox_ids,
                price_bbox_ids={price_ids[idx_of_nearest]},
            )
            pred_groups.append(implied_group)

        return pred_groups


class AssignProductToNearestPriceBelow(Heuristic):
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
        product_y = prod_centroids[:, 1].unsqueeze(1)
        price_y = price_centroids[:, 1].unsqueeze(0)
        # Recall: with bbox coordinates, the top of an image is y=0.
        under_mask = price_y > product_y
        distances[~under_mask] = float("inf")

        idx_of_nearest = torch.argmin(distances, dim=1).tolist()

        groups = []
        for i, (prod_id, nearest_price_idx) in enumerate(zip(prod_ids, idx_of_nearest)):
            if not distances[i, nearest_price_idx].isfinite().item():
                continue
            price_id = price_ids[nearest_price_idx]
            implied_group = PriceGroup(
                product_bbox_ids={prod_id}, price_bbox_ids={price_id}
            )
            groups.append(implied_group)

        return groups


class AssignProductToNearestPriceBelowPerGroup(Heuristic):
    def __call__(self, scene: PriceAssociationScene) -> list[PriceGroup]:
        price_ids: list[str] = []
        price_bboxes = []
        for id_, bbox in scene.price_bboxes.items():
            price_ids.append(id_)
            price_bboxes.append(bbox.to_tensor())
        price_bboxes = torch.stack(price_bboxes)

        pred_groups = []
        for group in scene.product_groups:
            group_bboxes = torch.stack(
                [
                    scene.product_bboxes[prod_id].to_tensor()
                    for prod_id in group.product_bbox_ids
                ]
            )
            group_centroids = group_bboxes[:, :3]
            price_centroids = price_bboxes[:, :3]
            distances = torch.cdist(group_centroids, price_centroids, p=2)
            group_y = group_centroids[:, 1].unsqueeze(1)
            price_y = price_centroids[:, 1].unsqueeze(0)
            # Recall: with bbox coordinates, the top of an image is y=0.
            under_mask = price_y > group_y
            distances[~under_mask] = float("inf")
            if not distances.isfinite().any():
                continue
            idx_of_nearest = torch.argmin(distances).item() % distances.shape[1]
            if idx_of_nearest >= len(price_ids):
                breakpoint()
            implied_group = PriceGroup(
                product_bbox_ids=group.product_bbox_ids,
                price_bbox_ids={price_ids[idx_of_nearest]},
            )
            pred_groups.append(implied_group)

        return pred_groups


class AssignProductToAllPricesWithinEpsilon(Heuristic):
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

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
        pairs_within_eps = torch.nonzero(distances < self.epsilon, as_tuple=False)
        groups = []
        for i, j in pairs_within_eps:
            implied_group = PriceGroup(
                product_bbox_ids={prod_ids[i]}, price_bbox_ids={price_ids[j]}
            )
            groups.append(implied_group)
        return groups


HEURISTIC_REGISTRY: dict[HeuristicType, type[Heuristic]] = {
    HeuristicType.EVERYTHING: AssignEverythingToEverything,
    HeuristicType.WITHIN_EPSILON: AssignProductToAllPricesWithinEpsilon,
    HeuristicType.NEAREST: AssignProductToNearestPrice,
    HeuristicType.NEAREST_BELOW: AssignProductToNearestPriceBelow,
    HeuristicType.NEAREST_PER_GROUP: AssignProductToNearestPricePerGroup,
    HeuristicType.NEAREST_BELOW_PER_GROUP: AssignProductToNearestPriceBelowPerGroup,
}
