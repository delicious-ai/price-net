from itertools import product
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from price_net.enums import HeuristicType
from price_net.schema import PriceAssociationScene
from price_net.schema import PriceGroup
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.ops import polygonize
from shapely.ops import unary_union


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


class AssignProductToNearestPriceInHoughRegions(Heuristic):
    def __init__(self, hough_lines_dir: str | Path):
        """
        Heuristic that uses Hough line segmentation to create regions and assigns
        products to nearest prices within each region.

        Args:
            hough_lines_dir: Directory containing .npy files with Hough line data. Must contain files named as <scene_id>.npy, with normalized coordinates and format [y1, x1, y2, x2].
        """
        self.hough_lines_dir = Path(hough_lines_dir)

    def __call__(self, scene: PriceAssociationScene) -> list[PriceGroup]:
        # Get scene data
        prod_ids: list[str] = []
        prod_bboxes = []
        for id_, bbox in scene.product_bboxes.items():
            prod_ids.append(id_)
            prod_bboxes.append(bbox.to_tensor())

        price_ids: list[str] = []
        price_bboxes = []
        for id_, bbox in scene.price_bboxes.items():
            price_ids.append(id_)
            price_bboxes.append(bbox.to_tensor())

        # Handle empty cases
        if len(prod_bboxes) == 0 or len(price_bboxes) == 0:
            return []

        prod_bboxes = torch.stack(prod_bboxes)
        price_bboxes = torch.stack(price_bboxes)

        # Normalized coordinates
        prod_centroids = prod_bboxes[:, :2]
        price_centroids = price_bboxes[:, :2]

        # Load Hough lines
        hl_path = self.hough_lines_dir / f"{scene.scene_id}.npy"
        if not hl_path.exists():
            # Fallback: if no Hough lines, assign to nearest price globally
            return self._fallback_nearest_assignment(
                prod_ids, price_ids, prod_centroids, price_centroids
            )

        shelf_array = np.load(hl_path)
        # Create normalized line strings from Hough lines + image boundaries
        line_strings = [
            LineString(
                [
                    (shelf_array[i, 1], shelf_array[i, 0]),
                    (shelf_array[i, 3], shelf_array[i, 2]),
                ]
            )
            for i in range(len(shelf_array))
        ]

        # Create regions using polygonize
        # Note: polygonize requires a union of lines to handle intersections properly
        union_result = unary_union(line_strings)
        if hasattr(union_result, "geoms"):
            regions = list(polygonize(union_result.geoms))
        else:
            regions = list(polygonize([union_result]))

        # Find associations within each region
        groups = []
        for region in regions:
            # Find products in this region
            region_products = []
            for idx in range(len(prod_ids)):
                point = Point(prod_centroids[idx])
                if region.contains(point):
                    region_products.append(idx)

            # Find prices in this region
            region_prices = []
            for idx in range(len(price_ids)):
                point = Point(price_centroids[idx])
                if region.contains(point):
                    region_prices.append(idx)

            # Assign each product to nearest price within region
            for prod_idx in region_products:
                if len(region_prices) == 0:
                    continue

                prod_coords = prod_centroids[prod_idx]
                price_coords = price_centroids[region_prices]

                # Compute distances in normalized space
                dists = torch.norm(price_coords - prod_coords, dim=1)

                # Find nearest price
                nearest_idx = torch.argmin(dists).item()
                nearest_price_idx = region_prices[nearest_idx]

                # Create price group
                group = PriceGroup(
                    product_bbox_ids={prod_ids[prod_idx]},
                    price_bbox_ids={price_ids[nearest_price_idx]},
                )
                groups.append(group)

        return groups

    def _fallback_nearest_assignment(
        self, prod_ids, price_ids, prod_centroids, price_centroids
    ):
        """Fallback when no Hough lines are available - assign to globally nearest price"""
        groups = []
        for i, prod_id in enumerate(prod_ids):
            if len(price_ids) == 0:
                continue

            prod_coords = prod_centroids[i]
            dists = torch.norm(price_centroids - prod_coords, dim=1)
            nearest_price_idx = torch.argmin(dists).item()

            group = PriceGroup(
                product_bbox_ids={prod_id},
                price_bbox_ids={price_ids[nearest_price_idx]},
            )
            groups.append(group)

        return groups


HEURISTIC_REGISTRY: dict[HeuristicType, type[Heuristic]] = {
    HeuristicType.EVERYTHING: AssignEverythingToEverything,
    HeuristicType.WITHIN_EPSILON: AssignProductToAllPricesWithinEpsilon,
    HeuristicType.NEAREST: AssignProductToNearestPrice,
    HeuristicType.NEAREST_BELOW: AssignProductToNearestPriceBelow,
    HeuristicType.NEAREST_PER_GROUP: AssignProductToNearestPricePerGroup,
    HeuristicType.NEAREST_BELOW_PER_GROUP: AssignProductToNearestPriceBelowPerGroup,
    HeuristicType.HOUGH_REGIONS: AssignProductToNearestPriceInHoughRegions,
}
