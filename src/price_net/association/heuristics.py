import json
import os
import time
from collections import defaultdict
from itertools import product as cross_product
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from google import genai
from google.genai.types import Content
from google.genai.types import GenerateContentConfig
from google.genai.types import Part
from google.genai.types import ThinkingConfig
from price_net.enums import HeuristicType
from price_net.schema import PriceGroup
from price_net.schema import PriceScene
from shapely.geometry import LineString
from shapely.geometry import Point
from shapely.ops import polygonize
from shapely.ops import unary_union


class Heuristic(Protocol):
    def __call__(self, scene: PriceScene) -> list[PriceGroup]: ...


class AssignEverythingToEverything(Heuristic):
    def __call__(self, scene: PriceScene) -> list[PriceGroup]:
        prod_ids = scene.product_bboxes.keys()
        price_ids = scene.price_bboxes.keys()
        groups = [
            PriceGroup(product_bbox_ids={prod_id}, price_bbox_ids={price_id})
            for prod_id, price_id in cross_product(prod_ids, price_ids)
        ]
        return groups


class AssignProductToNearestPrice(Heuristic):
    def __call__(self, scene: PriceScene) -> list[PriceGroup]:
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
    def __call__(self, scene: PriceScene) -> list[PriceGroup]:
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
    def __call__(self, scene: PriceScene) -> list[PriceGroup]:
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
    def __call__(self, scene: PriceScene) -> list[PriceGroup]:
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
            implied_group = PriceGroup(
                product_bbox_ids=group.product_bbox_ids,
                price_bbox_ids={price_ids[idx_of_nearest]},
            )
            pred_groups.append(implied_group)

        return pred_groups


class AssignProductToAllPricesWithinEpsilon(Heuristic):
    def __init__(self, epsilon: float):
        self.epsilon = epsilon

    def __call__(self, scene: PriceScene) -> list[PriceGroup]:
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

    def __call__(self, scene: PriceScene) -> list[PriceGroup]:
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


class Gemini(Heuristic):
    def __init__(self, model: str, price_product_metadata_path: str):
        from dotenv import load_dotenv

        load_dotenv()
        self.model = model
        self.client = genai.Client(
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION"),
        )
        with open("data/metadata/products.json", "r") as f:
            self.product_names: dict[str, str] = json.load(f)
        with open(price_product_metadata_path, "r") as f:
            self.price_tag_metadata = json.load(f)

    def gemini(
        self, pairs: list[dict[str, dict[str, str]]], retry: bool = True
    ) -> list[bool]:
        prompt = (
            open("prompts/associate_products_and_prices_from_extracted_text.txt")
            .read()
            .format(pairs=pairs)
        )
        try:
            raw = self.client.models.generate_content(
                model=self.model,
                contents=[Content(role="user", parts=[Part(text=prompt)])],
                config=GenerateContentConfig(
                    temperature=1,
                    top_p=0.95,
                    response_modalities=["TEXT"],
                    response_mime_type="application/json",
                    thinking_config=(
                        ThinkingConfig(thinking_budget=-1)
                        if "pro" in self.model
                        else None
                    ),
                ),
            )
        except Exception as e:
            print(f"Throttled. Retrying...: {e}")
            if retry:
                time.sleep(30)
                return self.gemini(pairs=pairs, retry=False)
            else:
                raise Exception(f"Failed too many times: {e}") from e
        try:
            predicted_ids = json.loads(raw.text.strip())
        except json.JSONDecodeError:
            print(f"Failed to parse response: `{raw.text.strip()}`")
            return [False] * len(pairs)
        assert isinstance(predicted_ids, list), (
            f"Response is not a list: {type(predicted_ids)}"
        )
        response = [
            str(pair["id"]) in predicted_ids or int(pair["id"]) in predicted_ids
            for pair in pairs
        ]
        return response

    def _prepare_products(self, scene: PriceScene) -> list[dict[str, str]]:
        """
        Gets a unique list of products in the scene, with the upc and name.
        """
        products = {}
        for upc in scene.products.values():
            name = self.product_names.get(upc)
            products[upc] = name
        return [
            {
                "upc": upc,
                "name": name,
            }
            for upc, name in products.items()
        ]

    def _prepare_prices(self, scene: PriceScene) -> list[dict[str, str]]:
        """
        Gets a unique list of prices in the scene, with the price type and contents, and the product metadata.
        """
        prices = {}
        for bbox_id, price in scene.prices.items():
            prices[(price.price_type.value, price.price_text)] = (
                self.price_tag_metadata[bbox_id]
            )
        return [
            {
                "price_type": price_type,
                "price_contents": price_contents,
                "more_info": product_metadata["price_product_metadata"],
            }
            for (price_type, price_contents), product_metadata in prices.items()
        ]

    def _prepare_ids(
        self, scene: PriceScene, pairs: list[dict]
    ) -> list[dict[str, list[str]]]:
        """
        Gets a list of ids for the products and prices in the scene.
        """
        product_reverse_lookup = defaultdict(list)
        for key, value in scene.products.items():
            product_reverse_lookup[value].append(key)

        price_reverse_lookup = defaultdict(list)
        for key, value in scene.prices.items():
            price_reverse_lookup[(value.price_type.value, value.price_text)].append(key)

        ids = []
        for idx, pair in enumerate(pairs):
            product, price = pair["product"], pair["price"]
            pair["id"] = idx
            ids.append(
                {
                    "product_bbox_ids": product_reverse_lookup[product["upc"]],
                    "price_bbox_ids": price_reverse_lookup[
                        (price["price_type"], price["price_contents"])
                    ],
                }
            )
        return ids

    def __call__(self, scene: PriceScene) -> list[PriceGroup]:
        # the model will return a list of
        # { product_bbox_ids: list[str], price_bbox_ids: list[str] }, cast to list[PriceGroup]
        products = self._prepare_products(scene)
        prices = self._prepare_prices(scene)
        pairs = [
            {"product": prod, "price": price}
            for prod, price in cross_product(products, prices)
        ]
        ids = self._prepare_ids(scene, pairs=pairs)
        response: list[bool] = self.gemini(pairs=pairs)
        return [
            PriceGroup(
                product_bbox_ids=ids[index]["product_bbox_ids"],
                price_bbox_ids=ids[index]["price_bbox_ids"],
            )
            for index, is_associated in enumerate(response)
            if is_associated
        ]


HEURISTIC_REGISTRY: dict[HeuristicType, type[Heuristic]] = {
    HeuristicType.EVERYTHING: AssignEverythingToEverything,
    HeuristicType.GEMINI: Gemini,
    HeuristicType.WITHIN_EPSILON: AssignProductToAllPricesWithinEpsilon,
    HeuristicType.NEAREST: AssignProductToNearestPrice,
    HeuristicType.NEAREST_BELOW: AssignProductToNearestPriceBelow,
    HeuristicType.NEAREST_PER_GROUP: AssignProductToNearestPricePerGroup,
    HeuristicType.NEAREST_BELOW_PER_GROUP: AssignProductToNearestPriceBelowPerGroup,
    HeuristicType.HOUGH_REGIONS: AssignProductToNearestPriceInHoughRegions,
}
