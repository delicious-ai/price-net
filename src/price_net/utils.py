import ast
import os
import random
from typing import Sequence

import lightning as L
import numpy as np
import torch
from matplotlib import patches
from matplotlib import pyplot as plt
from price_net.schema import BoundingBox
from price_net.schema import PriceScene
from seaborn import color_palette
from torch.nn.utils.rnn import pad_sequence


Color = str | Sequence[float]


def parse_bboxes(
    bbox_dict: dict[str, BoundingBox],
) -> tuple[torch.Tensor, list[str], dict[str, int]]:
    """Parse the given bbox dict and return a tensor representation (along with helper objects for indexing).

    Args:
        bbox_dict (dict[str, BoundingBox]): Dict mapping N bbox IDs to their associated xyzwh coordinates.

    Returns:
        - torch.Tensor: A (N, 4 | 5) tensor of bbox coordinates.
        - list[str]: A list of bbox IDs, with a 1:1 correspondence to the bbox tensor.
        - dict[str, int]: A dict mapping each bbox ID to its corresponding row in the bbox tensor.
    """
    bbox_tensor = torch.stack(
        [
            torch.tensor(
                [bbox.cx, bbox.cy, bbox.cz, bbox.w, bbox.h], dtype=torch.float32
            )
            if bbox.cz is not None
            else torch.tensor([bbox.cx, bbox.cy, bbox.w, bbox.h], dtype=torch.float32)
            for bbox in bbox_dict.values()
        ]
    )
    ids = list(bbox_dict.keys())
    id_to_idx = {id_: idx for idx, id_ in enumerate(ids)}
    return bbox_tensor, ids, id_to_idx


PriceAssociationDatasetItem = tuple[torch.Tensor, torch.Tensor, str]


def marginal_prediction_collate_fn(
    batch: list[PriceAssociationDatasetItem],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Organize a batch of potential prod-price associations (grouped by scene ID) into concatenated tensors.

    The batch is assumed to be a list of B tuples, where each tuple contains a variable number of potential product-price associations in a scene, an indicator
    of whether/not they are actual edges, and the scene ID.

    Args:
        batch (list[PriceAssociationDatasetItem]): A list of B tuples, each with an (N_i, D) tensor of associations, an (N_i,) label, and a scene id.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A (Σ N_i, D) input tensor, with corresponding (Σ N_i,) labels.
    """
    Xs, ys, _ = zip(*batch)
    return torch.cat(Xs, dim=0), torch.cat(ys, dim=0)


def joint_prediction_collate_fn(
    batch: list[PriceAssociationDatasetItem],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Organize a (potentially jagged) batch of scene-grouped prod-price edges into a zero-padded tensor with uniform sequence length.

    The batch is assumed to be a list of B tuples, where each tuple contains the potential product-price associations in a scene, an indicator
    of whether/not they are actual edges, and the scene ID.

    Args:
        batch (list[PriceAssociationDatasetItem]): A list of B tuples, each with an (N_i, D) tensor of associations, an (N_i,) label, and a scene id.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A zero-padded (B, N, D) input sequence tensor, with corresponding (B, N) labels and a (B, N) mask indicating if a specific token / label is zero-padded.
    """
    xs, ys, _ = zip(*batch)
    lengths = [x.shape[0] for x in xs]
    xs_padded = pad_sequence(xs, batch_first=True)
    ys_padded = pad_sequence(ys, batch_first=True)
    B, N = xs_padded.shape[:2]
    is_padded_mask = torch.zeros(B, N, dtype=torch.bool)
    for i, length in enumerate(lengths):
        is_padded_mask[i, length:] = True
    return xs_padded, ys_padded, is_padded_mask


def plot_price_scene(
    scene: PriceScene, ax: plt.Axes | None = None
) -> tuple[plt.Axes, dict[str, Color]]:
    """Plot the given price attribution scene as a graph (where nodes are the 2d centroids and edges exist if a product/price are associated).

    Args:
        scene (PriceAttributionScene): The scene to plot.
        ax (plt.Axes | None, optional): If specified, the axes to plot on. Defaults to None (creates one).

    Returns:
        tuple[plt.Axes, dict[str, Color]: The axes on which the scene is plotted, and a dict that specifies which color each product group was assigned.
    """
    if ax is None:
        ax = plt.gca()

    product_centroids = {
        id_: (bbox.cx, bbox.cy) for id_, bbox in scene.product_bboxes.items()
    }
    price_centroids = {
        id_: (bbox.cx, bbox.cy) for id_, bbox in scene.price_bboxes.items()
    }
    prod_id_to_group_idx = {}
    prod_id_to_group_id = {}
    for id_ in scene.product_bboxes.keys():
        for idx, group in enumerate(scene.product_groups):
            if id_ in group.product_bbox_ids:
                prod_id_to_group_idx[id_] = idx
                prod_id_to_group_id[id_] = group.group_id
                break

    colors = color_palette(n_colors=len(scene.product_groups))
    color_key = {}
    plotted = set()
    for price_group in scene.price_groups:
        for price_bbox_id in price_group.price_bbox_ids:
            price_centroid = price_centroids[price_bbox_id]
            ax.scatter(*price_centroid, marker="x", color="black")
            plotted.add(price_bbox_id)
            for prod_bbox_id in price_group.product_bbox_ids:
                prod_centroid = product_centroids[prod_bbox_id]
                color = colors[prod_id_to_group_idx[prod_bbox_id]]
                group_id = prod_id_to_group_id[prod_bbox_id]
                if group_id not in color_key:
                    color_key[group_id] = color
                ax.scatter(*prod_centroid, marker="o", color=color)
                ax.plot(
                    [price_centroid[0], prod_centroid[0]],
                    [price_centroid[1], prod_centroid[1]],
                    c="gray",
                    alpha=0.5,
                    lw=0.5,
                )
                plotted.add(prod_bbox_id)

    unmatched_prices = [
        price_centroids[id_] for id_ in set(price_centroids.keys()).difference(plotted)
    ]
    unmatched_products = [
        product_centroids[id_]
        for id_ in set(product_centroids.keys()).difference(plotted)
    ]
    for centroid in unmatched_prices:
        ax.scatter(*centroid, marker="x", color="black")
    for centroid in unmatched_products:
        ax.scatter(*centroid, marker="o", edgecolors="black", facecolors="none")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 0.0)
    ax.set_aspect("equal")
    ax.set_title(scene.scene_id, fontsize=10)
    return ax, color_key


def plot_bboxes(
    bboxes: list[BoundingBox],
    ax: plt.Axes,
    linestyle: str = "solid",
    color: str | tuple[float, float, float] | None = None,
    width: float = 1.0,
    height: float = 1.0,
):
    """Plot the given list of bounding boxes on the provided axes.

    Args:
        bboxes (torch.Tensor): The bounding boxes to plot.
        ax (plt.Axes): The axes to plot the boxes on.
        linestyle (str, optional): Linestyle for the bounding box (passed to matplotlib). Defaults to "solid".
        color (str | tuple[float, float, float] | None, optional): Edge color for the bounding box. Defaults to None.
        width (float): The desired width of the plot (bboxes will be rescaled from relative to abs. coordinates).
        height (float): The desired height of the plot (bboxes will be rescaled from relative to abs. coordinates).
    """
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")

    for bbox in bboxes:
        x, y, w, h = bbox.cx, bbox.cy, bbox.w, bbox.h
        x = x * width
        y = y * height
        w = w * width
        h = h * height
        rect = patches.Rectangle(
            xy=(x - w / 2, y - h / 2),
            width=w,
            height=h,
            linewidth=1.5,
            edgecolor=color or "k",
            facecolor="none",
            linestyle=linestyle,
        )
        ax.add_patch(rect)


def split_bboxes(
    bboxes: torch.Tensor,
    use_depth: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split the provided bboxes into their centroid and size components.

    Args:
        bboxes (torch.Tensor): A (N, 4 | 5) tensor of xyzwh bounding boxes.
        use_depth (bool, optional): Whether/not to return a centroid with a z (depth) dimension. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: A (N, 2 | 3) tensor of bbox centroids, and a (N, 2) tensor of bbox sizes.
    """
    if use_depth and bboxes.shape[1] != 5:
        raise ValueError("Expected bboxes to have 5 columns when use_depth is True.")
    centroid_end_idx = 3 if use_depth else 2
    centroids = bboxes[:, :centroid_end_idx]
    sizes = bboxes[:, centroid_end_idx : centroid_end_idx + 2]
    return centroids, sizes


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    L.seed_everything(seed, workers=True)


def parse_unknown_args(unknown_args: list[str]):
    it = iter(unknown_args)
    kwargs = {}
    for key in it:
        if not key.startswith("--"):
            raise ValueError(f"Unexpected argument format: {key}")
        key = key[2:].replace("-", "_")
        value = next(it)
        try:
            value = ast.literal_eval(value)
        except Exception:
            pass
        kwargs[key] = value
    return kwargs


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
