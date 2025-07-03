import torch
from matplotlib import pyplot as plt
from price_net.schema import BoundingBox
from price_net.schema import PriceAttributionScene
from seaborn import color_palette
from torch.nn.utils.rnn import pad_sequence


def xywh_to_xyzwh(xywh: torch.Tensor, depth_map: torch.Tensor) -> torch.Tensor:
    """Convert the provided xywh bboxes to xyzwh, given a depth map for the image the bboxes came from.

    Args:
        xywh (torch.Tensor): The bboxes to convert, with shape (N, 4).
        depth_map (torch.Tensor): A depth map for the image the bboxes came from, with shape (H, W).

    Returns:
        torch.Tensor: The resultant xyzwh bboxes, with shape (N, 5).
    """
    H, W = depth_map.shape
    N = xywh.shape[0]

    x = xywh[:, 0] * W
    y = xywh[:, 1] * H
    w = xywh[:, 2] * W
    h = xywh[:, 3] * H

    x1 = (x - w).clamp(0, W - 1)
    y1 = (y - h).clamp(0, H - 1)
    x2 = (x + w).clamp(0, W - 1)
    y2 = (y + h).clamp(0, H - 1)

    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    xx = xx.unsqueeze(0).expand(N, -1, -1)
    yy = yy.unsqueeze(0).expand(N, -1, -1)

    x1 = x1[:, None, None]
    x2 = x2[:, None, None]
    y1 = y1[:, None, None]
    y2 = y2[:, None, None]

    mask = (xx >= x1) & (xx <= x2) & (yy >= y1) & (yy <= y2)
    depth = depth_map.unsqueeze(0).expand(N, -1, -1)

    masked_depth = depth * mask
    sum_depth = masked_depth.sum(dim=(1, 2))
    count = mask.sum(dim=(1, 2)).clamp(min=1)
    z = sum_depth / count

    xyzwh = torch.cat([xywh[:, :2], z.view(-1, 1), xywh[:, 2:]], dim=1)
    return xyzwh


def parse_bboxes(
    bbox_dict: dict[str, BoundingBox],
) -> tuple[torch.Tensor, list[str], dict[str, int]]:
    """Parse the given bbox dict and return a tensor representation (along with helper objects for indexing).

    Args:
        bbox_dict (dict[str, BoundingBox]): Dict mapping N bbox IDs to their associated xywh coordinates.

    Returns:
        - torch.Tensor: A (N, 4) tensor of bbox coordinates.
        - list[str]: A list of bbox IDs, with a 1:1 correspondence to the bbox tensor.
        - dict[str, int]: A dict mapping each bbox ID to its corresponding row in the bbox tensor.
    """
    bbox_tensor = torch.stack(
        [
            torch.tensor([bbox.cx, bbox.cy, bbox.w, bbox.h], dtype=torch.float32)
            for bbox in bbox_dict.values()
        ]
    )
    ids = list(bbox_dict.keys())
    id_to_idx = {id_: idx for idx, id_ in enumerate(ids)}
    return bbox_tensor, ids, id_to_idx


def scene_level_collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Organize a (potentially jagged) batch of scene-grouped prod-price edges into a zero-padded tensor with uniform sequence length.

    The batch is assumed to be a list of B tuples, where each tuple contains the potential product-price edges in a scene and an indicator
    of whether/not they are actual edges.

    Args:
        batch (list[tuple[torch.Tensor, torch.Tensor]]): A list of B tuples, each with an (N_i, D) input sequence and an (N_i,) label.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A zero-padded (B, N, D) input sequence tensor, with corresponding (B, N) labels and a (B, N) mask indicating if a specific token / label is zero-padded.
    """
    xs, ys = zip(*batch)
    lengths = [x.shape[0] for x in xs]
    xs_padded = pad_sequence(xs, batch_first=True)
    ys_padded = pad_sequence(ys, batch_first=True)
    B, N = xs_padded.shape[:2]
    is_padded_mask = torch.zeros(B, N, dtype=torch.bool)
    for i, length in enumerate(lengths):
        is_padded_mask[i, length:] = True
    return xs_padded, ys_padded, is_padded_mask


def plot_price_attribution_scene(
    scene: PriceAttributionScene, ax: plt.Axes | None = None
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()

    product_centroids = {
        id_: (bbox.cx, bbox.cy) for id_, bbox in scene.product_bboxes.items()
    }
    price_centroids = {
        id_: (bbox.cx, bbox.cy) for id_, bbox in scene.price_bboxes.items()
    }
    prod_id_to_group_idx = {}
    for id_ in scene.product_bboxes.keys():
        for idx, group in enumerate(scene.product_groups):
            if id_ in group.product_bbox_ids:
                prod_id_to_group_idx[id_] = idx
                break

    colors = color_palette(n_colors=len(scene.product_groups))
    for price_group in scene.price_groups:
        for price_bbox_id in price_group.price_bbox_ids:
            price_centroid = price_centroids[price_bbox_id]
            ax.scatter(*price_centroid, marker="x", color="black")
            for prod_bbox_id in price_group.product_bbox_ids:
                prod_centroid = product_centroids[prod_bbox_id]
                prod_group_idx = prod_id_to_group_idx[prod_bbox_id]
                ax.scatter(*prod_centroid, marker="o", color=colors[prod_group_idx])
                ax.plot(
                    [price_centroid[0], prod_centroid[0]],
                    [price_centroid[1], prod_centroid[1]],
                    c="gray",
                    alpha=0.5,
                    lw=0.5,
                )

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(1.0, 0.0)
    ax.set_aspect("equal")
    ax.set_title(scene.scene_id, fontsize=10)
    return ax
