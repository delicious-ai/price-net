import json
import shutil
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image
from price_net.dataset import PriceAttributionDataset
from price_net.enums import FeaturizationMethod
from price_net.enums import InputGranularity
from price_net.schema import BoundingBox
from price_net.schema import PriceAttributionScene
from price_net.schema import PriceGroup
from price_net.schema import ProductGroup
from price_net.schema import StandardPrice


@pytest.fixture
def scenes():
    return [
        PriceAttributionScene(
            scene_id="1",
            product_bboxes={
                "a": BoundingBox(cx=0.4, cy=0.4, w=0.1, h=0.1),
                "b": BoundingBox(cx=0.6, cy=0.4, w=0.1, h=0.1),
                "c": BoundingBox(cx=0.4, cy=0.8, w=0.04, h=0.1),
            },
            products={
                "a": "Coke",
                "b": "Coke Zero",
                "c": "Pepsi",
            },
            price_bboxes={
                "d": BoundingBox(cx=0.2, cy=0.35, w=0.05, h=0.05),
                "e": BoundingBox(cx=0.2, cy=0.95, w=0.05, h=0.05),
            },
            prices={
                "d": StandardPrice(amount=1.99),
                "e": StandardPrice(amount=2.99),
            },
            product_groups=[
                ProductGroup(
                    group_id="group-1",
                    product_bbox_ids={"a", "b"},
                ),
                ProductGroup(
                    group_id="group-2",
                    product_bbox_ids={"c"},
                ),
            ],
            price_groups=[
                PriceGroup(
                    group_id="price-group-1",
                    product_bbox_ids={"a", "b"},
                    price_bbox_ids={"d"},
                ),
                PriceGroup(
                    group_id="price-group-2",
                    product_bbox_ids={"c"},
                    price_bbox_ids={"e"},
                ),
            ],
        )
    ]


@pytest.fixture
def test_data_dir(tmp_path: Path):
    data_dir = tmp_path / "test-price-attribution-dataset"
    data_dir.mkdir(parents=True, exist_ok=True)
    yield data_dir
    shutil.rmtree(data_dir)


@pytest.fixture
def setup_and_return_root_dir(
    test_data_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
    scenes: list[PriceAttributionScene],
):
    root_dir = test_data_dir
    with monkeypatch.context() as m:
        m.chdir(test_data_dir)

        # Set up data directory.
        images_dir = root_dir / PriceAttributionDataset.IMAGES_DIR
        depth_maps_dir = root_dir / PriceAttributionDataset.DEPTH_MAPS_DIR
        for d in (images_dir, depth_maps_dir):
            d.mkdir(parents=True, exist_ok=True)
        scenes_for_json = [x.model_dump(mode="json") for x in scenes]
        with open(
            root_dir / PriceAttributionDataset.RAW_PRICE_SCENES_FNAME, mode="w"
        ) as f:
            json.dump(scenes_for_json, f)
        fake_img = Image.fromarray(
            (torch.rand(100, 50, 3).numpy() * 255.0).astype(np.uint8),
            mode="RGB",
        )
        scene_id = scenes[0].scene_id
        fake_img.save(
            fp=images_dir / f"{scene_id}.jpg",
            format="JPEG",
        )
        fake_depth = Image.fromarray(
            torch.rand(100, 50).numpy(),
            mode="L",
        )
        fake_depth.save(
            fp=depth_maps_dir / f"{scene_id}.jpg",
            format="JPEG",
        )
    return root_dir


@pytest.mark.parametrize(
    argnames=("input_granularity", "featurization_method"),
    argvalues=product(
        (InputGranularity.PAIRWISE, InputGranularity.SCENE_LEVEL),
        (FeaturizationMethod.CENTROID, FeaturizationMethod.CENTROID_DIFF),
    ),
)
def test_price_attribution_dataset_initializes_correctly(
    setup_and_return_root_dir: Path,
    input_granularity: InputGranularity,
    featurization_method: FeaturizationMethod,
):
    root_dir = setup_and_return_root_dir
    dataset = PriceAttributionDataset(
        root_dir=root_dir,
        input_granularity=input_granularity,
        featurization_method=featurization_method,
    )
    x, y = dataset[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)

    if input_granularity == InputGranularity.SCENE_LEVEL:
        assert x.ndim == 2
        assert x.shape[0] == y.numel()
        assert x.shape[1] == PriceAttributionDataset.FEATURE_DIM
    else:
        assert x.numel() == PriceAttributionDataset.FEATURE_DIM

    if featurization_method == FeaturizationMethod.CENTROID:
        assert torch.all(x >= 0)
