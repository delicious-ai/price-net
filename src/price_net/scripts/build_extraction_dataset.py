import argparse
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

PRICE_BOXES_FNAME = 'price_boxes.csv'
ATTRIBUTION_SET_ID_COL = "attributionset_id"
PRICE_BBOX_ID_COL = "price_bbox_id"
SCENE_IMG_DIR = "images"
PRICE_IMG_DIR = "price-images"
IMG_EXT = "jpg"

def crop_with_relative_bbox(image: Image.Image, bbox: tuple[float, float, float, float]) -> Image.Image:
    """
    Crop an image using relative bounding box coordinates.

    Args:
        image (PIL.Image.Image): The input image.
        bbox (tuple): A 4-tuple (min_x, min_y, max_x, max_y), all relative (between 0 and 1).

    Returns:
        PIL.Image.Image: The cropped image.
    """
    min_x, min_y, max_x, max_y = bbox
    width, height = image.size

    left = int(min_x * width)
    upper = int(min_y * height)
    right = int(max_x * width)
    lower = int(max_y * height)

    return image.crop((left, upper, right, lower))


def save_cropped_image(image: Image.Image, output_path: Path) -> None:
    """
    Save the given image to the specified output path in JPEG format.

    Args:
        image (PIL.Image.Image): The image to save.
        output_path (Path): The full path (including filename) to save the image to.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path.with_suffix(".jpg"), format="JPEG")

def main(dataset_path: Path):

    price_boxes_path = dataset_path / PRICE_BOXES_FNAME
    price_boxes = pd.read_csv(price_boxes_path)

    num_images = price_boxes[ATTRIBUTION_SET_ID_COL].unique()
    num_price_tags = price_boxes[PRICE_BBOX_ID_COL].unique()

    print(f"Number of images: {len(num_images)}")
    print(f"Number of price boxes: {len(num_price_tags)}")

    for attribution_id, vals in tqdm(price_boxes.groupby(ATTRIBUTION_SET_ID_COL), total=len(num_images)):
        scene_img = Image.open(dataset_path / SCENE_IMG_DIR / f"{attribution_id}.{IMG_EXT}")

        for _, tag_instance in vals.iterrows():
            price_bbox_id = tag_instance[PRICE_BBOX_ID_COL]
            price_tag_img = crop_with_relative_bbox(
                scene_img,
                (
                    tag_instance['min_x'],
                    tag_instance['min_y'],
                    tag_instance['max_x'],
                    tag_instance['max_y']
                ),
            )

            save_cropped_image(
                price_tag_img,
                dataset_path / PRICE_IMG_DIR / f"{price_bbox_id}.{IMG_EXT}"
            )



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, help="Path to the dataset (train or test)")
    args = parser.parse_args()
    dataset_path = Path(args.dataset_path)
    main(dataset_path)

