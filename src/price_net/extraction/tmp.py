from argparse import ArgumentParser
import io
import json
import os
from pathlib import Path
import time
from typing import Callable
import polars as pl
from pydantic import ValidationError
from tqdm import tqdm


from PIL import Image

from price_net.schema import PriceBuilder


def get_price_from_crop(
        crop: Image.Image, client: genai.Client, model: str, retry: bool = True
) -> dict[str, str]:
    with open("prompts/extract_price.txt", "r") as f:
        prompt = f.read()
    buffer = io.BytesIO()
    crop.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    try:
        raw_response = client.models.generate_content(
            model=model,
            config=GenerateContentConfig(
                temperature=1,
                max_output_tokens=8192,
                response_modalities=["TEXT"],
                response_mime_type="application/json",
            ),
            contents=[
                Content(
                    role="user",
                    parts=[
                        Part.from_bytes(
                            data=image_bytes,
                            mime_type="image/jpeg",
                        ),
                        Part.from_text(text=prompt),
                    ],
                )
            ],
        )
    except Exception as e:
        print(f"Throttled. Retrying...: {e}")
        if retry:
            time.sleep(30)
            return get_price_from_crop(crop, client, model, retry=False)
        else:
            raise Exception(f"Failed too many times: {e}") from e

    try:
        output_data = json.loads(raw_response.text.replace("'", '"'))
        output = PriceBuilder(price=output_data)
        return output.price.model_dump(mode="json")
    except json.JSONDecodeError as e:
        message = f"Failed to decode JSON response from price extractor: {e}. Raw response: '{raw_response.text}'"
        raise Exception(message)
    except ValidationError as e:
        message = f"Encountered schema error while parsing price extractor output: {e}. Raw response: '{raw_response.text}'"
        raise Exception(message)


def extract_with_(
        boxes: pl.DataFrame,
        cache: dict[str, str],
        process: Callable[[Image.Image], str],
        cache_path: Path,
        **kwargs,
):
    num_images = len(boxes["local_path"].unique())
    for (image_path,), group in tqdm(boxes.group_by("local_path"), total=num_images):
        image = Image.open(image_path)
        for row in group.iter_rows(named=True):
            if row["price_bbox_id"] in cache:
                continue
            coords = (
                row["min_x"] * image.width,
                row["min_y"] * image.height,
                row["max_x"] * image.width,
                row["max_y"] * image.height,
            )
            price_crop = image.crop(coords)
            bbox_id = row["price_bbox_id"]
            try:
                extracted_text = process(price_crop, **kwargs)
                cache[bbox_id] = extracted_text
            except Exception:
                print(f"Failed to extract price for {bbox_id}")
        with open(cache_path, "w") as f:
            json.dump(cache, f)


if __name__ == "__main__":
    from dotenv import load_dotenv

    load_dotenv()
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    parser.add_argument("--model", type=str, default="gemini-2.0-flash")
    args = parser.parse_args()
    dataset_dir = Path(f"{args.dataset}/{args.split}")
    price_boxes = pl.read_csv(dataset_dir / "price_boxes.csv")
    product_boxes = pl.read_csv(dataset_dir / "product_boxes.csv").select(
        ["attributionset_id", "local_path"]
    )
    price_boxes = price_boxes.join(product_boxes, on="attributionset_id").unique()
    price_boxes = price_boxes.with_columns(
        local_path=pl.col("attributionset_id").map_elements(
            lambda x: os.path.join(dataset_dir, "images", f"{x}.jpg"), return_dtype=str
        )
    )
    cache_path = dataset_dir / "extracted_prices.json"
    if cache_path.exists():
        with open(cache_path, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    print("cache length: ", len(cache))

    extract_with_(
        boxes=price_boxes,
        cache=cache,
        process=get_price_from_crop,
        client=genai.Client(
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION"),
        ),
        model=args.model,
        cache_path=cache_path,
    )