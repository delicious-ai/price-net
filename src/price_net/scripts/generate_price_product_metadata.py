import io
import json
import os
import time
from argparse import ArgumentParser
from pathlib import Path
from typing import Callable

import easyocr
import polars as pl
from google import genai
from google.genai.types import Content
from google.genai.types import GenerateContentConfig
from google.genai.types import Part
from google.genai.types import ThinkingConfig
from PIL import Image
from price_net.utils import parse_unknown_args
from tqdm import tqdm


def get_text_from_crop__gemini(
    crop: Image.Image, client: genai.Client, model: str, retry: bool = True
) -> dict[str, str]:
    prompt = open("prompts/extract_product_info_from_tag.txt", "r").read()
    buffer = io.BytesIO()
    crop.save(buffer, format="JPEG")
    image_bytes = buffer.getvalue()
    try:
        raw = client.models.generate_content(
            model=model,
            config=GenerateContentConfig(
                temperature=1,
                top_p=0.95,
                max_output_tokens=8192,
                response_modalities=["TEXT"],
                response_mime_type="application/json",
                thinking_config=ThinkingConfig(thinking_budget=-1)
                if "pro" in model
                else None,
            ),
            contents=[
                Content(
                    role="user",
                    parts=[
                        Part.from_bytes(
                            Part.from_text(text=prompt),
                            data=image_bytes,
                            mime_type="image/jpeg",
                        ),
                    ],
                )
            ],
        )
    except Exception as e:
        print(f"Throttled. Retrying...: {e}")
        if retry:
            time.sleep(30)
            return get_text_from_crop__gemini(crop, client, model, retry=False)
        else:
            raise Exception(f"Failed too many times: {e}") from e
    return json.loads(raw.text.strip())


def get_text_from_crop__ocr(
    crop: Image.Image, reader: easyocr.Reader
) -> dict[str, str]:
    def is_number(s):
        try:
            float(s)  # works for both int and float strings
            return True
        except ValueError:
            return False

    buffer = io.BytesIO()
    crop.save(buffer, format="JPEG")
    text = " ".join(
        [
            resp[1]
            for resp in reader.readtext(image=buffer.getvalue())
            if not is_number(resp[1])
        ]
    )
    return {
        "price_product_metadata": text if len(text.strip()) else None,
    }


def extract_with_(
    boxes: pl.DataFrame,
    cache: dict[str, str],
    process: Callable[[Image.Image], str],
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
            extracted_text = process(price_crop, **kwargs)
            cache[row["price_bbox_id"]] = extracted_text
        with open(cache_path, "w") as f:
            json.dump(cache, f)


if __name__ == "__main__":
    """
    A script to generate price-product metadata that
    is consumed downstream in text-based association methods.
    """
    from dotenv import load_dotenv

    load_dotenv()
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=Path)
    parser.add_argument("--results-file", type=str)
    parser.add_argument("--split", type=str, choices=["val", "test"], default="test")
    parser.add_argument("--method", type=str, choices=["vllm", "ocr"])
    args, unknown_args = parser.parse_known_args()
    kwargs = parse_unknown_args(unknown_args)
    method = args.method
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
    cache_path = Path(os.path.join("data", "metadata", args.results_file))
    if cache_path.exists():
        with open(cache_path, "r") as f:
            cache = json.load(f)
    else:
        cache = {}

    print("cache length: ", len(cache))

    if method == "vllm":
        # open cache
        extract_with_(
            price_boxes,
            cache=cache,
            process=get_text_from_crop__gemini,
            client=genai.Client(
                vertexai=True,
                project=os.getenv("GOOGLE_CLOUD_PROJECT"),
                location=os.getenv("GOOGLE_CLOUD_LOCATION"),
            ),
            model=kwargs["model"],
        )
    elif method == "ocr":
        extract_with_(
            price_boxes,
            cache=cache,
            process=get_text_from_crop__ocr,
            reader=easyocr.Reader(["en"]),
        )
    else:
        raise ValueError(f"Unsupported: '{method}'")
