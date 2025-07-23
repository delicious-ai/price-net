import argparse
import os
import random
import json
from pathlib import Path
from enum import Enum
from tqdm import tqdm

import pandas as pd
from google.cloud import storage

from price_net.extraction.parsers import *
from price_net.enums import PriceType


random.seed(1998)

PRICE_IMAGE_DIR = 'price_images'
PRICE_BOX_CSV = 'price_boxes.csv'
PRICE_CONTENTS_COL = "price_contents"
PRICE_TYPE_TYPE_COL = "price_type"
BBOX_ID_COL = "price_bbox_id"
GCS_BUCKET = 'dai-ultra-datasets'
GCS_DIR = 'pricing/kdd26'
IMG_EXT = 'jpg'
FILENAME = "dataset.jsonl"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=str, help="path to dataset (train or val)")
    parser.add_argument("--prompt-path", type=str, help="path to prompt", default="price_net/extraction/prompts/few-shot.txt")
    parser.add_argument("--upload-images", action="store_true", default=False, help="whether to upload images to GCS")
    parser.add_argument("--val-prob", type=float, default=0.2)
    return parser.parse_args()


def upload_to_gcs(local_file_path: str, bucket_name: str, destination_blob_path: str) -> None:
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_path)

    if blob.exists():
        return

    blob.upload_from_filename(local_file_path)


def write_and_upload_jsonl(contents: list, local_path: Path, bucket_name: str, blob_path: str) -> None:
    with local_path.open("w", encoding="utf-8") as outfile:
        for entry in contents:
            json.dump(entry, outfile)
            outfile.write("\n")
    upload_to_gcs(str(local_path), bucket_name, blob_path)


def get_price_type_enum(price_type: str) -> PriceType:
    if price_type == "BUY_X_GET_Y":
        return PriceType.BUY_X_GET_Y_FOR_Z
    elif price_type == "SINGLE_SALE":
        return PriceType.STANDARD
    else:
        return PriceType(price_type)

def get_price_json(df_row: pd.Series) -> dict:
    price_type = get_price_type_enum(df_row[PRICE_TYPE_TYPE_COL])
    contents = df_row[PRICE_CONTENTS_COL]
    output = {'price_type': price_type.value}

    if price_type == PriceType.UNKNOWN:
        pass

    elif price_type == PriceType.STANDARD:
        price, price_string = parse_regular_price(contents)
        output["amount"] = price[0]
    elif price_type == PriceType.BULK_OFFER:
        price, price_string = parse_bulk_offer_price(contents)
        output["quantity"] = price[0]
        output["total_price"] = price[1]
    elif price_type == PriceType.BUY_X_GET_Y_FOR_Z:
        price, price_string =  parse_buy_x_get_y_price(contents)
        output["buy_quantity"] = price[0]
        output["get_quantity"] = price[1]
        output["get_price"] = price[2]
    else:
        raise RuntimeError(f"Unrecognized row type: {df_row.type}")
    return output

def main(dataset_dir: Path, prompt_path: Path, upload_images: bool, val_prob: float):

    df = pd.read_csv(str(dataset_dir / PRICE_BOX_CSV))
    df = df[~(pd.isnull(df[PRICE_CONTENTS_COL]) & pd.isnull(df[PRICE_TYPE_TYPE_COL]))]
    df = df[df.price_type != "MISC"]



    with prompt_path.open("r", encoding="utf-8") as f:
        prompt =  f.read()


    trn_contents = []
    val_contents = []
    for index, row in tqdm(df.iterrows(), total=len(df)):


        image_id = row[BBOX_ID_COL]
        img_bucket_path = f"{GCS_DIR}/price-images/{image_id}.{IMG_EXT}"
        price_dict = get_price_json(row)
        local_img_path = dataset_dir / "price-images" / f"{image_id}.{IMG_EXT}"

        if upload_images:
            upload_to_gcs(local_img_path, GCS_BUCKET, img_bucket_path)

        item = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "fileData": {
                                "mimeType": "image/jpeg",
                                "fileUri": f"gs://{GCS_BUCKET}/{img_bucket_path}",
                            }
                        },
                        {"text": prompt},
                    ],
                },
                {
                    "role": "model",
                    "parts": [
                        {"text": f"""```json\n{json.dumps(price_dict)}\n```"""}
                    ],
                },
            ]
        }
        alpha = np.random.random()
        if alpha < val_prob:
            val_contents.append(item)
        else:
            trn_contents.append(item)

    print(f"Number of validation examples: {len(val_contents)}/{df.shape[0]}")
    print(f"Number of training examples: {len(trn_contents)}/{df.shape[0]}")

    write_and_upload_jsonl(val_contents, Path(FILENAME), GCS_BUCKET, f"{GCS_DIR}/val/{FILENAME}")
    if val_contents:
        write_and_upload_jsonl(val_contents, Path(FILENAME), GCS_BUCKET, f"{GCS_DIR}/train/{FILENAME}")


if __name__ == "__main__":
    args = parse_args()
    main(
        dataset_dir=Path(args.dataset_dir),
        prompt_path=Path(args.prompt_path),
        upload_images=args.upload_images,
        val_prob=args.val_prob,
    )
