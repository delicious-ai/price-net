from __future__ import annotations

import csv
import json
import os
import sys
from argparse import ArgumentParser
from pathlib import Path

import yaml
from tqdm import tqdm

sys.path.append(str(Path(os.getcwd()).parent))

from price_net.schema import PriceAssociationScene
from price_net.extraction.end_to_end import (
    create_gemini_attribution_extractor,
    create_gpt_attribution_extractor,
)
from price_net.association.configs import EndToEndConfig


def get_products_from_scene(
    scene: PriceAssociationScene, upc_to_name_mapping: dict
) -> list[tuple]:
    """Get products list for the attribution extractor"""
    products = []

    all_upcs = list(scene.products.values())
    unique_upcs = set(all_upcs)
    for upc in unique_upcs:
        product_name = upc_to_name_mapping.get(upc, "unavailable")
        products.append((product_name, upc))

    return products


def save_attributions_to_file(new_attributions, all_attributions, output_path):
    """Save attributions to file, appending to existing data if file exists"""
    # Load existing data if file exists, otherwise start with empty list
    existing_data = []
    if output_path.exists():
        try:
            with open(output_path, "r") as f:
                existing_data = json.load(f)
            existing_data.extend(new_attributions)
        except (json.JSONDecodeError, IOError) as e:
            print(
                f"Warning: Could not read existing file, replacing with all_attributions: {e}"
            )
            existing_data = all_attributions

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the combined data back to file
    with open(output_path, "w") as f:
        json.dump(existing_data, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()

    # Read config to determine model type
    with open(args.config, "r") as f:
        config_data = yaml.safe_load(f)
        config = EndToEndConfig(**config_data)

    model_name = config_data.get("model_name", "").lower()

    # Create appropriate extractor based on model name
    if "gemini" in model_name:
        print(f"Detected Gemini model: {model_name}")
        extractor = create_gemini_attribution_extractor(args.config)
    elif "gpt" in model_name:
        print(f"Detected GPT model: {model_name}")
        extractor = create_gpt_attribution_extractor(args.config)
    else:
        raise ValueError(
            f"Unsupported model type: {model_name}. Supported: gemini*, gpt*"
        )

    # Load the dataset into scenes
    with open(config.dataset_dir / "raw_price_scenes.json", "r") as f:
        scenes = [PriceAssociationScene(**scene) for scene in json.load(f)]

    # Load the upc to product mapping
    upc_to_product_name = dict()
    with open(config.upc_to_name_path, newline="") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            upc, desc = row
            if upc == desc:
                desc = "unavailable"
            upc_to_product_name[upc] = desc

    # Process all scenes
    print(f"=== Processing {len(scenes)} scenes ===")

    all_attributions = []
    new_attributions_since_save = []
    save_interval = 20  # Save every 20 images

    for i in tqdm(range(len(scenes)), desc="Processing scenes", unit="scene"):
        scene = scenes[i]

        # Get products for this scene
        products = get_products_from_scene(scene, upc_to_product_name)

        # Get image path
        image_path = config.dataset_dir / "images" / f"{scene.scene_id}.jpg"

        # Check if image exists
        if not image_path.exists():
            print(f"  ⚠️ Warning: Image not found at {image_path}")
            continue

        # Extract attributions
        try:
            attributions = extractor(image_path, products, scene.scene_id)

            # Convert to dicts and add to collections
            attribution_dicts = [attr.model_dump() for attr in attributions]
            all_attributions.extend(attribution_dicts)
            new_attributions_since_save.extend(attribution_dicts)

            # Save progress every save_interval images
            if (i + 1) % save_interval == 0:
                print(f"  💾 Saving progress after {i + 1} images...")
                save_attributions_to_file(
                    new_attributions_since_save, all_attributions, config.output_path
                )
                print(
                    f"  ✅ Progress saved! New attributions: {len(new_attributions_since_save)}, Total so far: {len(all_attributions)}"
                )
                new_attributions_since_save = []  # Reset the new attributions list

        except Exception as e:
            print(f"  ❌ Error processing scene {scene.scene_id}: {e}")
            continue

    print("\n=== Processing Complete ===")
    print(f"Total attributions extracted: {len(all_attributions)}")

    # Final save for any remaining attributions
    if new_attributions_since_save:
        print(
            f"  💾 Final save of remaining {len(new_attributions_since_save)} attributions..."
        )
        save_attributions_to_file(
            new_attributions_since_save, all_attributions, config.output_path
        )

    print("Final save completed!")
