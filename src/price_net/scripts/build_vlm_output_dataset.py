from __future__ import annotations

from argparse import ArgumentParser
import os
import sys
from pathlib import Path
import json
import csv
import yaml
from tqdm import tqdm
sys.path.append(str(Path(os.getcwd()).parent))

from price_net.schema import PriceAssociationScene
from price_net.extraction.end_to_end import create_attribution_extractor
from price_net.association.configs import EndToEndConfig




def get_products_from_scene(scene: PriceAssociationScene, upc_to_name_mapping: dict) -> list[tuple]:
        """Get products list for the attribution extractor"""
        products = []
        
        all_upcs = list(scene.products.values())
        unique_upcs = set(all_upcs)
        for upc in unique_upcs:
            product_name = upc_to_name_mapping.get(upc, "unavailable")
            products.append((upc, product_name))
            
        return products


def save_attributions_to_file(new_attributions, output_path):
    """Save attributions to file, appending to existing data if file exists"""
    # Load existing data if file exists, otherwise start with empty list
    existing_data = []
    if output_path.exists():
        try:
            with open(output_path, "r") as f:
                existing_data = json.load(f)
                # Ensure existing_data is a list
                if not isinstance(existing_data, list):
                    existing_data = []
                    print("Warning: Existing file didn't contain a list, starting fresh")
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not read existing file, starting fresh: {e}")
            existing_data = []

    # Extend the existing list with new attributions
    existing_data.extend(new_attributions)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write the combined data back to file
    with open(output_path, "w") as f:
        json.dump(existing_data, f, indent=2)

def main():
    parser = ArgumentParser()
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = EndToEndConfig(**yaml.safe_load(f))

    # Load the dataset into scenes
    with open(config.dataset_dir / "raw_price_scenes.json", "r") as f:
        scenes = [PriceAssociationScene(**scene) for scene in json.load(f)]

    # Load the upc to product mapping
    upc_to_product_name = dict()
    with open(config.upc_to_name_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            upc, desc = row
            if upc == desc:
                desc = "unavailable"
            upc_to_product_name[upc] = desc

    # Process all scenes
    print(f"=== Processing {len(scenes)} scenes ===")

    # Create extractor
    extractor = create_attribution_extractor(args.config)

    all_attributions = []
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
            
            # Add to our collection
            all_attributions.extend([attr.model_dump() for attr in attributions])
            
            # Save progress every save_interval images
            if (i + 1) % save_interval == 0:
                print(f"  💾 Saving progress after {i + 1} images...")
                save_attributions_to_file(all_attributions, config.output_path)
                print(f"  ✅ Progress saved! Total attributions so far: {len(all_attributions)}")
            
        except Exception as e:
            print(f"  ❌ Error processing scene {scene.scene_id}: {e}")
            continue

    print(f"\n=== Processing Complete ===")
    print(f"Total attributions extracted: {len(all_attributions)}")

    # Final save
    save_attributions_to_file(all_attributions, config.output_path)
    print(f"Final save completed!")





if __name__ == "__main__":
    main()