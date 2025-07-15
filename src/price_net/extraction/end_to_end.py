import json
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Union

from extractors import GeminiExtractor as BaseGeminiExtractor
from price_net.schema import PriceAttribution
from price_net.schema import PriceBuilder


class AttributionExtractor(BaseGeminiExtractor):
    """Extractor for end-to-end price attribution using Gemini"""

    def __init__(self, model_name: str, client, prompt: str, temperature: float = 0.1):
        super().__init__(model_name, client, prompt, temperature)

    def __call__(
        self, img_input: Union[str, Path, bytes], products: List[tuple], scene_id: str
    ) -> List[PriceAttribution]:
        """
        Extract price attributions for products in an image.

        Args:
            img_input: Image file path, Path object, or bytes
            products: List of (upc, product_name) tuples
            scene_id: Scene identifier

        Returns:
            List of PriceAttribution objects
        """
        # Build the full prompt with products
        products_text = f"\nProducts to analyze:\n{products}\n"
        full_prompt = self.prompt + products_text

        # Route the image input
        img_part = self._route_input(img_input)

        # Update the prompt for this call
        original_prompt = self.prompt
        self.prompt = full_prompt

        try:
            # Make API call
            raw_response = self._api_call(img_part)
            response_data = json.loads(raw_response.text)

            # Convert to PriceAttribution objects
            attributions = []
            for item in response_data:
                try:
                    price = PriceBuilder(price=item["price"]).price
                    attribution = PriceAttribution(
                        scene_id=scene_id, upc=item["upc"], price=price
                    )
                    attributions.append(attribution)
                except Exception as e:
                    print(
                        f"Error parsing attribution for UPC {item.get('upc', 'unknown')}: {e}"
                    )
                    continue

            return attributions

        finally:
            # Restore original prompt
            self.prompt = original_prompt

    def format(self, price_json: dict) -> List[Dict[str, Any]]:
        """Format for evaluation - converts to list of dicts for JSON serialization"""
        return [attr.model_dump() for attr in price_json]


# Convenience function to create extractor from config
def create_attribution_extractor(config_path: Union[str, Path]) -> AttributionExtractor:
    """Create AttributionExtractor from YAML config"""
    cfg = AttributionExtractor.read_yaml(config_path)
    prompt = AttributionExtractor.read_txt(cfg["prompt_fpath"])
    client = AttributionExtractor.get_genai_client()

    return AttributionExtractor(
        model_name=cfg["model_name"],
        client=client,
        prompt=prompt,
        temperature=cfg.get("temperature", 0.1),
    )
