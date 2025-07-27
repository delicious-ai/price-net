from pathlib import Path
from typing import List
from typing import Union

from price_net.extraction.extractors import GeminiExtractor
from price_net.extraction.extractors import GPTExtractor
from price_net.schema import PriceAttribution
from price_net.schema import PriceBuilder


class GeminiAttributionExtractor(GeminiExtractor):
    """Extractor for end-to-end price attribution using Gemini"""

    def __init__(self, model_name: str, client, prompt: str, temperature: float = 0.0):
        super().__init__(model_name, client, prompt, temperature)

    def __call__(
        self, img_input: Union[str, Path, bytes], scene_id: str
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

        try:
            # Make API call using parent class __call__ method
            response_data = super().__call__(img_input)

            if not isinstance(response_data, list):
                print(f"  ⚠️ Unexpected response format: {type(response_data)}")
                response_data = []

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

        except Exception as e:
            raise e


class GPTAttributionExtractor(GPTExtractor):
    """Extractor for end-to-end price attribution using OpenAI GPT"""

    def __init__(self, model_name: str, client, prompt: str, temperature: float = 0.0):
        super().__init__(model_name, client, prompt, temperature)

    def __call__(
        self, img_input: Union[str, Path, bytes], scene_id: str
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

        try:
            # Make simple API call with prompt and image using parent class __call__
            response_data = super().__call__(img_input)

            if not isinstance(response_data, list):
                print(f"  ⚠️ Unexpected response format: {type(response_data)}")
                response_data = []

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

        except Exception as e:
            raise e


def create_gemini_attribution_extractor(
    config_path: Union[str, Path],
) -> GeminiAttributionExtractor:
    """Create AttributionExtractor from YAML config"""
    cfg = GeminiAttributionExtractor.read_yaml(config_path)
    prompt = GeminiAttributionExtractor.read_txt(cfg["prompt_path"])
    client = GeminiAttributionExtractor.get_genai_client()

    return GeminiAttributionExtractor(
        model_name=cfg["model_name"],
        client=client,
        prompt=prompt,
        temperature=cfg.get("temperature", 0.0),
    )


def create_gpt_attribution_extractor(
    config_path: Union[str, Path],
) -> GPTAttributionExtractor:
    """Create GPTAttributionExtractor from YAML config"""
    cfg = GPTAttributionExtractor.read_yaml(config_path)
    prompt = GPTAttributionExtractor.read_txt(cfg["prompt_path"])
    client = GPTAttributionExtractor.get_openai_client()

    return GPTAttributionExtractor(
        model_name=cfg["model_name"],
        client=client,
        prompt=prompt,
        temperature=cfg.get("temperature", 0.0),
    )
