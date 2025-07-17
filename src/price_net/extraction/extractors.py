import argparse
import json
import os
from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Any
from typing import Union

import numpy as np
import yaml
from dotenv import load_dotenv
from google import genai
from google.genai.types import Content
from google.genai.types import GenerateContentConfig
from google.genai.types import GenerateContentResponse
from google.genai.types import Part
from tqdm import tqdm

load_dotenv()


class BaseExtractor(ABC):
    @abstractmethod
    def __call__(self, img_input: Union[str, Path, bytes]) -> dict:
        """Major method for extracting price from an image"""
        pass

    @abstractmethod
    def format(self, price_json: dict) -> np.ndarray:
        """Formats the raw output into a numpy array to compute error metrics"""
        pass

    @abstractmethod
    def _route_input(self, img_input: Union[str, Path, bytes]) -> Any:
        """From the raw input, get the correct type for downstream extraction analysis"""
        pass

    @staticmethod
    def read_yaml(file_path: Union[str, Path]) -> dict:
        """
        Reads a YAML file from disk and returns its contents as a Python object.

        Args:
            file_path (Union[str, Path]): Path to the YAML file.

        Returns:
            Any: Parsed contents of the YAML file
        """
        file_path = Path(file_path)
        with file_path.open("r") as f:
            return yaml.safe_load(f)

    @staticmethod
    def read_txt(file_path: Union[str, Path]) -> str:
        """
        Reads a text file from disk and returns its contents as a string.

        Args:
            file_path (Union[str, Path]): Path to the text file.

        Returns:
            str: Contents of the file.
        """
        file_path = Path(file_path)
        with file_path.open("r", encoding="utf-8") as f:
            return f.read()


class GeminiExtractor(BaseExtractor):
    mime_type = "image/jpeg"

    def __init__(
        self, model_name: str, client: genai.Client, prompt: str, temperature: float
    ):
        self.model_name = model_name
        self.client = client
        self.prompt = prompt
        self.temperature = temperature

    @classmethod
    def get_genai_client(cls):
        return genai.Client(
            vertexai=True,
            project=os.getenv("GOOGLE_CLOUD_PROJECT"),
            location=os.getenv("GOOGLE_CLOUD_LOCATION"),
        )

    def _route_input(self, img_input: Union[str, Path, bytes]) -> Part:
        if isinstance(img_input, str) | isinstance(img_input, Path):
            return Part.from_bytes(
                data=open(img_input, "rb").read(), mime_type=self.mime_type
            )
        elif isinstance(img_input, bytes):
            return Part.from_bytes(data=img_input, mime_type=self.mime_type)
        else:
            raise TypeError("img_input must be str or bytes")

    def _api_call(self, img_input: Part) -> GenerateContentResponse:
        text_part = Part.from_text(text=self.prompt)
        raw_response = self.client.models.generate_content(
            model=self.model_name,
            config=GenerateContentConfig(
                temperature=self.temperature,
                max_output_tokens=8192,
                response_modalities=["TEXT"],
                response_mime_type="application/json",
            ),
            contents=[
                Content(
                    role="user",
                    parts=[
                        img_input,
                        text_part,
                    ],
                )
            ],
        )

        return raw_response

    def format(self, price_json: dict) -> np.ndarray:
        pass

    def __call__(self, img_input: Union[str, Path, bytes]) -> dict:
        img_input = self._route_input(img_input)
        raw_response = self._api_call(img_input)
        output = json.loads(raw_response.text.replace("'", '"'))

        return output


class EasyOcrExtractor(BaseExtractor):
    pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path", type=Path, help="Path to the dataset (train or test)"
    )
    args = parser.parse_args()
    cfg = BaseExtractor.read_yaml("configs/eval/extractors/base-gemini.yaml")
    prompt = BaseExtractor.read_txt(cfg["prompt_fpath"])
    client = GeminiExtractor.get_genai_client()
    gemini = GeminiExtractor(
        model_name=cfg["model_name"],
        client=client,
        prompt=prompt,
        temperature=cfg["temperature"],
    )
    results_path = args.dataset_path / "extracted_prices.json"
    if os.path.exists(results_path):
        result = json.load(open(results_path))
        cached_ids = [x["price_id"] for x in result]
    else:
        cached_ids = []
        result = []
    for filename in tqdm(os.listdir(args.dataset_path / "price-images")):
        price_id = filename.split(".")[0]
        if price_id in cached_ids:
            continue
        filepath = args.dataset_path / "price-images" / filename
        output = gemini(filepath)
        result.append({"price_id": price_id, "price": output})
        with open(results_path, "w") as f:
            json.dump(result, f, indent=2)
