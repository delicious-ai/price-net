from dotenv import load_dotenv

from price_net.extraction.configs import ExtractionEvaluationConfig

load_dotenv()

from typing import Union, Any, Tuple
from pathlib import Path
import json
import os
import yaml

from abc import ABC, abstractmethod

from google import genai
from google.genai.types import GenerateContentConfig, Content, Part, GenerateContentResponse
import easyocr

from price_net.enums import PriceType

class BaseExtractor(ABC):


    @abstractmethod
    def __call__(self, img_input: Union[str, Path, bytes]) -> dict:
        """Major method for extracting price from an image"""
        pass

    @abstractmethod
    def format(self, price_json: dict) -> Tuple[PriceType, Tuple]:
        """Formats the raw output into a numpy array to compute error metrics"""
        pass

    @abstractmethod
    def format_as_str(self, price_json: dict) -> str:
        """Formats the raw output into a string to compute error metrics"""
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

    @classmethod
    def from_dict(cls, spec: dict):
        pass

    @classmethod
    def from_yaml(cls, model_config: Path | str):
        cfg = cls.read_yaml(model_config)
        return cls.from_dict(cfg)


class GeminiExtractor(BaseExtractor):

    mime_type = "image/jpeg"

    def __init__(self, model_name: str, client: genai.Client, prompt: str, temperature: float):
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
            return Part.from_bytes(data=open(img_input, "rb").read(), mime_type=self.mime_type)
        elif isinstance(img_input, bytes):
            return Part.from_bytes(data=img_input, mime_type=self.mime_type)
        else:
            raise TypeError("img_input must be str or bytes")

    def _api_call(self, img_input: Part) -> GenerateContentResponse:

        text_part = Part.from_text(text=self.prompt)
        raw_response = self.client.models.generate_content(
            model=self.model_name,
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
                        img_input,
                        text_part,
                    ],
                )
            ],
        )

        return raw_response

    def format(self, price_json: dict) -> Tuple[PriceType, Tuple]:
        price_type = PriceType(price_json["price_type"])

        if price_type == PriceType.STANDARD:
            price = (price_json["amount"])
        elif price_type == PriceType.BULK_OFFER:
            price = (
                price_json["quantity"],
                price_json["total_price"]
            )
        elif price_type == PriceType.BUY_X_GET_Y_FOR_Z:
            price = (
                price_json["buy_quantity"],
                price_json["get_quantity"],
                price_json["get_price"]
            )

        elif price_type == PriceType.UNKNOWN:
            price = ()

        else:
            price = ()

        return price_type, price

    def format_as_str(self, price_json: dict) -> str:
        price_type = PriceType(price_json["price_type"])
        if price_type == PriceType.STANDARD:
            price = f"${float(price_json['amount']):.2f}"
        elif price_type == PriceType.BULK_OFFER:
            price = f"{price_json['quantity']} / ${float(price_json['total_price']):.2f}"
        elif price_type == PriceType.BUY_X_GET_Y_FOR_Z:
            price = f"Buy {price_json['buy_quantity']}, Get {price_json['get_quantity']} / ${float(price_json['get_price']):.2f} "

        elif price_type == PriceType.UNKNOWN:
            price = "Unreadable"

        else:
            price = "Unknown"
            print(price_type)

        return price_type, price



    def __call__(self, img_input: Union[str, Path, bytes]) -> dict:

        img_input = self._route_input(img_input)
        raw_response = self._api_call(img_input)
        output = json.loads(raw_response.text.replace("'", '"'))

        return output

    @classmethod
    def from_dict(cls, cfg: dict):
        prompt = BaseExtractor.read_txt(cfg["prompt_fpath"])
        client = GeminiExtractor.get_genai_client()
        gemini = GeminiExtractor(
            model_name=cfg["model_name"],
            client=client,
            prompt=prompt,
            temperature=cfg["temperature"],
        )

        return gemini



class EasyOcrExtractor(BaseExtractor):

    def __init__(self, gpu: bool = False):
        self.gpu = gpu
        self.engine = easyocr.Reader(['en'])



    def __call__(self, img_input: Union[str, Path, bytes]) -> dict:

        if isinstance(img_input, Path):
            img_input = str(img_input)
        elif isinstance(img_input, bytes):
            raise NotImplementedError(f"img_input must be str or Path")

        result = self.engine.readtext(img_input, detail=0)
        results_str = ", ".join(result)
        output = {"output": results_str}
        return output

    def format(self, price_json: dict) -> Tuple[PriceType, Tuple]:
        return (None, ())

    def format_as_str(self, price_json: dict) -> str:
        return None, price_json["output"]

    def _route_input(self, img_input: Union[str, Path, bytes]) -> Union[str, Path]:
        return img_input

    @classmethod
    def from_dict(cls, cfg: dict):
        return EasyOcrExtractor(gpu = cfg["gpu"])

if __name__ == "__main__":
    fname = ""
    config = "configs/eval/extractors/easy-ocr.yaml"
    model = EasyOcrExtractor.from_yaml(config)
    output = model(fname)
    print(output)