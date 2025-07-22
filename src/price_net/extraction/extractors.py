import numpy as np
from dotenv import load_dotenv
load_dotenv()

from typing import Union, Any, Tuple, Dict
from pathlib import Path
import json
import os
import yaml
import re
import io
import base64

from abc import ABC, abstractmethod

import easyocr
from google import genai
from google.cloud import vision
from google.genai.types import GenerateContentConfig, Content, Part, GenerateContentResponse


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

    def _route_input(self, img_input: Union[str, Path, bytes]) -> Union[str, Path]:
        return img_input


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
            price = (price_json["amount"],)
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
            price = (np.nan,)
        elif price_type == PriceType.MISC:
            price = (np.nan, price_json["contents"])
        else:
            raise ValueError(f"Unknown price type: {price_type}")

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

        elif price_type == PriceType.MISC:
            price = price_json["contents"]
        else:
            raise ValueError(f"Unknown price type: {price_type}")

        return price_type, price



    def __call__(self, img_input: Union[str, Path, bytes]) -> Dict:

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



    def __call__(self, img_input: Union[str, Path, bytes]) -> Dict:

        if isinstance(img_input, Path):
            img_input = str(img_input)
        elif isinstance(img_input, bytes):
            raise NotImplementedError(f"img_input must be str or Path")

        result = self.engine.readtext(img_input, detail=0)
        results_str = ", ".join(result)
        output = {"output": results_str}
        return output

    def format(self, price_json: dict) -> Tuple[PriceType, Tuple]:
        return None, (None,)

    def format_as_str(self, price_json: dict) -> str:
        text = price_json['output']

        extracted = []

        # X / $Y
        x_y_match = re.search(r'(\d+)\s*/\s*[$S](\d+(?:\.\d{1,2})?)', text)
        if x_y_match:
            quantity = int(x_y_match.group(1))
            price = float(x_y_match.group(2))
            price = f"{quantity} / ${price:.2f}"
            extracted.append(price)

        # BUY X GET Y FOR Z
        x_y_z_match = re.search(r'buy\s+(\d+)[,]?\s*get\s+(\d+)\s*/\s*[$s](\d+(?:\.\d{2})?)', text)
        if x_y_z_match:
            x = int(x_y_z_match[1])
            y = int(x_y_z_match[2])
            z = float(x_y_z_match[3])
            price = f"Buy {x}, Get {y} / ${z:.2f}"
            extracted.append(price)

        # Extract price (e.g., S5.99 or $5.99)
        price_match = re.search(r'[S\$](\d+(?:\.\d{1,2})?)', text)
        if price_match and not (x_y_match or x_y_z_match):
            price = float(price_match.group(1)) if price_match else None
            price = f"${price:.2f}"
            extracted.append(price)

        # Extract "BUY {quantity}" pattern
        buy_match = re.search(r'\bBUY\s+(\d+)', text, re.IGNORECASE)
        if buy_match and not x_y_z_match:
            buy_quantity = int(buy_match.group(1)) if buy_match else None
            buy_quantity = f"Buy {buy_quantity}"
            extracted.append(buy_quantity)

        output = " ".join(extracted)

        return None, output


    @classmethod
    def from_dict(cls, cfg: dict):
        return EasyOcrExtractor(gpu = cfg["gpu"])



class GoogleOcrExtractor(BaseExtractor):

    def __init__(self):
        # Initialize the client
        self.client = vision.ImageAnnotatorClient()

    @staticmethod
    def encode_image_to_base64(image_path: str) -> str:
        """
        Encode an image file to base64 string.

        Args:
            image_path: Path to the image file

        Returns:
            Base64 encoded string of the image
        """
        with open(image_path, 'rb') as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


    def extract_text_google_vision(self, image_path: str) -> Dict:
        """
        Extract text from an image using Google Cloud Vision API.

        Args:
            image_path: Path to the image file

        Returns:
            Dictionary containing extracted text and metadata
        """
        # Check if image file exists
        if not Path(image_path).exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")



        # Load the image into memory
        with io.open(image_path, 'rb') as image_file:
            content = image_file.read()

        # Create Image object
        image = vision.Image(content=content)

        # Perform text detection
        response = self.client.text_detection(image=image)

        # Check for errors
        if response.error.message:
            raise Exception(f"Google Vision API error: {response.error.message}")

        # Extract text annotations
        texts = response.text_annotations

        if not texts:
            print("No text detected in the image.")
            return {
                "full_text": "",
                "individual_words": [],
                "bounding_boxes": []
            }

        # The first annotation contains the full text
        full_text = texts[0].description

        # Extract individual words with their bounding boxes
        individual_words = []
        bounding_boxes = []

        for text in texts[1:]:  # Skip the first one as it's the full text
            # Get bounding box vertices
            vertices = []
            for vertex in text.bounding_poly.vertices:
                vertices.append({
                    "x": vertex.x,
                    "y": vertex.y
                })

            word_info = {
                "text": text.description,
                "bounding_box": vertices
            }

            individual_words.append(word_info)
            bounding_boxes.append(vertices)

        return {
            "full_text": full_text,
            "individual_words": individual_words
        }

    def __call__(self, image_path: str) -> Dict:
        return self.extract_text_google_vision(image_path)

    def format(self, price_json: dict) -> Tuple[PriceType | None, Tuple]:
        return None, (None,)

    def format_as_str(self, price_json: dict) -> str:
        text = price_json['full_text']

        extracted = []

        # X / $Y
        x_y_match = re.search(r'(\d+)\s*/\s*[$S](\d+(?:\.\d{1,2})?)', text)
        if x_y_match:
            quantity = int(x_y_match.group(1))
            price = float(x_y_match.group(2))
            price = f"{quantity} / ${price:.2f}"
            extracted.append(price)

        # BUY X GET Y FOR Z
        x_y_z_match = re.search(r'buy\s+(\d+)[,]?\s*get\s+(\d+)\s*/\s*[$s](\d+(?:\.\d{2})?)', text)
        if x_y_z_match:
            x = int(x_y_z_match[1])
            y = int(x_y_z_match[2])
            z = float(x_y_z_match[3])
            price = f"Buy {x}, Get {y} / ${z:.2f}"
            extracted.append(price)


        # Extract price (e.g., S5.99 or $5.99)
        price_match = re.search(r'[S\$](\d+(?:\.\d{1,2})?)', text)
        if price_match and not (x_y_match or x_y_z_match):
            price = float(price_match.group(1)) if price_match else None
            price = f"${price:.2f}"
            extracted.append(price)

        # Extract "BUY {quantity}" pattern
        buy_match = re.search(r'\bBUY\s+(\d+)', text, re.IGNORECASE)
        if buy_match and not x_y_z_match:
            buy_quantity = int(buy_match.group(1)) if buy_match else None
            buy_quantity = f"Buy {buy_quantity}"
            extracted.append(buy_quantity)


        output = " ".join(extracted)


        return None, output


    @classmethod
    def from_dict(cls, cfg: dict):
        return GoogleOcrExtractor()




if __name__ == "__main__":
    fname = "/Users/porterjenkins/data/price-attribution-scenes/test/price-images/0c0a4618-105f-4e6c-8047-d75c8d768489.jpg"
    config = "configs/eval/extractors/google-ocr.yaml"
    model = GoogleOcrExtractor.from_yaml(config)
    output = model(fname)
    print(output)