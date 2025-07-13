from dotenv import load_dotenv
load_dotenv()

from typing import Union, Any
from pathlib import Path
import json
import os

from abc import ABC, abstractmethod
import numpy as np

from google import genai
from google.genai.types import GenerateContentConfig, Content, Part, GenerateContentResponse



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



class GeminiExtractor(BaseExtractor):

    mime_type = "image/jpeg"

    def __init__(self, model_name: str, client: genai.Client, prompt: str):
        self.model_name = model_name
        self.client = client
        self.prompt = prompt


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
    fname = "/Users/porterjenkins/data/price-attribution-scenes/test/price-images/0a1ce647-c6e5-42d8-84ea-42248991171e.jpg"

    prompt = "read the price from this price tag"
    client = GeminiExtractor.get_genai_client()
    gemini = GeminiExtractor(
        model_name="gemini-2.0-flash-001",
        client=client,
        prompt=prompt,
    )

    output = gemini(fname)
    print(output)