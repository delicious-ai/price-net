import base64
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
from openai import OpenAI

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

    def _api_call(
        self, img_input: Part, custom_prompt: str = None
    ) -> GenerateContentResponse:
        """
        Make API call to Gemini with image and text prompt.

        Args:
            img_input: The image part for the API call
            custom_prompt: Optional prompt to use instead of self.prompt
        """
        text_content = self.prompt if custom_prompt is None else custom_prompt
        text_part = Part.from_text(text=text_content)
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

    def __call__(
        self, img_input: Union[str, Path, bytes], custom_prompt: str = None
    ) -> dict:
        img_input = self._route_input(img_input)
        raw_response = self._api_call(img_input, custom_prompt)
        output = json.loads(raw_response.text.replace("'", '"'))

        return output


class GPTExtractor(BaseExtractor):
    """OpenAI GPT-based extractor for vision-language tasks"""

    def __init__(
        self, model_name: str, client: OpenAI, prompt: str, temperature: float
    ):
        self.model_name = model_name
        self.client = client
        self.prompt = prompt
        self.temperature = temperature

    @classmethod
    def get_openai_client(cls):
        """Get OpenAI client with API key from environment"""
        return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def _route_input(self, img_input: Union[str, Path, bytes]) -> str:
        """Convert image input to base64 encoded string for OpenAI API"""
        if isinstance(img_input, (str, Path)):
            with open(img_input, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode("utf-8")
        elif isinstance(img_input, bytes):
            return base64.b64encode(img_input).decode("utf-8")
        else:
            raise TypeError("img_input must be str, Path, or bytes")

    def _api_call(self, img_input: str, custom_prompt: str = None) -> dict:
        """
        Make API call to OpenAI with image and text prompt.

        Args:
            img_input: Base64 encoded image string
            prompt: Prompt to use (defaults to self.prompt if None)
        """
        text_content = custom_prompt if custom_prompt is not None else self.prompt
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_content},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{img_input}"},
                        },
                    ],
                }
            ],
            temperature=self.temperature,
            max_tokens=8192,
        )
        return response

    def format(self, price_json: dict) -> np.ndarray:
        pass

    def __call__(
        self, img_input: Union[str, Path, bytes], custom_prompt: str = None
    ) -> dict:
        img_b64 = self._route_input(img_input)
        raw_response = self._api_call(img_b64, custom_prompt)

        # Extract content from OpenAI response
        response_text = raw_response.choices[0].message.content

        # Clean the response text (remove any markdown formatting or extra text)
        if "```json" in response_text:
            start = response_text.find("```json") + 7
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()
        elif "```" in response_text:
            start = response_text.find("```") + 3
            end = response_text.find("```", start)
            response_text = response_text[start:end].strip()

        output = json.loads(response_text)

        return output


class EasyOcrExtractor(BaseExtractor):
    pass


if __name__ == "__main__":
    fname = "path-to-test-file"
    cfg = BaseExtractor.read_yaml("configs/eval/extractors/base-gemini.yaml")
    prompt = BaseExtractor.read_txt(cfg["prompt_fpath"])
    client = GeminiExtractor.get_genai_client()
    gemini = GeminiExtractor(
        model_name=cfg["model_name"],
        client=client,
        prompt=prompt,
        temperature=cfg["temperature"],
    )

    output = gemini(fname)
    print(output)
