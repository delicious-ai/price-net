from pathlib import Path

from price_net.enums import Extractor
from price_net.extraction.extractors import BaseExtractor
from price_net.extraction.extractors import EasyOcrExtractor
from price_net.extraction.extractors import GeminiExtractor
from price_net.extraction.extractors import GoogleOcrExtractor


class ExtractorFactory(object):
    def __init__(self, model_config_path: Path | str):
        self.model_config = BaseExtractor.read_yaml(model_config_path)
        self.model_type = Extractor(self.model_config["type"])

    def build(self):
        if self.model_type == Extractor.SINGLE_GEMINI:
            return GeminiExtractor.from_dict(self.model_config)
        elif self.model_type == Extractor.EASY_OCR:
            return EasyOcrExtractor.from_dict(self.model_config)
        elif self.model_type == Extractor.GOOGLE_OCR:
            return GoogleOcrExtractor.from_dict(self.model_config)
        elif self.model_type == Extractor.ENSEMBLE_GEMINI:
            raise NotImplementedError(f"Factory cannot build {self.model_type}")
        else:
            raise NotImplementedError
