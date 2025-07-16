from pathlib import Path

from price_net.enums import Extractor
from price_net.extraction.configs import ExtractionEvaluationConfig

from price_net.extraction.extractors import *

class ExtractorFactory(object):

    def __init__(self, model_config_path: Path | str):
        self.model_config = BaseExtractor.read_yaml(model_config_path)
        self.model_type = self.model_config["type"]

    def build(self):

        if self.model_type == Extractor.SINGLE_GEMINI.value:
            return GeminiExtractor.from_dict(self.model_config)
        elif self.model_type == Extractor.EASY_OCR.value:
            pass
        elif self.model_type == Extractor.ENSEMBLE_GEMINI.value:
            pass
        else:
            raise NotImplementedError

