from pathlib import Path
from pydantic import BaseModel


class ExtractionEvaluationConfig(BaseModel):
    extractor_config_path: Path = Path("configs/eval/extractors/base-gemini.yaml")
    dataset_dir: Path = Path("/Users/porterjenkins/data/price-attribution-scenes/test")
    cacheing: bool = False


