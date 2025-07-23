from pathlib import Path
from pydantic import BaseModel
import yaml


class ExtractionEvaluationConfig(BaseModel):
    extractor_config_path: Path
    dataset_dir: Path
    cacheing: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExtractionEvaluationConfig":
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
