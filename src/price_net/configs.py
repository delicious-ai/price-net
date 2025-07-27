from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from price_net.enums import Accelerator
from price_net.enums import Aggregation
from price_net.enums import Precision
from price_net.enums import PredictionStrategy
from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator


class FeaturizationConfig(BaseModel):
    use_delta: bool = True
    use_prod_centroid: bool = True
    use_price_centroid: bool = True
    use_depth: bool = True
    use_prod_size: bool = True
    use_price_size: bool = True


class ModelConfig(BaseModel):
    prediction_strategy: PredictionStrategy = PredictionStrategy.MARGINAL
    aggregation: Aggregation = Aggregation.NONE
    featurization: FeaturizationConfig = FeaturizationConfig()
    settings: dict = {}


class LoggingConfig(BaseModel):
    use_wandb: bool = False
    project_name: str | None = None
    log_dir: Path = Path("logs")
    ckpt_dir: Path = Path("ckpt")


class AssociatorTrainingConfig(BaseModel):
    run_name: str
    dataset_dir: Path
    model: ModelConfig
    logging: LoggingConfig
    num_epochs: int = 1
    batch_size: int = 1
    num_workers: int = 0
    gamma: float = 1.0
    accelerator: Accelerator = Accelerator.CPU
    lr: float = 3e-4
    weight_decay: float = 1e-5
    warmup_pct: float = Field(ge=0.0, le=1.0, default=0.1)
    random_seed: int = 1998
    precision: Precision = Precision.FULL
    max_logit_magnitude: float | None = None
    accumulate_grad_batches: int = 1


class AssociatorEvaluationConfig(BaseModel):
    trn_config_path: Path
    ckpt_path: Path
    results_dir: Path


class ExtractionEvaluationConfig(BaseModel):
    extractor_config_path: Path
    dataset_dir: Path
    cacheing: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: str) -> ExtractionEvaluationConfig:
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


class AttributionEvaluationConfig(BaseModel):
    split: Literal["val", "test"] = "test"
    dataset_dir: Path
    results_dir: Path

    # Option 1: Cached attributions from a VLM.
    cached_attributions_path: Path | None = None

    # Option 2: PriceLens setup (assuming detection + extraction already run for ease of eval).
    cached_detections_path: Path | None = None
    associator_eval_config_path: Path | None = None

    @model_validator(mode="after")
    def check_mutually_exclusive_modes(self) -> AttributionEvaluationConfig:
        mode_flags = [
            bool(self.cached_attributions_path),
            bool(self.associator_eval_config_path),
        ]
        if sum(mode_flags) != 1:
            raise ValueError(
                "You must specify exactly one of: "
                "`cached_attributions_path` or `associator_eval_config_path`."
            )
        if self.associator_eval_config_path and not self.cached_detections_path:
            raise ValueError(
                "`cached_detections_path` is required when using an associator for attribution."
            )
        return self


class EndToEndConfig(BaseModel):
    model_name: str
    dataset_dir: Path
    prompt_path: Path
    output_path: Path
    temperature: float = 0.0
