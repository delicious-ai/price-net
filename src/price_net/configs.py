from __future__ import annotations

from pathlib import Path

from price_net.enums import Accelerator
from price_net.enums import Aggregation
from price_net.enums import FeaturizationMethod
from price_net.enums import PredictionStrategy
from pydantic import BaseModel


class ModelConfig(BaseModel):
    prediction_strategy: PredictionStrategy = PredictionStrategy.MARGINAL
    aggregation: Aggregation = Aggregation.NONE
    featurization_method: FeaturizationMethod = FeaturizationMethod.CENTROID
    use_depth: bool = True
    settings: dict = {}


class LoggingConfig(BaseModel):
    use_wandb: bool = False
    project_name: str | None = None
    log_dir: Path = Path("logs")
    ckpt_dir: Path = Path("ckpt")


class TrainingConfig(BaseModel):
    run_name: str
    dataset_dir: Path
    model: ModelConfig
    logging: LoggingConfig
    num_epochs: int = 1
    batch_size: int = 1
    num_workers: int = 0
    gamma: float = 0.0
    accelerator: Accelerator = Accelerator.CPU
    lr: float = 3e-4
    weight_decay: float = 1e-5


class EvaluationConfig(BaseModel):
    trn_config_path: Path
    ckpt_path: Path
    results_dir: Path
