from __future__ import annotations

from pathlib import Path

from price_net.enums import Accelerator
from price_net.enums import Aggregation
from price_net.enums import Precision
from price_net.enums import PredictionStrategy
from pydantic import BaseModel


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
    random_seed: int = 1998
    precision: Precision = Precision.FULL
    max_logit_magnitude: float | None = None
    accumulate_grad_batches: int = 1


class AssociatorEvaluationConfig(BaseModel):
    trn_config_path: Path
    ckpt_path: Path
    results_dir: Path
