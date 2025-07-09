import argparse
from pathlib import Path

import lightning as L
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.loggers import WandbLogger
from price_net.configs import TrainingConfig
from price_net.datamodule import PriceAssociationDataModule
from price_net.models import PriceAssociatorLightningModule
from price_net.utils import seed_everything


def train(config: TrainingConfig):
    seed_everything(config.random_seed)
    if config.logging.use_wandb:
        logger = WandbLogger(
            project=config.logging.project_name,
            name=config.run_name,
            tags=["TRAIN"],
        )
    else:
        logger = CSVLogger(
            save_dir=config.logging.log_dir,
            name=config.run_name,
            version="",
        )
    log_dir = config.logging.log_dir / config.run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / "train_config.yaml", "w") as f:
        yaml.dump(config.model_dump(mode="json"), f)

    best_ckpt_callback = ModelCheckpoint(
        dirpath=config.logging.ckpt_dir / config.run_name,
        filename="best_loss",
        monitor="val/loss",
        mode="min",
        save_top_k=1,
        enable_version_counter=False,
    )
    last_ckpt_callback = ModelCheckpoint(
        dirpath=config.logging.ckpt_dir / config.run_name,
        filename="last",
        save_last=True,
        enable_version_counter=False,
    )
    trainer = L.Trainer(
        accelerator=config.accelerator.value,
        min_epochs=config.num_epochs,
        max_epochs=config.num_epochs,
        logger=logger,
        callbacks=[best_ckpt_callback, last_ckpt_callback],
        enable_model_summary=False,
        precision=config.precision.value,
    )
    model = PriceAssociatorLightningModule(
        num_epochs=config.num_epochs,
        model_config=config.model,
        lr=config.lr,
        weight_decay=config.weight_decay,
        gamma=config.gamma,
        max_logit_magnitude=config.max_logit_magnitude,
    )
    datamodule = PriceAssociationDataModule(
        data_dir=config.dataset_dir,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        prediction_strategy=config.model.prediction_strategy,
        aggregation=config.model.aggregation,
        featurization_config=config.model.featurization,
    )
    trainer.fit(model, datamodule=datamodule)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    config = TrainingConfig(**config_dict)
    train(config)


if __name__ == "__main__":
    main()
