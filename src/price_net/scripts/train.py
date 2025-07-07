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


def train(config: TrainingConfig, num_trials: int = 1):
    for i in range(num_trials):
        run_name = f"{config.run_name}_{i}"
        if config.logging.use_wandb:
            logger = WandbLogger(
                project=config.logging.project_name,
                name=run_name,
                tags=["TRAIN"],
            )
        else:
            logger = CSVLogger(
                save_dir=config.logging.log_dir,
                name=run_name,
                version="",
            )
        log_dir = config.logging.log_dir / run_name
        log_dir.mkdir(parents=True, exist_ok=True)
        with open(log_dir / "train_config.yaml", "w") as f:
            yaml.dump(config.model_dump(mode="json"), f)

        best_ckpt_callback = ModelCheckpoint(
            dirpath=config.logging.ckpt_dir / run_name,
            filename="best_loss",
            monitor="val/loss",
            mode="min",
            save_top_k=1,
        )
        last_ckpt_callback = ModelCheckpoint(
            dirpath=config.logging.ckpt_dir / run_name,
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
        )
        model = PriceAssociatorLightningModule(
            num_epochs=config.num_epochs,
            model_config=config.model,
            lr=config.lr,
            weight_decay=config.weight_decay,
            gamma=config.gamma,
        )
        datamodule = PriceAssociationDataModule(
            data_dir=config.dataset_dir,
            batch_size=config.batch_size,
            num_workers=config.num_workers,
            input_reduction=config.model.input_reduction,
            prediction_strategy=config.model.prediction_strategy,
            featurization_method=config.model.featurization_method,
            use_depth=config.model.use_depth,
        )
        trainer.fit(model, datamodule=datamodule)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("-n", "--num-trials", type=int, default=1)
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config_dict = yaml.safe_load(f)
    config = TrainingConfig(**config_dict)
    train(config, num_trials=args.num_trials)


if __name__ == "__main__":
    main()
