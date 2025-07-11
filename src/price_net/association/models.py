from functools import partial
from typing import Callable
from typing import Literal

import lightning as L
import torch
from price_net.association.configs import ModelConfig
from price_net.association.losses import sigmoid_focal_loss_star
from price_net.enums import PredictionStrategy
from torch import nn
from torch.nn import TransformerEncoder
from torch.nn import TransformerEncoderLayer
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import BinaryF1Score
from torchmetrics.classification import BinaryPrecision
from torchmetrics.classification import BinaryRecall


class MarginalPriceAssociator(nn.Module):
    """A neural network that predicts the association probability between a product group and price tag, independent of any other potential assocations in the scene.

    This network is a simple MLP that takes in a representation of the proposed association and outputs binary classification logits representing its belief that the
    proposal is a true product-price association.
    """

    def __init__(self, layer_widths: list[int] = [128, 64], dropout: float = 0.0):
        """Initialize a `MarginalPriceAssociator`.

        Args:
            layer_widths (list[int], optional): The MLP layer widths for this model. Defaults to [128, 64].
            dropout (float, optional): Dropout probability (to be used in each MLP layer besides the last). Defaults to 0.0.
        """
        super().__init__()
        self.mlp = nn.ModuleList()
        for i, width in enumerate(layer_widths):
            self.mlp.append(nn.LazyLinear(out_features=width))
            if i < len(layer_widths) - 1:
                self.mlp.append(nn.ReLU())
                self.mlp.append(nn.Dropout(dropout))
            else:
                self.mlp.append(nn.LazyLinear(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Make a forward pass through the model.

        Args:
            x (torch.Tensor): A (B, D) tensor of potential product-price associations.

        Returns:
            torch.Tensor: A (B,) tensor of association probability logits (to be passed through a sigmoid function).
        """
        for layer in self.mlp:
            x = layer(x)
        return x


class JointPriceAssociator(nn.Module):
    """A neural network that predicts the association probability between a product group and price tag, conditioned on all other potential associations in the scene.

    This network jointly predicts on all potential associations in a scene using a transformer encoder architecture (followed by a shared MLP that is applied to all tokens).
    """

    def __init__(
        self,
        num_encoder_layers: int = 1,
        d_model: int = 128,
        d_feedforward: int = 512,
        num_heads: int = 2,
        mlp_layer_widths: list[int] = [128, 64],
        dropout: float = 0.0,
    ):
        """Initialize a `JointPriceAssociator`.

        Args:
            num_encoder_layers (int, optional): The number of encoder layers in the associator. Defaults to 1.
            d_model (int, optional): Dimension of internal encoder representations. Defaults to 128.
            d_feedforward (int, optional): Dimension of internal representation used by feedforward portion of Transformer block. Defaults to 512.
            num_heads (int, optional): The number of heads for the Transformer's multi-headed attention operation. Defaults to 2.
            mlp_layer_widths (list[int], optional): The layer widths for the shared predictor MLP (used on each encoding). Defaults to [128, 64].
            dropout (float, optional): Dropout probability to use in each encoder layer (as well as each MLP layer besides the final one). Defaults to 0.0.
        """
        super().__init__()
        layer_template = TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.d_model = d_model
        self.projector = nn.LazyLinear(d_model)
        self.encoder = TransformerEncoder(
            encoder_layer=layer_template,
            num_layers=num_encoder_layers,
            enable_nested_tensor=False,
        )

        self.mlp = nn.ModuleList()
        for i, width in enumerate(mlp_layer_widths):
            self.mlp.append(nn.LazyLinear(out_features=width))
            if i < len(mlp_layer_widths) - 1:
                self.mlp.append(nn.ReLU())
                self.mlp.append(nn.Dropout(dropout))
            else:
                self.mlp.append(nn.LazyLinear(1))

    def forward(
        self, x: torch.Tensor, padded_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Make a forward pass through the model.

        Args:
            x (torch.Tensor): A (B, N, D) tensor of potential product-price associations. Each (N, D) subtensor represents all potential associations for a single shelf image (scene).
            padded_mask (torch.Tensor, optional): A (B, N) tensor indicating which tokens are zero-padded in the batched input (and should be ignored). Defaults to None (no tokens ignored).

        Returns:
            torch.Tensor: A (B, N) tensor of association probability logits (to be passed through a sigmoid function).
        """
        B, N, _ = x.shape
        x_proj = torch.reshape(self.projector(x.flatten(0, 1)), (B, N, self.d_model))
        h: torch.Tensor = self.encoder(
            src=x_proj, is_causal=False, src_key_padding_mask=padded_mask
        )
        h_flat = h.flatten(0, 1)
        for layer in self.mlp:
            h_flat = layer(h_flat)
        logits = h_flat.reshape(B, N)
        return logits


ASSOCIATOR_REGISTRY: dict[PredictionStrategy, type[nn.Module]] = {
    PredictionStrategy.MARGINAL: MarginalPriceAssociator,
    PredictionStrategy.JOINT: JointPriceAssociator,
}


class PriceAssociatorLightningModule(L.LightningModule):
    """A Lightning wrapper around a `{ Marginal | Joint }PriceAssociator`."""

    objective: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

    def __init__(
        self,
        num_epochs: int,
        model_config: ModelConfig,
        lr: float = 3e-4,
        weight_decay: float = 1e-5,
        gamma: float = 0.0,
        max_logit_magnitude: float | None = None,
    ):
        super().__init__()

        # Setup model.
        self.strategy = model_config.prediction_strategy
        self.associator = ASSOCIATOR_REGISTRY[self.strategy](**model_config.settings)

        # Setup training hparams.
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.objective = partial(
            sigmoid_focal_loss_star,
            alpha=-1,
            gamma=self.gamma,
            reduction="none",
        )
        self.max_logit_magnitude = max_logit_magnitude

        # Setup metrics.
        self.trn_precision = BinaryPrecision()
        self.trn_recall = BinaryRecall()
        self.trn_f1 = BinaryF1Score()
        self.val_precision = BinaryPrecision()
        self.val_recall = BinaryRecall()
        self.val_f1 = BinaryF1Score()

        self.save_hyperparameters()

    def forward(
        self, x: torch.Tensor, padded_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self.strategy == PredictionStrategy.MARGINAL:
            return self.associator(x)
        else:
            return self.associator(x=x, padded_mask=padded_mask)

    def training_step(self, batch, batch_idx):
        return self._step(batch, step_type="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, step_type="val")

    def _step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        step_type: Literal["train", "val"],
    ) -> torch.Tensor:
        """Handle all the necessary logic for a single training / validation / testing forward pass.

        Args:
            batch (tuple[torch.Tensor, torch.Tensor, torch.Tensor]): A batch of data.
            step_type (Literal["train", "val"]: Specifies which type of step we are taking.

        Returns:
            torch.Tensor: The loss accumulated during the forward pass.
        """
        if step_type == "train":
            precision = self.trn_precision
            recall = self.trn_recall
            f1 = self.trn_f1
        elif step_type == "val":
            precision = self.val_precision
            recall = self.val_recall
            f1 = self.val_f1
        else:
            raise NotImplementedError(
                "Unsupported step_type passed to PriceAttributor._step"
            )
        if self.strategy == PredictionStrategy.MARGINAL:
            X, y = batch
            num_associations = len(y)
            logits = self.forward(X).flatten()
            if self.max_logit_magnitude is not None:
                logits = logits.clamp(
                    -self.max_logit_magnitude, self.max_logit_magnitude
                )
            loss = self.objective(logits, y).mean()
            probs = logits.sigmoid()
            precision.update(probs, y)
            recall.update(probs, y)
            f1.update(probs, y)
        else:
            X, y, padded_mask = batch
            logits = self.forward(X, padded_mask)
            if self.max_logit_magnitude is not None:
                logits = logits.clamp(
                    -self.max_logit_magnitude, self.max_logit_magnitude
                )
            loss_per_token = self.objective(logits, y)
            active_mask = ~padded_mask
            num_associations = active_mask.sum().item()
            loss = (loss_per_token * active_mask).sum() / num_associations
            probs = logits.sigmoid()
            precision.update(probs[active_mask], y[active_mask])
            recall.update(probs[active_mask], y[active_mask])
            f1.update(probs[active_mask], y[active_mask])
        self.log(
            f"{step_type}/loss",
            loss,
            on_epoch=True,
            on_step=False,
            prog_bar=True,
            batch_size=num_associations,
        )
        return loss

    def _epoch_end(self, step_type: Literal["train", "val", "test"]):
        if step_type == "train":
            precision = self.trn_precision.compute()
            recall = self.trn_recall.compute()
            f1 = self.trn_f1.compute()
        elif step_type == "val":
            precision = self.val_precision.compute()
            recall = self.val_recall.compute()
            f1 = self.val_f1.compute()
        elif step_type == "test":
            precision = self.test_precision.compute()
            recall = self.test_recall.compute()
            f1 = self.test_f1.compute()
        else:
            raise NotImplementedError("Unsupported step type")

        self.log(f"{step_type}/precision", precision)
        self.log(f"{step_type}/recall", recall)
        self.log(f"{step_type}/f1", f1)

    def on_train_epoch_end(self):
        self._epoch_end("train")

    def on_validation_epoch_end(self):
        self._epoch_end("val")

    def on_test_epoch_end(self):
        self._epoch_end("test")

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.num_epochs, eta_min=self.lr / 10
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
