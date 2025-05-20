import os
import torch
import torch.nn as nn
import lightning as L
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class VideoClassificationPL(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.config = config

        self.model = None  # TODO

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("lr", self.optimizers().param_groups[0]["lr"])

        sch = self.lr_schedulers()
        sch.step()
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("test/acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), **self.config.optimizer)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #     optimizer, **self.config.scheduler
        # )
        steps_per_epoch = self.config.scheduler.steps_per_epoch
        total_steps = self.config.scheduler.total_epochs * steps_per_epoch
        warmup_steps = self.config.scheduler.warmup_epochs * steps_per_epoch
        decay_steps = total_steps - warmup_steps

        peak_lr = self.config.optimizer.lr
        min_lr = self.config.scheduler.min_lr

        if warmup_steps > 0:
            # Initial LR for LinearLR is optimizer's base LR (peak_lr) times start_factor
            # We want to start near zero. Let's use a small factor.
            warmup_start_factor = (
                1e-7 / peak_lr if peak_lr > 0 else 0
            )  # Effectively start near zero
            scheduler_warmup = LinearLR(
                optimizer,
                start_factor=warmup_start_factor,
                end_factor=1.0,  # Reach peak_lr at the end
                total_iters=warmup_steps,
            )
        else:
            scheduler_warmup = None  # No warmup needed if warmup_epochs = 0

        if decay_steps > 0:
            scheduler_decay = CosineAnnealingLR(
                optimizer,
                T_max=decay_steps,  # Number of steps for the cosine decay phase
                eta_min=min_lr,  # Minimum learning rate
            )
        else:
            # If no decay phase (e.g., only warmup), create a dummy scheduler
            scheduler_decay = LinearLR(
                optimizer, start_factor=1.0, end_factor=1.0, total_iters=1
            )

        # --- Combine Schedulers ---
        if scheduler_warmup:
            # If there's a warmup phase, chain warmup then decay
            scheduler = SequentialLR(
                optimizer,
                schedulers=[scheduler_warmup, scheduler_decay],
                milestones=[warmup_steps],  # Switch schedulers *after* warmup_steps
            )
        else:
            # If no warmup, just use the decay scheduler
            scheduler = scheduler_decay

        return [optimizer], [scheduler]

    def train_dataloader(self):
        train_dataset = None  # TODO
        return torch.utils.data.DataLoader(
            train_dataset, **self.config.dataloader.train
        )

    def val_dataloader(self):
        val_dataset = None  # TODO
        return torch.utils.data.DataLoader(val_dataset, **self.config.dataloader.val)
