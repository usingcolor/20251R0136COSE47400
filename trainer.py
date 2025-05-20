import torch
from omegaconf import OmegaConf

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from lightning.pytorch.loggers import WandbLogger


def main(train_config, model_config, ckpt_path=None, wandb_id=None):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    model_config = OmegaConf.load(model_config)
    train_config = OmegaConf.load(train_config)

    model = None  # TODO

    wandb_logger = WandbLogger(entity="translomo", project="MambaEye", id=wandb_id)
    wandb_logger.watch(model)

    wandb_id = wandb_logger.experiment.id

    trainer = L.Trainer(
        **train_config,
        callbacks=[
            ModelCheckpoint(
                monitor="val/classification_loss_last",
                dirpath="checkpoints/",
                filename=f"{wandb_id}_" + "{epoch}_{val/loss:.2f}_{val/acc_last:.2f}",
                save_top_k=5,
            )
        ],
        logger=wandb_logger,
    )

    trainer.fit(model, ckpt_path=ckpt_path)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
