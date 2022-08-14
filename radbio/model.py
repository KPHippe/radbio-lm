import json
import os
from argparse import ArgumentParser
from typing import Any, List, Optional

import pytorch_lightning as pl
import torch
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.runtime.lr_schedules import WarmupLR
from pytorch_lightning.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# from pytorch_lightning.plugins import DeepSpeedPlugin
from pytorch_lightning.profiler import PyTorchProfiler
from pytorch_lightning.strategies.deepspeed import DeepSpeedStrategy
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast

from radbio.config import ModelSettings
from radbio.dataset import PILE_Dataset
from radbio.utils import LoadDeepSpeedStrategy, LoadPTCheckpointStrategy


class TransformerModel(pl.LightningModule):

    cfg: ModelSettings
    train_dataset: Dataset
    val_dataset: Dataset
    test_dataset: Dataset

    def __init__(self, cfg: ModelSettings) -> None:
        super().__init__()

        settings_dict = cfg.dict()
        with open(cfg.model_config_json, "r") as f:
            architecture = json.load(f)
            settings_dict["model_architecture"] = architecture
        self.save_hyperparameters(settings_dict)

        self.cfg = cfg
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.tokenizer_name, cache_dir=self.cfg.cache_dir
        )
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # loads from a json file like this: https://huggingface.co/google/reformer-enwik8/blob/main/config.json
        self.base_config = AutoConfig.from_pretrained(self.cfg.model_config_json)
        self.model = AutoModelForCausalLM.from_config(self.base_config)

    # def configure_sharded_model(self):
    #     self.model = AutoModelForCausalLM.from_config(self.base_config)

    def get_dataset(self, split: Optional[str] = None) -> Dataset:
        """Helper function to generate dataset."""
        return PILE_Dataset(
            name=self.cfg.dataset_name,
            tokenizer=self.tokenizer,
            cache_dir=self.cfg.cache_dir,
            split=split,
        )

    def get_dataloader(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        """Helper function to generate dataloader."""
        return DataLoader(
            dataset,
            shuffle=shuffle,
            drop_last=True,
            batch_size=self.cfg.batch_size,
            # num_workers=self.cfg.num_data_workers,
            # prefetch_factor=self.cfg.prefetch_factor,
            pin_memory=self.cfg.pin_memory,
            # persistent_workers=self.cfg.persistent_workers,
        )

    def train_dataloader(self) -> DataLoader:
        self.train_dataset = self.get_dataset(self.cfg.train_split)
        return self.get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        self.val_dataset = self.get_dataset(self.cfg.validation_split)
        return self.get_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        self.test_dataset = self.get_dataset(self.cfg.test_split)
        return self.get_dataloader(self.test_dataset, shuffle=False)

    def forward(self, x: torch.Tensor, **kwargs: Any) -> CausalLMOutputWithPast:  # type: ignore[override]
        return self.model(x, **kwargs)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.FloatTensor:
        outputs = self(batch)
        loss = outputs.loss
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.FloatTensor:  # type: ignore[override]
        outputs = self(batch)
        loss = outputs.loss
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch: torch.Tensor, batch_idx: int) -> torch.FloatTensor:
        outputs = self(batch)
        loss = outputs.loss
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> DeepSpeedCPUAdam:
        # optimizer = DeepSpeedCPUAdam(self.parameters(), lr=self.cfg.learning_rate)
        optimizer = FusedAdam(self.parameters(), lr=self.cfg.learning_rate)
        if self.cfg.warm_up_lr is not None:
            scheduler = WarmupLR(
                optimizer,
                warmup_min_lr=self.cfg.warm_up_lr.min_lr,
                warmup_max_lr=self.cfg.learning_rate,
                warmup_num_steps=self.cfg.warm_up_lr.num_steps,
            )
            return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

        return optimizer

    def lr_scheduler_step(self, scheduler, optimizer_idx, metric) -> None:
        scheduler.step()


def train(cfg: ModelSettings) -> None:
    if cfg.load_pt_checkpoint is not None:
        load_strategy = LoadPTCheckpointStrategy(cfg.load_pt_checkpoint, cfg=cfg)
        model = load_strategy.get_model(TransformerModel)
    elif cfg.load_ds_checkpoint is not None:
        load_strategy = LoadDeepSpeedStrategy(cfg.load_ds_checkpoint, cfg=cfg)
        model = load_strategy.get_model(TransformerModel)
        print(f"Loaded existing model at checkpoint {cfg.load_ds_checkpoint}....")
    else:
        model = TransformerModel(cfg)

    # Setup wandb
    wandb_logger = None
    if cfg.wandb_active:
        print("Using Weights and Biases for logging...")
        wandb_logger = WandbLogger(project=cfg.wandb_project_name)

    callbacks: List[Callback] = []
    if cfg.checkpoint_dir is not None:
        callbacks.append(
            ModelCheckpoint(
                dirpath=cfg.checkpoint_dir,
                save_last=True,
                verbose=True,
                monitor="val/loss",
                auto_insert_metric_name=False,
                filename="model-epoch{epoch:02d}-val_loss{val/loss:.2f}",
                save_top_k=3,
            )
        )

    if cfg.wandb_active:
        callbacks.append(LearningRateMonitor(logging_interval="step"))

    profiler = None
    if cfg.profiling_path:
        profiler = PyTorchProfiler(
            dirpath=cfg.profiling_path,
            profiler_kwargs={
                "activities": [
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                "schedule": torch.profiler.schedule(wait=0, warmup=1, active=3),
                "on_trace_ready": torch.profiler.tensorboard_trace_handler("./"),
            },
        )

    trainer = pl.Trainer(
        # use all available gpus
        accelerator="gpu",
        devices=-1,
        default_root_dir=str(cfg.checkpoint_dir),
        # Use NVMe offloading on other clusters see more here:
        # https://pytorch-lightning.readthedocs.io/en/stable/advanced/advanced_gpu.html#deepspeed-infinity-nvme-offloading
        strategy=DeepSpeedStrategy(
            stage=3,
            # offload_optimizer=True,
            # offload_parameters=True,
            # remote_device="cpu",
            # offload_params_device="cpu",
            # offload_optimizer_device="nvme",
            # nvme_path="/tmp",
            logging_batch_size_per_gpu=cfg.batch_size,
            # add the option to load a config from json file with more deepspeed options
            # note that if supplied all defaults are ignored - model settings defaults this arg to None
            # config=cfg.deepspeed_cfg_file
        ),
        callbacks=callbacks,
        # max_steps=cfg.training_steps,
        logger=wandb_logger,
        profiler=profiler,
        accumulate_grad_batches=cfg.accumulate_grad_batches,
        num_sanity_val_steps=0,
        precision=cfg.precision,
        max_epochs=cfg.epochs,
        num_nodes=cfg.num_nodes,
        check_val_every_n_epoch=-1,
    )

    trainer.fit(model)
    trainer.test(model)

    if trainer.is_global_zero:
        print("Completed training.")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-c", "--config", required=True)
    args = parser.parse_args()
    config = ModelSettings.from_yaml(args.config)

    # Setup torch environment
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["TORCH_EXTENSIONS_DIR"] = "/home/khippe/raid/torch_extensions"

    torch.set_num_threads(config.num_data_workers)  # type: ignore[attr-defined]
    pl.seed_everything(0)

    train(config)
