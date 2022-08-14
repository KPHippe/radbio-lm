"""Model configuration."""
import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Type, TypeVar, Union

import yaml
from pydantic import BaseSettings as _BaseSettings
from pydantic import root_validator, validator

_T = TypeVar("_T")

PathLike = Union[str, Path]


class BaseSettings(_BaseSettings):
    """Base settings to provide an easier interface to read/write YAML files."""

    def dump_yaml(self, cfg_path: PathLike) -> None:
        with open(cfg_path, mode="w") as fp:
            yaml.dump(json.loads(self.json()), fp, indent=4, sort_keys=False)

    @classmethod
    def from_yaml(cls: Type[_T], filename: PathLike) -> _T:
        with open(filename) as fp:
            raw_data = yaml.safe_load(fp)
        return cls(**raw_data)


class WarmupLRSettings(BaseSettings):
    """Learning rate warm up settings."""

    min_lr: float = 5e-8
    """The starting learning rate."""
    num_steps: int = 50000
    """Steps to warm up for."""


class ModelSettings(BaseSettings):
    """Settings for the TransformerModel model."""

    # logging settings
    wandb_active: bool = True
    """Whether to use wandb for logging."""
    wandb_project_name: str = "radbio_lm"
    """Wandb project name to log to."""
    checkpoint_dir: Optional[Path] = None
    """Checkpoint directory to backup model weights."""
    load_pt_checkpoint: Optional[Path] = None
    """Checkpoint pt file to initialze model weights."""
    load_ds_checkpoint: Optional[Path] = None
    """DeepSpeed checkpoint file to initialze model weights."""
    node_local_path: Optional[Path] = None
    """A node local storage option to write temporary files to."""
    num_nodes: int = 1
    """Flag for profiling - uses small subset to compute average throughput over 5 epochs after pinning."""
    profiling_path: Optional[Path] = None
    """Set to path if we want to run pytorch profiler"""

    # data settings
    dataset_name: Optional[str] = None
    """PILE subdataset to grab, defaults to none which grabs whole dataset"""
    tokenizer_name: Optional[str] = "EleutherAI/gpt-neox-20b"
    """Name of HF GPTNeoXTokenizerFast tokenizer to get"""
    split: Optional[str] = None
    """Dataset split to get, defaults to none"""
    train_split: str = "train"
    """Split to pull from for training"""
    test_split: Optional[str] = None
    """Split to pull from for testing, smaller pile datasets do not have this"""
    validation_split: Optional[str] = None
    """Split to pull from for validation, smaller pile datasets do not have this"""
    cache_dir: str = "/home/khippe/raid/gpt-neox-train/data/hf_pile"
    """Cache dir to look for the HF dataset"""

    # model settings
    model_config_json: Path
    """Huggingface json dict to load config from."""
    batch_size: int = 8
    """Training micro-batch size."""
    epochs: int = 5
    """Number of training epochs."""
    block_size: int = 2048
    """Block size to specify sequence length passed to the transformer."""
    accumulate_grad_batches: int = 1
    """Number of batches to accumulate before gradient updates."""
    learning_rate: float = 5e-5
    """Learning rate to use for training."""
    precision: int = 16
    """Training precision."""
    warm_up_lr: Optional[WarmupLRSettings] = None
    """If specified, will use a learning rate warmup scheduler."""
    deepspeed_cfg_file: Optional[Path] = None
    """The deepspeed configuration file (currently unused)."""
    check_val_every_n_epoch: int = 1
    """Run validation every n number of epochs"""

    # training ops (see PyTorch DataLoader for details.)
    num_data_workers: int = 4
    """Number of subprocesses to use for data loading."""
    prefetch_factor: int = 4
    """Number of batches loaded in advance by each worker."""
    pin_memory: bool = True
    """If True, the data loader will copy Tensors into device/CUDA pinned memory before returning them."""
    persistent_workers: bool = True
    """If True, the data loader will not shutdown the worker processes after a dataset has been consumed once."""

    @validator("node_local_path")
    def resolve_node_local_path(cls, v: Optional[Path]) -> Optional[Path]:
        # Check if node local path is stored in environment variable
        # Example: v = Path("$PSCRATCH") => str(v)[1:] == "PSCRATCH"
        return None if v is None else Path(os.environ.get(str(v)[1:], v))

    @root_validator
    def warn_checkpoint_load(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        load_pt_checkpoint = values.get("load_pt_checkpoint")
        load_ds_checkpoint = values.get("load_ds_checkpoint")
        if load_pt_checkpoint is not None and load_ds_checkpoint is not None:
            warnings.warn(
                "Both load_pt_checkpoint and load_ds_checkpoint are "
                "specified in the configuration. Loading from load_pt_checkpoint."
            )
        return values


if __name__ == "__main__":
    settings = ModelSettings(model_config_json=Path("../configs/1.3b_train_cfg.yml"))
    print(settings)
