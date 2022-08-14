from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Type

import pytorch_lightning as pl
from pytorch_lightning.utilities.deepspeed import (
    convert_zero_checkpoint_to_fp32_state_dict,
)


class ModelLoadStrategy(ABC):
    @abstractmethod
    def get_model(self, pl_module: "Type[pl.LightningModule]") -> "pl.LightningModule":
        """Load and return a module object."""


class LoadDeepSpeedStrategy(ModelLoadStrategy):
    def __init__(self, weight_path: Path, **kwargs: Any) -> None:
        """Load DeepSpeed checkpoint path.
        Parameters
        ----------
        weight_path : Path
            DeepSpeed checkpoint directory.
        """
        self.weight_path = weight_path
        self.kwargs = kwargs

    def get_model(self, pl_module: "Type[pl.LightningModule]") -> "pl.LightningModule":
        """Utility function for deepspeed conversion"""
        pt_file = str(self.weight_path.with_suffix(".pt"))
        # perform the conversion from deepspeed to pt weights
        convert_zero_checkpoint_to_fp32_state_dict(str(self.weight_path), pt_file)
        # load model
        model = pl_module.load_from_checkpoint(pt_file, strict=False, **self.kwargs)
        return model


class LoadPTCheckpointStrategy(ModelLoadStrategy):
    def __init__(self, weight_path: Path, **kwargs: Any) -> None:
        """Load a PyTorch model weight file.
        Parameters
        ----------
        weight_path : Path
            PyTorch model weight file.
        Raises
        ------
        ValueError
            If the `weight_path` does not have the `.pt` extension.
        """
        if weight_path.suffix != ".pt":
            raise ValueError("weight_path must be a .pt file")
        self.weight_path = weight_path
        self.kwargs = kwargs

    def get_model(self, pl_module: "Type[pl.LightningModule]") -> "pl.LightningModule":
        model = pl_module.load_from_checkpoint(
            str(self.weight_path), strict=False, **self.kwargs
        )
        return model
