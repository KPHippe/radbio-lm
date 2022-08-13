from functools import partial
from pathlib import Path
from typing import Union, Optional
from argparse import ArgumentParser


import torch
from torch.utils.data import Dataset
from transformers import GPTNeoXTokenizerFast
from datasets import load_dataset


class PILE_Dataset(Dataset):
    """Pile dataset wrapper. Using hugging face datasets"""

    def __init__(
        self,
        tokenizer: GPTNeoXTokenizerFast,
        block_size: int,
        name: Optional[str] = None,
        cache_dir: Union[str, Path, None] = None,
        split: str = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.path = "the_pile"
        self.name = name
        self.cache_dir = cache_dir
        self.split = split

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # TODO: dataset is different type if split is None
        self.dataset = load_dataset(self.path, name=self.name, cache_dir=self.cache_dir, split=self.split)

        self.pad_sequence = partial(torch.nn.functional.pad, value=tokenizer.pad_token_id)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        raw_text = self.dataset[idx]["text"]
        print(raw_text)
        encoded_text = torch.tensor(
            self.tokenizer.encode(raw_text, max_length=self.block_size, padding="max_length")
        ).long()

        return encoded_text


def main(args):
    dataset = PILE_Dataset(
        tokenizer=GPTNeoXTokenizerFast.from_pretrained("gpt2"),
        block_size=1024,
        name="enron_emails",
        cache_dir="/raid/khippe/hf_dataset_test",
        split="all",
    )

    print(len(dataset))
    print(dataset[100])


if __name__ == "__main__":
    parser = ArgumentParser()

    args = parser.parse_args()
    main(args)
