from argparse import ArgumentParser
from typing import Optional

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import GPTNeoXTokenizerFast


class PILE_Dataset(Dataset):
    """Pile dataset wrapper. Using hugging face datasets"""

    def __init__(
        self,
        tokenizer: GPTNeoXTokenizerFast,
        block_size: int = 2048,
        name: Optional[str] = None,
        cache_dir: Optional[str] = None,
        split: Optional[str] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.block_size = block_size

        # TODO: This line can be removed, since this is run during tokenizer initialization.
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

        # TODO: dataset is different type if split is None
        self.dataset = load_dataset(
            "the_pile", name=name, cache_dir=cache_dir, split=split
        )

        # TODO: Better to be explicit here, can we require that split is never None?
        #       The load_dataset docs say: If None, will return a `dict` with all splits
        #       This might cause OOM issues if each rank is loading data it doesn't need.
        if not split:
            # default to train, then test, then valid
            if "train" in self.dataset.keys():
                self.dataset = self.dataset["train"]
            elif "test" in self.dataset.keys():
                self.dataset = self.dataset["test"]
            elif "validation" in self.dataset.keys():
                self.dataset = self.dataset["validation"]
            else:
                raise Exception("No split found for dataset")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> torch.Tensor:
        raw_text = self.dataset[idx]["text"]
        # return torch.Tensor(
        #     self.tokenizer(raw_text, return_tensors="pt", max_length=self.block_size, padding="max_length")
        # ).long()

        return torch.Tensor(
            self.tokenizer.encode(
                raw_text,
                max_length=self.block_size,
                truncation=True,
                padding="max_length",
            )
        ).long()


def main(args):
    dataset = PILE_Dataset(
        tokenizer=GPTNeoXTokenizerFast.from_pretrained("gpt2"),
        block_size=1024,
        name="enron_emails",
        cache_dir="/raid/khippe/hf_dataset_test",
    )

    print(len(dataset))
    print(dataset[100])


if __name__ == "__main__":
    parser = ArgumentParser()

    args = parser.parse_args()
    main(args)
