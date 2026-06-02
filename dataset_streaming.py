import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset
import tiktoken

class FineWebEduDataset(IterableDataset):
    def __init__(self, seq_len=1024):
        self.seq_len = seq_len
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc.eot_token

    def __iter__(self):
        buffer = []

        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )

        for example in dataset:
            tokens = self.enc.encode_ordinary(example["text"])
            tokens.append(self.eot)
            buffer.extend(tokens)

            # sempre que tiver tokens suficientes, emite uma sequência
            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len + 1:]

                x = torch.tensor(chunk[:-1], dtype=torch.long)  # input
                y = torch.tensor(chunk[1:],  dtype=torch.long)  # target (shifted by 1)
                yield {"input_ids": x, "labels": y}