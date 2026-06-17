from datasets import load_dataset, concatenate_datasets, interleave_datasets
import tiktoken
import torch
from torch.utils.data import IterableDataset


CHAT_TEMPLATE = "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>"


def format_smoltalk(example):
    messages = example["messages"]
    pairs = []
    for i in range(0, len(messages) - 1, 2):
        if messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
            pairs.append(CHAT_TEMPLATE.format(
                user=messages[i]["content"],
                assistant=messages[i+1]["content"],
            ))
    return "\n".join(pairs)


def format_gsm8k(example):
    return CHAT_TEMPLATE.format(
        user=example["question"],
        assistant=example["answer"],
    )


class MidTrainingDataset(IterableDataset):
    def __init__(self, seq_len=1024):
        self.seq_len = seq_len
        self.enc = tiktoken.get_encoding("gpt2")

    def __iter__(self):
        smoltalk = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)
        gsm8k = load_dataset("openai/gsm8k", "main", split="train", streaming=True)

        buffer = []

        for example in smoltalk:
            text = format_smoltalk(example)
            if not text:
                continue
            tokens = self.enc.encode_ordinary(text)
            tokens.append(self.enc.eot_token)
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len + 1:]
                yield {"input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                       "labels":    torch.tensor(chunk[1:],  dtype=torch.long)}

        for example in gsm8k:
            text = format_gsm8k(example)
            tokens = self.enc.encode_ordinary(text)
            tokens.append(self.enc.eot_token)
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len + 1:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len + 1:]
                yield {"input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                       "labels":    torch.tensor(chunk[1:],  dtype=torch.long)}