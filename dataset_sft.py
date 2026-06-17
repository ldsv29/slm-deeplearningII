import tiktoken
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset

def build_sample(user: str, assistant: str, enc, seq_len: int):
    user_part = f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    assistant_part = f"{assistant}<|im_end|>"

    user_tokens = enc.encode_ordinary(user_part)
    assistant_tokens = enc.encode_ordinary(assistant_part)
    assistant_tokens.append(enc.eot_token)

    input_ids = user_tokens + assistant_tokens
    labels = [-100] * len(user_tokens) + assistant_tokens

    input_ids = input_ids[:seq_len]
    labels = labels[:seq_len]

    pad_len = seq_len - len(input_ids)
    input_ids += [enc.eot_token] * pad_len
    labels += [-100] * pad_len

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


class SFTDataset(IterableDataset):
    def __init__(self, seq_len=1024):
        self.seq_len = seq_len
        self.enc = tiktoken.get_encoding("gpt2")

    def __iter__(self):
        dataset = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True)

        for example in dataset:
            messages = example["messages"]
            for i in range(0, len(messages) - 1, 2):
                if messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
                    yield build_sample(
                        user=messages[i]["content"],
                        assistant=messages[i+1]["content"],
                        enc=self.enc,
                        seq_len=self.seq_len,
                    )