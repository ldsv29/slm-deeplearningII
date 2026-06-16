from datasets import load_dataset, concatenate_datasets
import tiktoken
import torch
from torch.utils.data import Dataset


CHAT_TEMPLATE = "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>"

def format_smoltalk(example):
    messages = example["messages"]
    # SmolTalk tem lista de mensagens alternando user/assistant
    pairs = []
    for i in range(0, len(messages) - 1, 2):
        if messages[i]["role"] == "user" and messages[i+1]["role"] == "assistant":
            pairs.append(CHAT_TEMPLATE.format(
                user=messages[i]["content"],
                assistant=messages[i+1]["content"],
            ))
    return {"text": "\n".join(pairs)}


def format_gsm8k(example):
    return {"text": CHAT_TEMPLATE.format(
        user=example["question"],
        assistant=example["answer"],
    )}


def load_midtraining_data():
    smoltalk = load_dataset("HuggingFaceTB/smoltalk", "all", split="train")
    gsm8k    = load_dataset("openai/gsm8k", "main", split="train")

    smoltalk = smoltalk.map(format_smoltalk, remove_columns=smoltalk.column_names)
    gsm8k    = gsm8k.map(format_gsm8k,    remove_columns=gsm8k.column_names)

    return concatenate_datasets([smoltalk, gsm8k]).shuffle(seed=42)


class MidTrainingDataset(Dataset):
    def __init__(self, seq_len=1024):
        self.seq_len = seq_len
        self.enc = tiktoken.get_encoding("gpt2")

        raw = load_midtraining_data()

        # tokeniza tudo e concatena num buffer único (sequence packing)
        buffer = []
        for example in raw:
            tokens = self.enc.encode_ordinary(example["text"])
            tokens.append(self.enc.eot_token)
            buffer.extend(tokens)

        # corta o buffer em sequências de seq_len
        self.samples = []
        for i in range(0, len(buffer) - seq_len, seq_len):
            chunk = buffer[i : i + seq_len + 1]
            x = torch.tensor(chunk[:-1], dtype=torch.long)
            y = torch.tensor(chunk[1:],  dtype=torch.long)
            self.samples.append({"input_ids": x, "labels": y})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
