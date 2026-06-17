from dataset_sft import SFTDataset
import tiktoken

ds = SFTDataset(seq_len=1024)
enc = tiktoken.get_encoding("gpt2")

sample = next(iter(ds))
print(sample["input_ids"].shape)   # [1024]
print(sample["labels"].shape)      # [1024]

# confirma que tokens do user têm label -100
labels = sample["labels"].tolist()
ids    = sample["input_ids"].tolist()
for token, label in zip(ids[50:100], labels[50:100]):
    print(f"{enc.decode([token])!r:20} -> label: {label}")