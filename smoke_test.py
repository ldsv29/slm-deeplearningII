from transformers import Trainer, TrainingArguments
from config import SLMConfig, SLMModel
from dataset_streaming import FineWebEduDataset

config = SLMConfig()
model = SLMModel(config)

# dataset pequeno: pega só 200 sequências do mesmo batch para overfit
from torch.utils.data import Dataset
import torch

class SmokeBatch(Dataset):
    def __init__(self):
        x = torch.randint(0, 50304, (1024,))
        self.sample = {"input_ids": x, "labels": x}

    def __len__(self):
        return 200

    def __getitem__(self, idx):
        return self.sample

args = TrainingArguments(
    output_dir="checkpoints_smoke",
    per_device_train_batch_size=4,
    max_steps=200,
    learning_rate=1e-3,
    report_to="none",
    use_cpu=True, 
)

trainer = Trainer(model=model, args=args, train_dataset=SmokeBatch())
trainer.train()