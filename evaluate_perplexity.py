import torch
import tiktoken
from datasets import load_dataset
from torch.utils.data import DataLoader, IterableDataset
from config import SLMModel


class ValidationDataset(IterableDataset):
    def __init__(self, seq_len=1024, n_batches=50):
        self.seq_len = seq_len
        self.n_batches = n_batches
        self.enc = tiktoken.get_encoding("gpt2")

    def __iter__(self):
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="sample-10BT",
            split="train",
            streaming=True,
        )
        buffer = []
        count = 0
        skip = 500000

        for i, example in enumerate(dataset):
            if i < skip:
                continue

            tokens = self.enc.encode_ordinary(example["text"])
            tokens.append(self.enc.eot_token)
            buffer.extend(tokens)

            while len(buffer) >= self.seq_len + 1 and count < self.n_batches:
                chunk = buffer[:self.seq_len + 1]
                buffer = buffer[self.seq_len + 1:]
                yield {
                    "input_ids": torch.tensor(chunk[:-1], dtype=torch.long),
                    "labels":    torch.tensor(chunk[1:],  dtype=torch.long),
                }
                count += 1

            if count >= self.n_batches:
                break


def calculate_perplexity(model_id: str, n_batches: int = 50) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SLMModel.from_pretrained(model_id)
    model.eval()
    model.to(device)

    dataset = ValidationDataset(n_batches=n_batches)
    loader = DataLoader(dataset, batch_size=8)

    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            labels    = batch["labels"].to(device)
            outputs   = model(input_ids=input_ids, labels=labels)
            total_loss += outputs.loss.item()
            total_batches += 1

    avg_loss = total_loss / total_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    return perplexity


if __name__ == "__main__":
    models = {
        "pós pré-treino":  "ldsv29/slm-pretrained",
        "pós mid-training": "ldsv29/slm-midtraining",
    }

    for stage, model_id in models.items():
        print(f"Calculando perplexidade [{stage}]...")
        ppl = calculate_perplexity(model_id)
        print(f"  Perplexidade: {ppl:.2f}\n")