import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from config import SLMModel


def evaluate_hellaswag(model_id: str, n_samples: int = 200) -> float:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SLMModel.from_pretrained(model_id)
    model.eval()
    model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset("Rowan/hellaswag", split="validation", streaming=True)

    correct = 0
    total = 0

    with torch.no_grad():
        for example in dataset:
            if total >= n_samples:
                break

            context = example["ctx"]
            endings = example["endings"]
            label = int(example["label"])

            scores = []
            for ending in endings:
                text = context + " " + ending
                inputs = tokenizer(text, return_tensors="pt", truncation=True,
                                   max_length=1024).to(device)
                input_ids = inputs["input_ids"]

                outputs = model(input_ids=input_ids, labels=input_ids)
                n_tokens = input_ids.shape[1]
                log_prob = -outputs.loss.item() * n_tokens
                scores.append(log_prob)

            predicted = scores.index(max(scores))
            if predicted == label:
                correct += 1
            total += 1

    accuracy = correct / total
    return accuracy


if __name__ == "__main__":
    models = {
        "pós pré-treino":   "ldsv29/slm-pretrained",
        "pós mid-training": "ldsv29/slm-midtraining",
        "pós SFT":          "ldsv29/slm-sft",
    }

    for stage, model_id in models.items():
        print(f"Avaliando HellaSwag [{stage}]...")
        acc = evaluate_hellaswag(model_id, n_samples=200)
        print(f"  Acurácia: {acc:.2%}\n")
