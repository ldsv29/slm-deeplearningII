from transformers import Trainer, TrainingArguments
from config import SLMConfig, SLMModel
from dataset_sft import SFTDataset

MIDTRAINING_CHECKPOINT = "ldsv29/slm-midtraining"

config = SLMConfig()
model  = SLMModel.from_pretrained(MIDTRAINING_CHECKPOINT)

dataset = SFTDataset(seq_len=config.max_position_embeddings)

args = TrainingArguments(
    output_dir="checkpoints_sft",
    per_device_train_batch_size=16,
    gradient_accumulation_steps=8,
    bf16=True,
    max_steps=500=,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_steps=50,
    weight_decay=0.1,
    adam_beta1=0.9,
    adam_beta2=0.95,
    max_grad_norm=1.0,
    logging_steps=10,
    save_steps=200,
    save_total_limit=3,
    report_to="none",
    push_to_hub=True,
    hub_model_id="ldsv29/slm-sft",
    hub_strategy="checkpoint",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset,
)

trainer.train()