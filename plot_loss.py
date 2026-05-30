import json
import matplotlib.pyplot as plt

# o Trainer salva isso em checkpoints/trainer_state.json
with open("checkpoints_train\\checkpoint-200\\trainer_state.json") as f:
    state = json.load(f)

steps, losses = [], []
for entry in state["log_history"]:
    if "loss" in entry:
        steps.append(entry["step"])
        losses.append(entry["loss"])

plt.figure(figsize=(10, 5))
plt.plot(steps, losses)
plt.xlabel("Step")
plt.ylabel("Loss")
plt.title("Pre-training Loss")
plt.grid(True)
plt.savefig("loss_curve.png", dpi=150)
plt.show()