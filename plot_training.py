import json
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")

# Load the trainer state from the final checkpoint
with open("taikai-support-model/checkpoint-726/trainer_state.json") as f:
    state = json.load(f)

log_history = state["log_history"]

# Separate training loss and eval loss entries
train_steps = []
train_losses = []
eval_steps = []
eval_losses = []

for entry in log_history:
    if "loss" in entry and "eval_loss" not in entry:
        train_steps.append(entry["step"])
        train_losses.append(entry["loss"])
    if "eval_loss" in entry:
        eval_steps.append(entry["step"])
        eval_losses.append(entry["eval_loss"])

# Plot
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(train_steps, train_losses, label="Training Loss", color="#4A90D9", linewidth=1.5, alpha=0.8)
ax.plot(eval_steps, eval_losses, label="Validation Loss", color="#E74C3C", linewidth=2, marker="o", markersize=5)

# Mark the best checkpoint
best_step = state["best_global_step"]
best_metric = state["best_metric"]
ax.axvline(x=best_step, color="#2ECC71", linestyle="--", alpha=0.7, label=f"Best checkpoint (step {best_step})")
ax.annotate(
    f"Best eval loss: {best_metric:.4f}",
    xy=(best_step, best_metric),
    xytext=(best_step - 150, best_metric + 0.3),
    fontsize=9,
    arrowprops=dict(arrowstyle="->", color="#2ECC71"),
    color="#2ECC71",
)

ax.set_xlabel("Training Step", fontsize=12)
ax.set_ylabel("Loss", fontsize=12)
ax.set_title("Qwen3-4B LoRA Fine-Tuning on TAIKAI FAQ Data", fontsize=14, fontweight="bold")
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig("training_loss.png", dpi=150, bbox_inches="tight")
print("Saved training_loss.png")
