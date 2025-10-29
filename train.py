import os
import math
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
from utils import build_wcst_token_map
from dataset import prepare_wcst_datasets
from transformer import Transformer  # must output attention weights when return_attention=True




# =====================================
# 1️⃣ Configuration
# =====================================
#
token_map = build_wcst_token_map()
config = {
    "src_vocab_size": 70,  # 0–69 as per WCST
    "tgt_vocab_size": 70,
    "d_model": 256,
    "num_heads": 8,
    "num_layers": 4,
    "d_ff": 1024,
    "dropout": 0.1,
    "max_seq_length": 20,
    "batch_size": 64,
    "epochs": 1500,
    "lr": 1e-4,
    "save_dir": "./checkpoints",
    "data_path": "./wcst_data",
    "resume": True,  # resume if checkpoint exists
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu"
}


# =====================================
# 2️⃣ Set Random Seeds
# =====================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(config["seed"])


# =====================================
# 3️⃣ Initialize W&B
# =====================================
wandb.init(project="wcst-transformer", id="wcst-transformer", config=config)
wandb.run.name = f"wcst_transformer_seed{config['seed']}"


# =====================================
# 4️⃣ Load Data
# =====================================
train_loader, val_loader, test_loader = prepare_wcst_datasets(
    batch_size=config["batch_size"],
    num_batches=1000,
    save_path=config["data_path"],
    regenerate=False
)


# =====================================
# 5️⃣ Model, Loss, Optimizer
# =====================================
model = Transformer(
    config["src_vocab_size"],
    config["tgt_vocab_size"],
    config["d_model"],
    config["num_heads"],
    config["num_layers"],
    config["d_ff"],
    config["max_seq_length"],
    config["dropout"]
).to(config["device"])

criterion = nn.CrossEntropyLoss(ignore_index=0)
optimizer = optim.Adam(model.parameters(), lr=config["lr"], betas=(0.9, 0.98), eps=1e-9)


# =====================================
# 6️⃣ Checkpoint Handling
# =====================================
os.makedirs(config["save_dir"], exist_ok=True)
checkpoint_path = os.path.join(config["save_dir"], "wcst_transformer.pt")

start_epoch = 0
best_val_loss = float("inf")

if config["resume"] and os.path.exists(checkpoint_path):
    ckpt = torch.load(checkpoint_path)
    model.load_state_dict(ckpt["model_state"])
    optimizer.load_state_dict(ckpt["optimizer_state"])
    start_epoch = ckpt["epoch"] + 1
    best_val_loss = ckpt["best_val_loss"]
    print(f"🔁 Resuming from epoch {start_epoch}, best val loss {best_val_loss:.4f}")


# =====================================
# 7️⃣ Helper Functions
# =====================================
def compute_accuracy(preds, targets):
    preds = preds.argmax(dim=-1)
    mask = targets != 0
    correct = (preds == targets) & mask
    return (correct.sum().float() / mask.sum().float()).item()

def per_category_accuracy(preds, targets):
    preds = preds.argmax(dim=-1)
    mask = targets != 0
    preds, targets = preds[mask], targets[mask]
    accs = {}
    for cat_idx, name in zip(range(64, 68), ["C1", "C2", "C3", "C4"]):
        relevant = targets == cat_idx
        if relevant.sum() == 0:
            accs[name] = None
        else:
            accs[name] = (preds[relevant] == targets[relevant]).float().mean().item()
    return accs


def visualize_attention(all_attn_weights, epoch, src_sample, tgt_sample, token_map, max_heads=2):
    """
    Visualize both self- and cross-attention from the last decoder layer,
    with readable WCST token labels.

    Args:
        all_attn_weights (list[dict]): attention maps from decoder layers
        epoch (int): current epoch
        src_sample (Tensor): sample source sequence (token IDs)
        tgt_sample (Tensor): sample target sequence (token IDs)
        token_map (dict): maps token IDs to human-readable strings
        max_heads (int): number of attention heads to visualize
    """
    if not all_attn_weights or not isinstance(all_attn_weights[-1], dict):
        return

    last_layer = all_attn_weights[-1]
    self_attn = last_layer.get("self_attn", None)
    cross_attn = last_layer.get("cross_attn", None)
    if self_attn is None or cross_attn is None:
        return

    self_attn = self_attn.detach().cpu().numpy()
    cross_attn = cross_attn.detach().cpu().numpy()

    batch_idx = 0
    num_heads = min(self_attn.shape[1], max_heads)

    src_tokens = [token_map.get(int(tok), str(int(tok))) for tok in src_sample.cpu().numpy() if tok != 0]
    tgt_tokens = [token_map.get(int(tok), str(int(tok))) for tok in tgt_sample.cpu().numpy() if tok != 0]

    fig, axes = plt.subplots(2, num_heads, figsize=(4 * num_heads, 8))

    for i in range(num_heads):
        # Self-Attention (Decoder ↔ Decoder)
        ax = axes[0, i]
        attn = self_attn[batch_idx, i, :len(tgt_tokens), :len(tgt_tokens)]
        im = ax.imshow(attn, cmap="viridis")
        ax.set_title(f"Self-Attn Head {i}")
        ax.set_xticks(range(len(tgt_tokens)))
        ax.set_xticklabels(tgt_tokens, rotation=90, fontsize=8)
        ax.set_yticks(range(len(tgt_tokens)))
        ax.set_yticklabels(tgt_tokens, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Cross-Attention (Decoder ↔ Encoder)
        ax = axes[1, i]
        attn = cross_attn[batch_idx, i, :len(tgt_tokens), :len(src_tokens)]
        im = ax.imshow(attn, cmap="magma")
        ax.set_title(f"Cross-Attn Head {i}")
        ax.set_xticks(range(len(src_tokens)))
        ax.set_xticklabels(src_tokens, rotation=90, fontsize=8)
        ax.set_yticks(range(len(tgt_tokens)))
        ax.set_yticklabels(tgt_tokens, fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    wandb.log({f"attention_epoch_{epoch+1}": wandb.Image(fig)})
    plt.close(fig)

# =====================================
# 8️⃣ Training Loop
# =====================================
for epoch in range(start_epoch, config["epochs"]):
    model.train()
    total_train_loss, total_train_acc = 0, 0

    for src, tgt in tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]"):
        src, tgt = src.to(config["device"]), tgt.to(config["device"])
        optimizer.zero_grad()

        output, attn_weights = model(src, tgt[:, :-1], return_attention=True)
        loss = criterion(output.reshape(-1, config["tgt_vocab_size"]), tgt[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()

        acc = compute_accuracy(output, tgt[:, 1:])
        total_train_loss += loss.item()
        total_train_acc += acc

    avg_train_loss = total_train_loss / len(train_loader)
    avg_train_acc = total_train_acc / len(train_loader)

    # =====================================
    # Validation
    # =====================================
    model.eval()
    total_val_loss, total_val_acc = 0, 0
    cat_acc_totals = {"C1": [], "C2": [], "C3": [], "C4": []}

    with torch.no_grad():
        for src, tgt in tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]"):
            src, tgt = src.to(config["device"]), tgt.to(config["device"])
            output, attn_weights = model(src, tgt[:, :-1], return_attention=True)
            loss = criterion(output.reshape(-1, config["tgt_vocab_size"]), tgt[:, 1:].reshape(-1))
            acc = compute_accuracy(output, tgt[:, 1:])
            cats = per_category_accuracy(output, tgt[:, 1:])

            total_val_loss += loss.item()
            total_val_acc += acc
            for k, v in cats.items():
                if v is not None:
                    cat_acc_totals[k].append(v)

    avg_val_loss = total_val_loss / len(val_loader)
    avg_val_acc = total_val_acc / len(val_loader)
    avg_cat_acc = {k: np.mean(v) if len(v) > 0 else None for k, v in cat_acc_totals.items()}

    # Log one attention map per epoch
    visualize_attention(attn_weights, epoch, src[0], tgt[0], token_map)

    # =====================================
    # Logging to W&B
    # =====================================
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "train_accuracy": avg_train_acc,
        "val_loss": avg_val_loss,
        "val_accuracy": avg_val_acc,
        **{f"val_acc_{k}": v for k, v in avg_cat_acc.items()}
    })

    print(
        f"Epoch {epoch+1}: "
        f"TrainLoss={avg_train_loss:.4f}, ValLoss={avg_val_loss:.4f}, "
        f"TrainAcc={avg_train_acc:.4f}, ValAcc={avg_val_acc:.4f}, "
        f"PerCat={avg_cat_acc}"
    )

    # =====================================
    # Checkpoint
    # =====================================
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_loss": best_val_loss
        }, checkpoint_path)
        print("💾 Saved new best model.")


# =====================================
# 9️⃣ Test Evaluation
# =====================================
ckpt = torch.load(checkpoint_path)
model.load_state_dict(ckpt["model_state"])
model.eval()

total_test_loss, total_test_acc = 0, 0
cat_acc_totals = {"C1": [], "C2": [], "C3": [], "C4": []}

with torch.no_grad():
    for src, tgt in tqdm(test_loader, desc="Testing"):
        src, tgt = src.to(config["device"]), tgt.to(config["device"])
        output, _ = model(src, tgt[:, :-1], return_attention=True)
        loss = criterion(output.reshape(-1, config["tgt_vocab_size"]), tgt[:, 1:].reshape(-1))
        acc = compute_accuracy(output, tgt[:, 1:])
        cats = per_category_accuracy(output, tgt[:, 1:])
        total_test_loss += loss.item()
        total_test_acc += acc
        for k, v in cats.items():
            if v is not None:
                cat_acc_totals[k].append(v)

avg_test_loss = total_test_loss / len(test_loader)
avg_test_acc = total_test_acc / len(test_loader)
avg_cat_acc = {k: np.mean(v) if len(v) > 0 else None for k, v in cat_acc_totals.items()}

print(f"✅ Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_acc:.4f}")
print("Per-category Test Acc:", avg_cat_acc)

wandb.log({
    "test_loss": avg_test_loss,
    "test_accuracy": avg_test_acc,
    **{f"test_acc_{k}": v for k, v in avg_cat_acc.items()}
})

wandb.finish()
