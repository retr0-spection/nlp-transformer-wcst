import os
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from wcst import WCST

def prepare_wcst_datasets(
    batch_size=32,
    num_batches=1000,
    train_split=0.8,
    val_split=0.1,
    seed=42,
    save_path="./wcst_data",
    regenerate=False
):
    """
    Generate (or load) WCST datasets and return DataLoaders.

    Args:
        batch_size (int): Batch size for data generation and loaders.
        num_batches (int): Number of batches to generate from WCST.
        train_split (float): Fraction of total data used for training.
        val_split (float): Fraction of total data used for validation.
        seed (int): Random seed.
        save_path (str): Directory to save/load data.
        regenerate (bool): Force regeneration even if data already exists.

    Returns:
        train_loader, val_loader, test_loader (torch DataLoaders)
    """
    os.makedirs(save_path, exist_ok=True)
    np.random.seed(seed)

    # Check if data exists
    data_files = [f"{split}_src.pt" for split in ["train", "val", "test"]]
    files_exist = all(os.path.exists(os.path.join(save_path, f)) for f in data_files)

    if files_exist and not regenerate:
        print("📦 Loading saved WCST data...")
        train_src = torch.load(os.path.join(save_path, "train_src.pt"))
        train_tgt = torch.load(os.path.join(save_path, "train_tgt.pt"))
        val_src = torch.load(os.path.join(save_path, "val_src.pt"))
        val_tgt = torch.load(os.path.join(save_path, "val_tgt.pt"))
        test_src = torch.load(os.path.join(save_path, "test_src.pt"))
        test_tgt = torch.load(os.path.join(save_path, "test_tgt.pt"))
    else:
        print("🧠 Generating new WCST data...")
        wcst = WCST(batch_size)
        gen = wcst.gen_batch()

        all_src, all_tgt = [], []
        for i in range(num_batches):
            src, tgt = next(gen)
            all_src.append(src)
            all_tgt.append(tgt)

            # Optional: simulate context switches
            if (i + 1) % 30 == 0:
                wcst.context_switch()

        all_src = np.vstack(all_src)
        all_tgt = np.vstack(all_tgt)

        # Shuffle
        indices = np.arange(len(all_src))
        np.random.shuffle(indices)
        all_src, all_tgt = all_src[indices], all_tgt[indices]

        # Split
        n_train = int(len(all_src) * train_split)
        n_val = int(len(all_src) * val_split)
        train_src, train_tgt = all_src[:n_train], all_tgt[:n_train]
        val_src, val_tgt = all_src[n_train:n_train+n_val], all_tgt[n_train:n_train+n_val]
        test_src, test_tgt = all_src[n_train+n_val:], all_tgt[n_train+n_val:]

        # Convert to tensors
        train_src, train_tgt = torch.tensor(train_src, dtype=torch.long), torch.tensor(train_tgt, dtype=torch.long)
        val_src, val_tgt = torch.tensor(val_src, dtype=torch.long), torch.tensor(val_tgt, dtype=torch.long)
        test_src, test_tgt = torch.tensor(test_src, dtype=torch.long), torch.tensor(test_tgt, dtype=torch.long)

        # Save for reuse
        torch.save(train_src, os.path.join(save_path, "train_src.pt"))
        torch.save(train_tgt, os.path.join(save_path, "train_tgt.pt"))
        torch.save(val_src, os.path.join(save_path, "val_src.pt"))
        torch.save(val_tgt, os.path.join(save_path, "val_tgt.pt"))
        torch.save(test_src, os.path.join(save_path, "test_src.pt"))
        torch.save(test_tgt, os.path.join(save_path, "test_tgt.pt"))
        print(f"💾 Data saved to {save_path}")

    # Create DataLoaders
    train_dataset = TensorDataset(train_src, train_tgt)
    val_dataset = TensorDataset(val_src, val_tgt)
    test_dataset = TensorDataset(test_src, test_tgt)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
