import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

from config import DEVICE, MODEL_DIR
from models.temporal_transformer import TemporalTransformerEncoder
from training.dataset import VideoSummarizationDataset, collate_video_batch

def train_transformer(
    dataset: VideoSummarizationDataset,
    epochs: int = 15,
    lr: float = 1e-4,
    batch_size: int = 4,
    checkpoint_path: str = None
):
    """
    Training loop for Temporal Transformer Encoder.
    Optimizes MSE / Huber loss between predicted importance scores and ground truth annotations.
    """
    if checkpoint_path is None:
        checkpoint_path = os.path.join(MODEL_DIR, "transformer_checkpoint.pth")

    device = torch.device(DEVICE)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_video_batch)

    model = TemporalTransformerEncoder(in_dim=2048, d_model=256, nhead=8, num_layers=3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.HuberLoss(delta=0.1)

    print(f"\n==================================================")
    print(f"   STARTING TRANSFORMER TRAINING LOOP ({epochs} EPOCHS)")
    print(f"==================================================")
    print(f"Device: {device} | Dataset size: {len(dataset)} videos")

    model.train()
    best_loss = float('inf')

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        total_batches = 0

        for padded_feats, padded_targets, padding_mask, names in dataloader:
            padded_feats = padded_feats.to(device)
            padded_targets = padded_targets.to(device)
            padding_mask = padding_mask.to(device)

            optimizer.zero_grad()
            predictions = model(padded_feats, mask=padding_mask)  # (B, T)

            # Mask out loss on padded frame indices
            valid_mask = ~padding_mask
            loss = criterion(predictions[valid_mask], padded_targets[valid_mask])

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            total_batches += 1

        avg_loss = epoch_loss / max(1, total_batches)
        print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Training Huber Loss: {avg_loss:.6f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), checkpoint_path)

    print(f"\n[OK] Training complete! Best model saved to: {checkpoint_path}")
    return checkpoint_path

if __name__ == "__main__":
    from config import FEATURE_DIR
    ds = VideoSummarizationDataset(feature_dir=FEATURE_DIR)
    if len(ds) > 0:
        train_transformer(ds, epochs=5)
    else:
        print("No cached feature vectors found in features/ to train. Run feature extraction first.")
