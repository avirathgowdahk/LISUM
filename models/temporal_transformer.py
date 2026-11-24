import os
import torch
import torch.nn as nn
import numpy as np
import math
from typing import List, Tuple, Dict, Any, Optional

from config import (
    DEVICE, CHECKPOINT_PATH, BEST_MODEL_PATH,
    IN_DIM, D_MODEL, NHEAD, NUM_LAYERS, DIM_FEEDFORWARD, DROPOUT
)

# Global Singleton Model Storage
_GLOBAL_TRANSFORMER_MODEL = None
_GLOBAL_TRANSFORMER_LOADED = None

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding to supply temporal order information."""
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class TemporalTransformerEncoder(nn.Module):
    """
    Temporal Transformer Encoder for Video Summarization.
    Takes sequence of visual features (ResNet-50 2048-dim) and models global temporal
    context to output importance scores in range [0, 1] for each frame.
    """
    def __init__(
        self,
        in_dim: int = IN_DIM,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = NUM_LAYERS,
        dim_feedforward: int = DIM_FEEDFORWARD,
        dropout: float = DROPOUT
    ):
        super().__init__()
        self.in_dim = in_dim
        self.d_model = d_model

        self.projection = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.score_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.projection(x)
        h = self.pos_encoder(h)
        enc_out = self.transformer_encoder(h, src_key_padding_mask=mask)
        scores = self.score_head(enc_out).squeeze(-1)
        return scores

def load_trained_model(
    checkpoint_path: Optional[str] = None,
    device: Optional[str] = None
) -> Tuple[TemporalTransformerEncoder, bool]:
    """
    Instantiates TemporalTransformerEncoder and loads trained state_dict.
    Uses singleton pattern to avoid re-loading weights on every job.
    """
    global _GLOBAL_TRANSFORMER_MODEL, _GLOBAL_TRANSFORMER_LOADED

    target_device = torch.device(device if device else (DEVICE if torch.cuda.is_available() else "cpu"))

    if _GLOBAL_TRANSFORMER_MODEL is not None:
        print("[TRANSFORMER] Temporal Transformer model loaded once and reused")
        return _GLOBAL_TRANSFORMER_MODEL, _GLOBAL_TRANSFORMER_LOADED

    model = TemporalTransformerEncoder(
        in_dim=IN_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        dropout=DROPOUT
    ).to(target_device)

    ckpt_candidates = [checkpoint_path, CHECKPOINT_PATH, BEST_MODEL_PATH]
    loaded = False

    for path in ckpt_candidates:
        if path and os.path.exists(path):
            try:
                state_dict = torch.load(path, map_location=target_device)
                model.load_state_dict(state_dict)
                model.eval()
                print(f"[TRANSFORMER] Loaded trained Temporal Transformer checkpoint from: {path}")
                loaded = True
                break
            except Exception as e:
                print(f"[TRANSFORMER Warning] Failed loading '{path}': {e}")

    if not loaded:
        print("[DEMO / UNTRAINED FALLBACK MODE] No trained checkpoint found on disk. Operating in demo mode with initialized Temporal Transformer weights.")
        model.eval()

    _GLOBAL_TRANSFORMER_MODEL = model
    _GLOBAL_TRANSFORMER_LOADED = loaded

    return model, loaded

def compute_shot_importance_scores(
    frame_scores: np.ndarray,
    timestamps: np.ndarray,
    shots: List[Tuple[float, float]]
) -> List[float]:
    """
    Aggregates frame-level importance scores into shot-level importance scores.
    """
    shot_scores = []
    for (start_t, end_t) in shots:
        mask = (timestamps >= start_t) & (timestamps <= end_t)
        shot_frame_scores = frame_scores[mask]
        
        if len(shot_frame_scores) > 0:
            shot_score = float(np.mean(shot_frame_scores))
        else:
            shot_score = 0.5
        
        shot_scores.append(round(shot_score, 4))
    
    return shot_scores

if __name__ == "__main__":
    model, is_loaded = load_trained_model()
    print("Model created/loaded successfully. Is Loaded:", is_loaded)
