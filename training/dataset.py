import os
import torch
from torch.utils.data import Dataset
import numpy as np
from typing import List, Tuple, Dict, Any

class VideoSummarizationDataset(Dataset):
    """
    PyTorch Dataset for Video Summarization (TVSum / SumMe format).
    Each sample consists of:
    - features: Tensor of shape (T, 2048)
    - targets: Tensor of shape (T,) containing ground-truth human importance scores normalized in [0, 1]
    - video_name: string identifier
    """
    def __init__(self, feature_dir: str, annotation_dir: str = None, video_names: List[str] = None):
        self.feature_dir = feature_dir
        self.annotation_dir = annotation_dir
        
        if video_names is not None:
            self.video_names = video_names
        else:
            # Auto-discover feature files in directory
            self.video_names = [
                f.replace("_features.npy", "") for f in os.listdir(feature_dir)
                if f.endswith("_features.npy")
            ]

    def __len__(self) -> int:
        return len(self.video_names)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        name = self.video_names[idx]
        feat_path = os.path.join(self.feature_dir, f"{name}_features.npy")
        features_np = np.load(feat_path).astype(np.float32)

        # Load ground truth annotations if available, else synthetic targets for initial verification
        target_path = os.path.join(self.annotation_dir, f"{name}_targets.npy") if self.annotation_dir else None
        
        if target_path and os.path.exists(target_path):
            targets_np = np.load(target_path).astype(np.float32)
        else:
            # Generate synthetic target scores normalized [0, 1] for initial setup
            np.random.seed(abs(hash(name)) % (2**32))
            targets_np = np.random.uniform(0.1, 0.9, size=len(features_np)).astype(np.float32)

        features_tensor = torch.from_numpy(features_np)
        targets_tensor = torch.from_numpy(targets_np)

        return features_tensor, targets_tensor, name

def collate_video_batch(batch: List[Tuple[torch.Tensor, torch.Tensor, str]]):
    """Custom collate function for variable sequence length video batches with padding."""
    features_list, targets_list, names = zip(*batch)
    
    lengths = [f.size(0) for f in features_list]
    max_len = max(lengths)
    in_dim = features_list[0].size(1)

    padded_features = torch.zeros(len(batch), max_len, in_dim)
    padded_targets = torch.zeros(len(batch), max_len)
    padding_mask = torch.ones(len(batch), max_len, dtype=torch.bool)  # True indicates padded/masked element

    for i in range(len(batch)):
        l = lengths[i]
        padded_features[i, :l, :] = features_list[i]
        padded_targets[i, :l] = targets_list[i]
        padding_mask[i, :l] = False

    return padded_features, padded_targets, padding_mask, names
