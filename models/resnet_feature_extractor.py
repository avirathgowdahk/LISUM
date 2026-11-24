import os
import cv2
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import hashlib
from typing import List, Tuple, Dict, Any, Optional

from config import DEVICE, FEATURE_DIR, RESNET_BATCH_SIZE, FRAME_SAMPLE_FPS
from preprocessing.video_reader import get_video_metadata
from preprocessing.shot_detector import detect_shots

# Singleton ResNet Model Instance Storage
_GLOBAL_RESNET_EXTRACTOR = None

def get_resnet_feature_extractor(device: str = None) -> "ResNet50FeatureExtractor":
    """Lazy-loads and returns a singleton ResNet50FeatureExtractor model instance."""
    global _GLOBAL_RESNET_EXTRACTOR
    if _GLOBAL_RESNET_EXTRACTOR is None:
        _GLOBAL_RESNET_EXTRACTOR = ResNet50FeatureExtractor(device=device)
    else:
        print("[RESNET] ResNet-50 model loaded once and reused")
    return _GLOBAL_RESNET_EXTRACTOR

class ResNet50FeatureExtractor:
    """
    Pretrained ResNet-50 visual feature extractor.
    Strips classification head to extract 2048-dim feature representations per frame.
    Loaded ONCE globally to avoid re-downloading/re-instantiating model.
    """
    def __init__(self, device: str = None):
        self.device = torch.device(device if device else (DEVICE if torch.cuda.is_available() else "cpu"))
        if not torch.cuda.is_available():
            print(f"[RESNET] Initializing CPU-optimized ResNet-18 Feature Extractor on {self.device}...")
            weights = models.ResNet18_Weights.DEFAULT
            full_model = models.resnet18(weights=weights)
            self.feature_extractor = nn.Sequential(
                *list(full_model.children())[:-1],
                nn.Flatten(),
                nn.Linear(512, 2048)
            )
        else:
            print(f"[RESNET] Initializing ResNet-50 Feature Extractor on {self.device}...")
            weights = models.ResNet50_Weights.DEFAULT
            full_model = models.resnet50(weights=weights)
            self.feature_extractor = nn.Sequential(*list(full_model.children())[:-1])

        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        self.norm_transform = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        if not torch.cuda.is_available():
            torch.set_num_threads(min(8, os.cpu_count() or 4))

    @torch.inference_mode()
    def extract_batch(self, bgr_frames: List[np.ndarray]) -> np.ndarray:
        """
        Extracts 2048-dim feature vectors for a batch of BGR frames using OpenCV C++ resizing.
        """
        if not bgr_frames:
            return np.empty((0, 2048), dtype=np.float32)

        rgb_tensors = []
        for frame in bgr_frames:
            resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            t = self.norm_transform(t)
            rgb_tensors.append(t)

        batch_tensor = torch.stack(rgb_tensors).to(self.device)

        features = self.feature_extractor(batch_tensor)
        if features.ndim > 2:
            features = features.squeeze(-1).squeeze(-1)
        features = nn.functional.normalize(features, p=2, dim=1)

        return features.cpu().numpy().astype(np.float32)

    def extract_and_cache(
        self,
        video_path: str,
        target_fps: float = None,
        batch_size: int = None,
        force_recompute: bool = False,
        keyframes_per_shot: int = 1
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extracts deep visual features per shot keyframe with disk caching.
        """
        target_fps = target_fps or FRAME_SAMPLE_FPS or 2.0
        batch_size = batch_size or RESNET_BATCH_SIZE or 32

        abs_path = os.path.abspath(video_path)
        file_size = os.path.getsize(video_path) if os.path.exists(video_path) else 0
        file_base = os.path.splitext(os.path.basename(video_path))[0]
        path_hash = hashlib.md5(f"{abs_path}_{file_size}_{target_fps}_{keyframes_per_shot}".encode('utf-8')).hexdigest()[:8]
        base_name = f"{file_base}_{path_hash}"

        feat_path = os.path.join(FEATURE_DIR, f"{base_name}_features.npy")
        ts_path = os.path.join(FEATURE_DIR, f"{base_name}_timestamps.npy")
        shot_path = os.path.join(FEATURE_DIR, f"{base_name}_shots.npy")

        if not force_recompute and os.path.exists(feat_path) and os.path.exists(ts_path) and os.path.exists(shot_path):
            print(f"[RESNET] Loading cached features for '{base_name}' from disk...")
            return np.load(feat_path), np.load(ts_path), np.load(shot_path)

        from preprocessing.shot_detector import detect_shots_and_keyframes
        shots_list, kf_frames, kf_timestamps, kf_shot_indices = detect_shots_and_keyframes(video_path, target_fps=1.0)
        shots = np.array(shots_list, dtype=np.float32)

        print(f"[RESNET] Batched feature extraction on {len(kf_frames)} keyframes (Batch Size: {batch_size}, Device: {self.device})...")
        kf_feature_batches = []
        for i in range(0, len(kf_frames), batch_size):
            batch = kf_frames[i : i + batch_size]
            try:
                feats = self.extract_batch(batch)
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and batch_size > 4:
                    print(f"[!] GPU CUDA OOM. Reducing batch size from {batch_size} to {batch_size // 2}...")
                    batch_size = max(4, batch_size // 2)
                    feats = self.extract_batch(batch)
                else:
                    raise e
            kf_feature_batches.append(feats)

        kf_features = np.vstack(kf_feature_batches) if kf_feature_batches else np.empty((0, 2048), dtype=np.float32)

        num_shots = len(shots_list)
        shot_vectors = np.zeros((num_shots, 2048), dtype=np.float32)
        shot_counts = np.zeros(num_shots, dtype=np.int32)

        for feat, s_idx in zip(kf_features, kf_shot_indices):
            if 0 <= s_idx < num_shots:
                shot_vectors[s_idx] += feat
                shot_counts[s_idx] += 1

        for s_idx in range(num_shots):
            if shot_counts[s_idx] > 0:
                shot_vectors[s_idx] /= shot_counts[s_idx]
                norm = np.linalg.norm(shot_vectors[s_idx])
                if norm > 1e-6:
                    shot_vectors[s_idx] /= norm

        meta = get_video_metadata(video_path)
        dur = meta["duration_seconds"]
        num_timestamps = max(1, int(round(dur * target_fps)))
        timestamps_list = [round(i / target_fps, 3) for i in range(num_timestamps)]
        timestamps = np.array(timestamps_list, dtype=np.float32)

        features = np.zeros((num_timestamps, 2048), dtype=np.float32)
        for t_idx, ts_sec in enumerate(timestamps_list):
            matched_shot = 0
            for s_idx, (s_start, s_end) in enumerate(shots_list):
                if s_start <= ts_sec <= s_end:
                    matched_shot = s_idx
                    break
            features[t_idx] = shot_vectors[matched_shot]

        np.save(feat_path, features)
        np.save(ts_path, timestamps)
        np.save(shot_path, shots)
        print(f"[RESNET] Extracted & cached {len(features)} feature vectors @ {target_fps} FPS.")

        return features, timestamps, shots

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        extractor = get_resnet_feature_extractor()
        feats, ts, shots = extractor.extract_and_cache(sys.argv[1])
        print(f"Extracted {feats.shape[0]} vectors of dim {feats.shape[1]}")
