import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Directory paths
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
FEATURE_DIR = os.path.join(BASE_DIR, "features")

# Ensure required directories exist
for directory in [UPLOAD_DIR, OUTPUT_DIR, MODEL_DIR, FEATURE_DIR]:
    os.makedirs(directory, exist_ok=True)

# Video Processing Configurations
ALLOWED_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
DEFAULT_SAMPLE_FPS = 2.0  # Downsample from native FPS to ~2 FPS for feature extraction
TARGET_RATIOS = [0.10, 0.15, 0.20]  # Summary duration options (10%, 15%, 20%)
MAX_VIDEO_SIZE_MB = 2000.0  # 2 GB maximum file size
MAX_LECTURE_DURATION_SECONDS = 3 * 3600  # 3 hours (10,800 seconds) max duration for single lecture

# Hardware Device Configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Multimodal & Performance Configurations
PROCESSING_MODE = "FAST"  # Options: "FAST", "BALANCED", "QUALITY"
WHISPER_MODEL = os.getenv("WHISPER_MODEL_SIZE", os.getenv("WHISPER_MODEL", "base"))
WHISPER_MODEL_SIZE = WHISPER_MODEL  # Backward compatibility
WHISPER_BACKEND = os.getenv("WHISPER_BACKEND", "faster-whisper")  # Options: "faster-whisper", "whisper"
WHISPER_COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"  # Options: "int8", "float16", "float32"
FRAME_SAMPLE_FPS = 2.0
RESNET_BATCH_SIZE = 32
DEFAULT_CNN_BACKBONE = "resnet18"  # Options: 'resnet50', 'resnet18', 'mobilenet_v3'
TTS_TIMEOUT_SEC = 30.0
TTS_ENGINE_PREFERENCE = "auto"  # Options: 'auto', 'gtts', 'pyttsx3'
AUDIO_MODE = "original"  # Options: 'original' (Lecturer Voice), 'ai_narration', 'dubbing'

# Shot Detection Configurations
MIN_SHOT_DURATION = 2.0
MAX_SHOT_DURATION = 15.0
THRESHOLD_FACTOR = 1.5
FALLBACK_WINDOW_SEC = 5.0

# Diversity Penalty Configuration
DIVERSITY_LAMBDA = 0.3

# Temporal Transformer Model Configurations
CHECKPOINT_PATH = os.path.join(MODEL_DIR, "transformer_checkpoint.pth")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_transformer.pth")
IN_DIM = 2048
D_MODEL = 256
NHEAD = 8
NUM_LAYERS = 3
DIM_FEEDFORWARD = 512
DROPOUT = 0.1

# Supported Language Configurations
SUPPORTED_LANGUAGES = {
    "en": "English",
    "hi": "Hindi (हिंदी)",
    "kn": "Kannada (ಕನ್ನಡ)",
    "ta": "Tamil (தமிழ்)",
    "te": "Telugu (తెలుగు)",
    "ml": "Malayalam (മലയാളം)"
}

