import os
import sys
import numpy as np
import cv2

def create_synthetic_test_video(output_path: str, duration_sec: int = 5, fps: int = 30, width: int = 640, height: int = 480):
    """Creates a sample video file for testing ingestion and sampling functionality."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    total_frames = duration_sec * fps
    for i in range(total_frames):
        # Generate dynamic frame (moving color bar)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        color_val = int((i / total_frames) * 255)
        # Gradient background + frame counter text
        frame[:, :, 0] = (color_val + 50) % 256
        frame[:, :, 1] = (color_val + 100) % 256
        frame[:, :, 2] = (color_val + 150) % 256

        cv2.putText(frame, f"Frame {i+1}/{total_frames}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        writer.write(frame)

    writer.release()
    print(f"[+] Synthetic test video created at: {output_path}")

def run_phase1_and_phase2_verification():
    print("==================================================")
    print("   PHASE 1 & PHASE 2 VERIFICATION TEST SUITE     ")
    print("==================================================")

    # 1. Environment & Package Verification
    print("\n--- Phase 1: Environment Check ---")
    try:
        import torch
        import torchvision
        import cv2
        import numpy as np
        import pandas as pd
        import sklearn
        import flask

        print(f"  [OK] PyTorch Version      : {torch.__version__}")
        print(f"  [OK] Torchvision Version  : {torchvision.__version__}")
        print(f"  [OK] OpenCV Version       : {cv2.__version__}")
        print(f"  [OK] NumPy Version        : {np.__version__}")
        print(f"  [OK] Pandas Version       : {pd.__version__}")
        print(f"  [OK] Scikit-Learn Version : {sklearn.__version__}")
        print(f"  [OK] Flask Version        : {flask.__version__}")
        print(f"  [OK] CUDA Available       : {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  [OK] GPU Device Name      : {torch.cuda.get_device_name(0)}")
        else:
            print("  [i] CUDA is not available. System will run on CPU.")
    except Exception as e:
        print(f"  [FAIL] Environment import error: {e}")
        return False

    # 2. Video Reader Verification (Phase 2)
    print("\n--- Phase 2: Video Ingestion Check ---")
    from preprocessing.video_reader import get_video_metadata, validate_video_file

    sample_video_path = os.path.join("uploads", "sample_test.mp4")
    create_synthetic_test_video(sample_video_path, duration_sec=5, fps=30, width=640, height=480)

    try:
        metadata = get_video_metadata(sample_video_path)
        print("\nExtracted Video Metadata:")
        for key, val in metadata.items():
            print(f"  - {key:20s}: {val}")

        # Assertions
        assert metadata["fps"] == 30.0, f"Expected FPS 30.0, got {metadata['fps']}"
        assert metadata["total_frames"] == 150, f"Expected 150 frames, got {metadata['total_frames']}"
        assert metadata["resolution"] == "640x480", f"Expected 640x480 resolution, got {metadata['resolution']}"
        assert metadata["duration_seconds"] == 5.0, f"Expected 5.0s duration, got {metadata['duration_seconds']}"

        print("\n[OK] Phase 1 & Phase 2 successfully verified!")
        return True

    except Exception as e:
        print(f"\n[FAIL] Video Reader Verification failed: {e}")
        return False

if __name__ == "__main__":
    success = run_phase1_and_phase2_verification()
    sys.exit(0 if success else 1)
