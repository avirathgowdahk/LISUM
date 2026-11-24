import os
import shutil

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
outputs_dir = os.path.join(base_dir, "outputs")

print(f"Purging stale files in {outputs_dir}...")
if os.path.exists(outputs_dir):
    for root, dirs, files in os.walk(outputs_dir):
        for f in files:
            if f.endswith(".srt") or f.endswith(".json") or f.endswith(".mp4") or f.endswith(".txt"):
                file_path = os.path.join(root, f)
                try:
                    os.remove(file_path)
                    print(f"Deleted stale output: {f}")
                except Exception as e:
                    print(f"Could not delete {f}: {e}")

print("Purge complete!")
