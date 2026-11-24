import os
import sys
import subprocess
import json
import hashlib
import time
from typing import Dict, Any
from gtts import gTTS

from config import TTS_TIMEOUT_SEC, TTS_ENGINE_PREFERENCE

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

def get_mp3_duration(mp3_path: str) -> float:
    """Uses ffprobe to measure exact audio duration in seconds of an MP3 file."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        mp3_path
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        return float(res.stdout.strip())
    except ValueError:
        return 0.0

def text_to_speech(
    text: str,
    language: str = "en",
    output_dir: str = "outputs/multimodal",
    base_name: str = "summary",
    force_recompute: bool = False
) -> Dict[str, Any]:
    """
    Converts translated text summary into audio voiceover MP3 file using cached gTTS / pyttsx3 fallback.
    Utilizes short timeouts, fast fallbacks, and audio caching.
    """
    os.makedirs(output_dir, exist_ok=True)
    lang = language.lower().strip()

    if not text or not text.strip():
        return {
            "success": False,
            "stage": "tts",
            "language": lang,
            "error": "Cannot synthesize audio from empty text.",
            "audio_file": None
        }

    # Hash caching based on text content & language
    text_hash = hashlib.md5(f"{text.strip()}_{lang}".encode('utf-8')).hexdigest()[:8]
    audio_path = os.path.join(output_dir, f"{base_name}_{text_hash}_narration.mp3")

    if not force_recompute and os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
        dur = get_mp3_duration(audio_path)
        print(f"[TTS] Using cached narration for language '{lang}' ({dur:.2f}s audio)")
        return {
            "success": True,
            "stage": "tts",
            "language": lang,
            "audio_file": audio_path,
            "audio_url": f"/outputs/multimodal/{os.path.basename(audio_path)}",
            "audio_duration_sec": dur,
            "cached": True
        }

    t0 = time.time()
    print(f"[TTS] Synthesizing speech for language '{lang}'...")
    gtts_success = False
    last_error = None

    t_gtts_start = time.time()
    # Attempt gTTS if network available with strict thread timeout
    if TTS_ENGINE_PREFERENCE in ["auto", "gtts"]:
        import threading
        def synthesize_gtts():
            nonlocal gtts_success, last_error
            try:
                tts = gTTS(text=text, lang=lang, tld='com', slow=False)
                tts.save(audio_path)
                if os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000:
                    dur_check = get_mp3_duration(audio_path)
                    if dur_check > 0.5:
                        gtts_success = True
            except Exception as e:
                last_error = e

        gtts_thread = threading.Thread(target=synthesize_gtts)
        gtts_thread.daemon = True
        gtts_thread.start()
        gtts_thread.join(timeout=6.0)

        if gtts_success:
            print(f"[TTS] gTTS: {time.time() - t_gtts_start:.2f} sec (Duration: {get_mp3_duration(audio_path):.2f}s)")
        else:
            print(f"[TTS Warning] gTTS synthesis failed/timed out ({time.time() - t_gtts_start:.2f}s). Immediately falling back to local engine...")

    # Fallback to local pyttsx3 engine
    t_fb_start = time.time()
    if not gtts_success:
        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty('rate', 175)
            voices = engine.getProperty('voices')
            for v in voices:
                if lang in str(v.languages).lower() or lang in v.name.lower():
                    engine.setProperty('voice', v.id)
                    break

            temp_wav = os.path.join(output_dir, f"{base_name}_{text_hash}_temp.wav")
            engine.save_to_file(text, temp_wav)
            engine.runAndWait()

            if os.path.exists(temp_wav) and os.path.getsize(temp_wav) > 0:
                cmd = ["ffmpeg", "-y", "-i", temp_wav, "-c:a", "libmp3lame", "-q:a", "2", audio_path]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if os.path.exists(temp_wav):
                    os.remove(temp_wav)
                fb_dur = get_mp3_duration(audio_path)
                if fb_dur > 0.5:
                    gtts_success = True
                    print(f"[TTS] Fallback: {time.time() - t_fb_start:.2f} sec (Duration: {fb_dur:.2f}s)")
                else:
                    print(f"[TTS Warning] Local pyttsx3 produced empty audio ({fb_dur:.2f}s) for '{lang}'.")
        except Exception as e2:
            print(f"[TTS Error] Local pyttsx3 fallback failed: {e2}")
            last_error = last_error or e2

    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0 or get_mp3_duration(audio_path) <= 0.5:
        return {
            "success": False,
            "stage": "tts",
            "language": lang,
            "error": f"TTS synthesis failed for '{lang}': {str(last_error or 'Audio duration 0s')}",
            "audio_file": None
        }

    audio_dur = get_mp3_duration(audio_path)
    synth_time = time.time() - t0
    print(f"[TTS Success] Synthesized audio in {synth_time:.2f}s. Duration: {audio_dur:.2f}s -> '{audio_path}'")

    return {
        "success": True,
        "stage": "tts",
        "language": lang,
        "audio_file": audio_path,
        "audio_url": f"/outputs/multimodal/{os.path.basename(audio_path)}",
        "audio_duration_sec": audio_dur,
        "synth_time_sec": round(synth_time, 2)
    }

if __name__ == "__main__":
    test_text = "Machine learning is a subset of artificial intelligence enabling automated learning from data."
    print("English TTS Result:", text_to_speech(test_text, "en"))
