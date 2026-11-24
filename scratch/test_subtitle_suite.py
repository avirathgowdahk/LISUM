import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from multimodal.subtitle_generator import (
    generate_srt_subtitles,
    wrap_text_into_lines,
    split_text_into_short_chunks,
    remap_timestamps_to_summary
)

print("==================================================")
print("   RUNNING AUTOMATED TEST SUITE FOR SUBTITLE SYSTEM   ")
print("==================================================")

# TEST 1: Long Paragraph Chunking (No giant text blocks)
long_text = (
    "Good morning everyone it's the fifth day here in Goa. The sun is shining and it looks like a fabulous day. "
    "India needs to survive the spinners and the fast bowlers on this tricky pitch while building a strong target."
)
print("\n--- TEST 1: Long Paragraph Chunking ---")
chunks = split_text_into_short_chunks(long_text, max_words=10)
print(f"[PASS] Split long paragraph into {len(chunks)} short chunks:")
for idx, ch in enumerate(chunks):
    word_count = len(ch.split())
    print(f"       Chunk {idx + 1} ({word_count} words): '{ch}'")
    assert word_count <= 10, f"Chunk has more than 10 words: '{ch}'"

# TEST 2: Line Wrapping (Max 2 lines, Max 42 Chars per line)
print("\n--- TEST 2: Line Wrapping & Balancing ---")
sample_sentence = "It is the fifth day here in Goa and the weather is looking absolutely fantastic."
wrapped = wrap_text_into_lines(sample_sentence, max_chars_per_line=42)
lines = wrapped.split("\n")
print(f"[PASS] Wrapped into {len(lines)} lines:")
for idx, line in enumerate(lines):
    print(f"       Line {idx + 1} ({len(line)} chars): '{line}'")
    assert len(line) <= 42, f"Line exceeds 42 characters: '{line}'"
assert len(lines) <= 2, f"Exceeded maximum 2 lines!"

# TEST 3: Timeline Remapping for Summary Clips
print("\n--- TEST 3: Timeline Remapping for Summary Clips ---")
raw_segments = [
    {"start": 10.0, "end": 14.0, "text": "Good morning everyone."},
    {"start": 52.0, "end": 56.0, "text": "It's the fifth day here in Goa."},
    {"start": 120.0, "end": 125.0, "text": "The sun is shining brightly."}
]
selected_clips = [
    (10.0, 15.0),   # 5s duration -> summary 0.0 to 5.0
    (50.0, 58.0),   # 8s duration -> summary 5.0 to 13.0
    (120.0, 130.0)  # 10s duration -> summary 13.0 to 23.0
]
remapped = remap_timestamps_to_summary(raw_segments, selected_clips)
print(f"[PASS] Remapped {len(remapped)} segments to summary timeline:")
for seg in remapped:
    print(f"       {seg['start']}s --> {seg['end']}s : '{seg['text']}'")
    assert seg['start'] >= 0, "Negative start timestamp!"
    assert seg['end'] > seg['start'], "Invalid end timestamp!"

# TEST 4: Full SRT Generation & Validation
print("\n--- TEST 4: Full SRT File Generation & Constraint Validation ---")
srt_path = "outputs/test_suite_captions.srt"
generated_path = generate_srt_subtitles(
    summary_text=long_text,
    total_audio_duration_sec=25.0,
    output_srt_path=srt_path
)

assert os.path.exists(generated_path), "SRT file was not created!"
with open(generated_path, "r", encoding="utf-8") as f:
    content = f.read()

print(f"[PASS] Generated SRT content:\n{content}")

# Parse SRT entries and validate constraints
blocks = content.strip().split("\n\n")
print(f"[PASS] Generated {len(blocks)} subtitle blocks. Validating rules:")

for b_idx, block in enumerate(blocks):
    lines = block.split("\n")
    assert len(lines) >= 3, f"Block {b_idx+1} is malformed!"
    cue_id = lines[0]
    timestamps = lines[1]
    text_lines = lines[2:]

    assert "-->" in timestamps, f"Missing timestamp arrow in block {cue_id}"
    assert len(text_lines) <= 2, f"Block {cue_id} exceeds 2 lines: {text_lines}"

    full_text = " ".join(text_lines)
    word_count = len(full_text.split())
    assert word_count <= 12, f"Block {cue_id} exceeds word limit: {word_count} words"

    for t_line in text_lines:
        assert len(t_line) <= 45, f"Line exceeds character limit ({len(t_line)} chars): '{t_line}'"

print("\n==================================================")
print("   ALL SUBTITLE TEST CASES EXECUTED SUCCESSFULLY!  ")
print("==================================================")
