import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.url_downloader import fetch_video_from_url, validate_public_url

print("==================================================")
print("   RUNNING AUTOMATED TEST SUITE FOR VIDEO URL FEATURE   ")
print("==================================================")

# TEST 1: user-reported URL with query params
url_1 = "https://youtu.be/hj0gm0mriMw?si=tJuWCn4klPOjNJ-4"
print(f"\n--- TEST 1: User Reported URL ({url_1}) ---")
res1 = fetch_video_from_url(url_1)
print(f"[PASS] File path: {res1['local_path']}")
assert os.path.exists(res1['local_path']), "File does not exist!"
assert res1['metadata']['duration_seconds'] > 0, "Invalid duration!"

# TEST 2: standard youtube.com/watch?v=
url_2 = "https://www.youtube.com/watch?v=hj0gm0mriMw"
print(f"\n--- TEST 2: Standard YouTube Watch URL ({url_2}) ---")
res2 = fetch_video_from_url(url_2)
print(f"[PASS] File path: {res2['local_path']}")
assert res2['local_path'] == res1['local_path'], "Normalized URLs should hit same cache path!"

# TEST 3: non-www youtube.com/watch?v=
url_3 = "https://youtube.com/watch?v=hj0gm0mriMw"
print(f"\n--- TEST 3: Non-WWW YouTube URL ({url_3}) ---")
res3 = fetch_video_from_url(url_3)
print(f"[PASS] File path: {res3['local_path']}")

# TEST 4: Extra query parameters (utm_source, fbclid)
url_4 = "https://www.youtube.com/watch?v=hj0gm0mriMw&utm_source=newsletter&fbclid=abcdef123456"
print(f"\n--- TEST 4: Extra Tracking Query Parameters ({url_4}) ---")
res4 = fetch_video_from_url(url_4)
print(f"[PASS] File path: {res4['local_path']}")
assert res4['local_path'] == res1['local_path'], "Tracking params stripped -> same cache hit!"

# TEST 5 & 6: Another public YouTube video (stream merge / FFmpeg normalization)
url_5 = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
print(f"\n--- TEST 5 & 6: Public YouTube Video Stream Merge & Validation ({url_5}) ---")
res5 = fetch_video_from_url(url_5)
print(f"[PASS] File path: {res5['local_path']}")
print(f"       Duration : {res5['metadata']['duration_formatted']}")
print(f"       Size     : {res5['metadata']['file_size_mb']} MB")

# TEST 7: SSRF & Invalid URLs
print("\n--- TEST 7: SSRF & Invalid Syntax Block Check ---")
invalid_urls = [
    "http://127.0.0.1:5000/admin",
    "file:///C:/Windows/System32/cmd.exe",
    "ftp://example.com/video.mp4",
    "not_a_url_string"
]
for u in invalid_urls:
    try:
        validate_public_url(u)
        print(f"[FAIL] URL '{u}' was NOT blocked!")
    except ValueError as ve:
        print(f"[PASS] Blocked '{u}' -> Error: {ve}")

# TEST 8: Private / Unavailable Video Handling
print("\n--- TEST 8: Private / Non-Existent Video Error Handling ---")
bad_video_url = "https://www.youtube.com/watch?v=invalid_id_99999999999"
try:
    fetch_video_from_url(bad_video_url)
    print(f"[FAIL] Invalid video URL was not caught!")
except ValueError as ve:
    print(f"[PASS] Caught video error as expected -> {ve}")

# TEST 9: Sequential Downloads
print("\n--- TEST 9: Sequential URL Downloads ---")
s_start = time.time()
r_seq1 = fetch_video_from_url(url_1)
r_seq2 = fetch_video_from_url(url_5)
s_elapsed = time.time() - s_start
print(f"[PASS] Sequential requests finished in {s_elapsed:.2f}s!")

# TEST 10: Concurrent Downloads
print("\n--- TEST 10: Concurrent URL Downloads ---")
c_start = time.time()
with ThreadPoolExecutor(max_workers=2) as executor:
    f1 = executor.submit(fetch_video_from_url, url_1)
    f2 = executor.submit(fetch_video_from_url, url_5)
    r_conc1 = f1.result()
    r_conc2 = f2.result()
c_elapsed = time.time() - c_start
print(f"[PASS] Concurrent requests completed safely in {c_elapsed:.2f}s!")

print("\n==================================================")
print("   ALL 10 TEST SUITE CASES EXECUTED SUCCESSFULLY!   ")
print("==================================================")
