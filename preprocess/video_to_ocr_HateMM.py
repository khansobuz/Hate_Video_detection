import os
import re
import cv2
import easyocr
import numpy as np
import pandas as pd
import pytesseract
from autocorrect import Speller
from Levenshtein import ratio
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import threading

spell = Speller(lang="en")
reader = easyocr.Reader(["en"], gpu=True)

VIDEO_TIMEOUT = 300  # seconds

# Extract frames from a video
def extract_frames(video_path, fps=1):
    frames = []
    video = cv2.VideoCapture(video_path)
    video_fps = video.get(cv2.CAP_PROP_FPS)
    interval = int(video_fps / fps) if video_fps > 0 else 1

    frame_count = 0
    while True:
        ret, frame = video.read()
        if not ret:
            break
        if frame_count % interval == 0:
            frames.append(frame)
        frame_count += 1

    video.release()
    return frames

# Compute similarity between frames
def frame_similarity(frame1, frame2):
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, full=True)
    return score

# Remove similar frames to reduce OCR
def remove_similar_frames(frames, threshold=0.95):
    if not frames:
        return []

    unique_frames = [frames[0]]
    for i in range(1, len(frames)):
        if frame_similarity(frames[i], frames[i - 1]) < threshold:
            unique_frames.append(frames[i])
    return unique_frames

# OCR frames using EasyOCR
def ocr_frames(frames):
    texts = []
    for frame in frames:
        text = reader.readtext(frame, detail=0)
        text = " ".join(text)
        if text:
            texts.append(text.strip())
    return texts

# Remove duplicate texts
def remove_duplicate_texts(texts, threshold=0.7):
    if not texts:
        return []

    unique_texts = [texts[0]]
    for i in range(1, len(texts)):
        if ratio(texts[i], texts[i - 1]) < threshold:
            unique_texts.append(texts[i])
    return unique_texts

# Clean and optionally autocorrect text
def clean_and_correct_text(text):
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# Extract text from a video
def extract_text_from_video(video_path, result_container):
    try:
        frames = extract_frames(video_path)
        unique_frames = remove_similar_frames(frames)
        texts = ocr_frames(unique_frames)
        cleaned_texts = [clean_and_correct_text(text) for text in texts]
        unique_texts = remove_duplicate_texts(cleaned_texts)
        result_container.append(unique_texts)
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        result_container.append([])

# Main processing
src_dir = "data/HateMM/videos"
dst_file = "data/HateMM/ocr.jsonl"

if not os.path.exists(dst_file):
    dst_df = pd.DataFrame(columns=["vid", "ocr"])
    dst_df.to_json(dst_file, orient="records", lines=True)
else:
    dst_df = pd.read_json(dst_file, lines=True)

cur_ids = dst_df["vid"].values if len(dst_df) > 0 else []

for file in tqdm(os.listdir(src_dir)):
    video_file = os.path.join(src_dir, file)
    video_id = file.replace(".mp4", "")

    if video_id in cur_ids:
        continue

    ocr_text = []
    result_container = []

    # Create thread for video processing
    thread = threading.Thread(target=extract_text_from_video, args=(video_file, result_container))
    thread.start()
    thread.join(timeout=VIDEO_TIMEOUT)

    if thread.is_alive():
        print(f"Skipping video due to timeout: {video_file}")
        continue

    if result_container:
        for text in result_container[0]:
            if len(text) > 3:
                ocr_text.append(text + "\n")

    tmp_df = pd.DataFrame([{"vid": video_id, "ocr": "".join(ocr_text)}])
    dst_df = pd.concat([dst_df, tmp_df], ignore_index=True)
    dst_df.to_json(dst_file, orient="records", lines=True, force_ascii=False)
