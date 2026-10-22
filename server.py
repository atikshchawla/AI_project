"""Local API for AI Project — Shot Classification."""

from __future__ import annotations

from pathlib import Path
import shutil
import time
import uuid

import cv2
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODELS, UPLOADS, OUTPUTS = ROOT / "models", ROOT / "uploads", ROOT / "outputs"
FRAME_COUNT, IMAGE_SIZE = 15, (224, 224)
CLASS_NAMES = [
    "Cover Drive", "Defensive", "Down The Wicket", "Flick", "Hook", "Late Cut",
    "Lofted Legside", "Lofted Offside", "Pull", "Reverse Sweep", "Scoop",
    "Square Cut", "Straight Drive", "Sweep", "Upper Cut",
]

app = FastAPI(title="AI Project — Shot Classification")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"], allow_methods=["*"], allow_headers=["*"])
app.mount("/outputs", StaticFiles(directory=OUTPUTS), name="outputs")
_models = None


def model_path(name: str) -> Path:
    path = MODELS / name
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(f"Model file is missing: {name}")
    return path


def get_models():
    global _models
    if _models is None:
        _models = (
            YOLO(model_path("Player_Type_Detection_Model.pt")),
            YOLO(model_path("Bat_Detection_Model.pt")),
            YOLO(model_path("Striker_Bat_Segmentation_Model.pt")),
            tf.keras.models.load_model(model_path("Efficientnetv2-s_GRU_128.keras"), compile=False),
        )
    return _models


def labelled_box(result, model: YOLO, target: str):
    for box in result.boxes:
        if model.names[int(box.cls[0].item())] == target:
            return box.xyxy[0].cpu().numpy().astype(int)
    return None


def crop_frame(frame: np.ndarray, striker_model: YOLO, bat_model: YOLO) -> np.ndarray:
    striker = labelled_box(striker_model(frame, verbose=False)[0], striker_model, "Striker")
    bat = labelled_box(bat_model(frame, verbose=False, max_det=1)[0], bat_model, "Cricket Bat")
    if striker is None:
        return cv2.resize(frame, IMAGE_SIZE)
    x1, y1, x2, y2 = striker
    if bat is not None:
        x1, y1, x2, y2 = min(x1, bat[0]), min(y1, bat[1]), max(x2, bat[2]), max(y2, bat[3])
    padding = int((x2 - x1) * 0.1)
    x1, x2, y1, y2 = max(0, x1 - padding), min(frame.shape[1], x2 + padding), max(0, y1), min(frame.shape[0], y2)
    return cv2.resize(frame[y1:y2, x1:x2], IMAGE_SIZE) if x2 > x1 and y2 > y1 else cv2.resize(frame, IMAGE_SIZE)


def process_video(path: Path):
    striker, bat, segmenter, classifier = get_models()
    capture = cv2.VideoCapture(str(path))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if count < 1:
        raise ValueError("No readable frames found in the video.")
    indices = set(np.linspace(0, count - 1, FRAME_COUNT).round().astype(int).tolist())
    frames, preview, index = [], [], 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index in indices:
            cropped = crop_frame(frame, striker, bat)
            segmented = segmenter(cropped, imgsz=256, max_det=2, verbose=False)[0].plot(boxes=False, probs=False, labels=False)
            frames.append(cv2.cvtColor(segmented, cv2.COLOR_BGR2RGB))
            preview.append(segmented)
        index += 1
    capture.release()
    if len(frames) != FRAME_COUNT:
        raise ValueError("The video ended before 15 frames could be sampled.")
    scores = classifier.predict(np.asarray(frames, dtype=np.uint8)[np.newaxis, ...], verbose=0)[0]
    ordered = np.argsort(scores)[::-1][:5]
    output_name = f"preprocessed-{int(time.time())}-{uuid.uuid4().hex[:8]}.mp4"
    writer = cv2.VideoWriter(str(OUTPUTS / output_name), cv2.VideoWriter_fourcc(*"mp4v"), 5, IMAGE_SIZE)
    for frame in preview:
        writer.write(frame)
    writer.release()
    return [{"label": CLASS_NAMES[i], "confidence": round(float(scores[i]), 6)} for i in ordered], output_name


@app.get("/api/health")
def health():
    return {"status": "ready"}


@app.post("/api/predict")
async def predict(video: UploadFile = File(...)):
    suffix = Path(video.filename or "video.mp4").suffix.lower()
    if suffix not in {".mp4", ".avi", ".mov", ".mkv"}:
        raise HTTPException(400, "Please upload an MP4, AVI, MOV, or MKV video.")
    input_path = UPLOADS / f"{uuid.uuid4().hex}{suffix}"
    with input_path.open("wb") as file:
        shutil.copyfileobj(video.file, file)
    try:
        ranking, preview = process_video(input_path)
        return {"prediction": ranking[0], "ranking": ranking, "previewUrl": f"http://127.0.0.1:8000/outputs/{preview}"}
    except Exception as error:
        raise HTTPException(500, str(error)) from error
    finally:
        input_path.unlink(missing_ok=True)
