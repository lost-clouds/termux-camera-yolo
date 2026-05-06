# Camera YOLO Logger Skill

Minimal Termux camera → YOLO detection → log to CSV, all in one command.

## Quick Start
```bash
uv run camera-yolo
```
That's it. Takes a new photo with `termux-camera-photo`, runs YOLOv8 object detection, prints the result to stdout, appends it to `camera_log.csv`, and keeps the latest photo at `Image/camera_yolo_temp.jpg`.

## For AI Agents — How to Use This Tool

### 1. Installation (one-time)
```bash
cd camera_yolo_logger
uv sync
```
If the Termux environment already provides numpy/pillow via pkg, use:
```bash
uv venv --system-site-packages .venv && uv sync
```

### 2. Run Detection
```bash
uv run camera-yolo
```
**Output**: `<timestamp>,<detection_string>` to stdout and CSV.  
**Exit code**: 0 on success, 1 on camera failure.

### 3. Camera Requirements
- Termux with `termux-api` package installed
- Camera permission granted to Termux

### 4. Model Management
*Default*: looks for `yolov8n.onnx` in project root.  
*Auto‑download*: set `MODEL_DOWNLOAD_URL` to a direct raw model URL; the tool downloads on first run.  
Examples:
```bash
# HuggingFace
export MODEL_DOWNLOAD_URL="https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.onnx"

# ModelScope (replace with actual raw link)
export MODEL_DOWNLOAD_URL="https://modelscope.cn/api/v1/models/.../resolve/master/yolov8n.onnx"
```
The tool adapts to the model's input precision (float16/float32) automatically.

### 5. Configuration (environment variables)
| Variable | Default | Description |
|---|---|---|
| `CAMERA_PHOTO_CMD` | `termux-camera-photo` | Camera command |
| `CAMERA_PHOTO_PATH` | `Image/camera_yolo_temp.jpg` | Photo path (latest kept) |
| `CAMERA_LOG_FILE` | `camera_log.csv` | CSV log file |
| `CAMERA_MODEL_PATH` | `yolov8n.onnx` | ONNX model path |
| `MODEL_DOWNLOAD_URL` | (empty) | Auto-download URL |
| `DETECT_CONF_THRESH` | `0.45` | Confidence threshold |
| `DETECT_IOU_THRESH` | `0.5` | NMS IoU threshold |

### 6. Reading Results
```bash
uv run python -c "
import csv
with open('camera_log.csv') as f:
    for row in csv.DictReader(f):
        print(row['timestamp'], '—', row['detected'])
"
```

## Architecture
```
camera_yolo_logger/
├── capture.py   — Termux camera capture
├── detect.py    — YOLO ONNX inference + model download
├── main.py      — CLI orchestration
└── __init__.py  — version
```
All modules ≤130 lines. No extra dependencies beyond numpy, pillow, onnxruntime.
