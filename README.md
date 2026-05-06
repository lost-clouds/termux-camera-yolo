> **🤖 Vibecoding 产物** — 本仓库由 AI 辅助生成（vibecoding），大量代码通过自然语言描述自动产出。

# Camera YOLO Logger — Minimal Termux Edition

Single-command camera capture → YOLOv8 object detection → CSV log.

## Quick Start
```bash
cd camera_yolo_logger
uv sync
uv run camera-yolo
```
The command takes a new photo with `termux-camera-photo`, runs YOLOv8 ONNX inference, prints the result, appends it to `camera_log.csv`, and keeps the latest photo at `Image/camera_yolo_temp.jpg`.

## Model
- Default: `yolov8n.onnx` in project root.
- Auto‑download: set `MODEL_DOWNLOAD_URL` to a raw model URL (HuggingFace, ModelScope, etc.). The model is downloaded on first run.
- Input precision (float16/float32) is auto‑detected — just swap the model file.

## Configuration
Environment variables (all optional):

| Variable | Default | Description |
|---|---|---|
| `CAMERA_PHOTO_CMD` | `termux-camera-photo` | Camera command |
| `CAMERA_PHOTO_PATH` | `Image/camera_yolo_temp.jpg` | Photo path (latest kept) |
| `CAMERA_LOG_FILE` | `camera_log.csv` | Log file |
| `CAMERA_MODEL_PATH` | `yolov8n.onnx` | Model path |
| `MODEL_DOWNLOAD_URL` | (empty) | Auto‑download URL |
| `DETECT_CONF_THRESH` | `0.45` | Confidence threshold |
| `DETECT_IOU_THRESH` | `0.5` | NMS IoU threshold |

## For AI Agents
See `SKILL.md` for integration instructions.

## License
MIT
