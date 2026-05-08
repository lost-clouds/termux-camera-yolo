> **🤖 Vibecoding 产物** — 本仓库由 AI 辅助生成（vibecoding），大量代码通过自然语言描述自动产出。

# Camera YOLO Logger — Minimal Termux Edition

Single-command camera capture → YOLOv8 object detection → CSV / JSON / HTTP API.

## Quick Start
```bash
cd camera_yolo_logger
uv sync
uv run camera-yolo
```

Takes a photo with `termux-camera-photo`, runs YOLOv8 ONNX inference, prints result, appends to `camera_log.csv`, and keeps the latest photo at `Image/camera_yolo_temp.jpg`.

## Modes

| Mode | Command | Description |
|------|---------|-------------|
| One-shot text | `uv run camera-yolo` | Basic, backwards compatible |
| One-shot JSON | `uv run camera-yolo --json` | Structured output for AI agents |
| Class filtering | `uv run camera-yolo --json --classes person car` | Only detect specific objects |
| Motion pre-filter | `uv run camera-yolo --json --motion` | Skip YOLO when scene is static |
| Continuous monitor | `uv run camera-yolo --monitor --json --motion --archive` | Run indefinitely |
| HTTP Server | `uv run camera-yolo --server --server-port 5000` | REST API for AI agents |

## JSON Output

```json
{
  "version": "1.1.0",
  "timestamp": "2026-05-08T12:34:56Z",
  "capture": {"success": true, "path": "Image/camera_yolo_temp.jpg", "elapsed_ms": 423.5},
  "detection": {
    "success": true,
    "objects": [
      {"class": "person", "class_id": 0, "confidence": 0.92, "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 500}}
    ],
    "summary": "1个人",
    "elapsed_ms": 850.2
  }
}
```

## HTTP API

```bash
uv run camera-yolo --server --server-port 5000
```

Endpoints: `/detect`, `/status`, `/log`, `/config`, `/stream` (see [SKILL.md](SKILL.md)).

## Power Optimization

- **Motion pre-filter** (`--motion`): Skips YOLO inference when scene is static — saves ~90% power in monitoring mode
- **Adaptive interval**: 1s when objects detected, 5s normally, 30s when no motion
- **Class filtering** (`--classes`): Only run detection on target classes

## Real-time Preview

For 15+ FPS live view, install "IP Webcam" from Play Store, then:
```bash
export IP_WEBCAM_URL=http://192.168.1.100:8080/shot.jpg
uv run camera-yolo --server --capture-backend ipwebcam
```
Visit `http://localhost:5000/stream` for MJPEG live view.

## Model

- Default: `yolov8n.onnx` in project root
- Auto-download: set `MODEL_DOWNLOAD_URL` to a raw ONNX URL
- Custom models: any YOLOv8 ONNX export works

## Configuration

Priority: CLI flags > environment variables > TOML config file > defaults.

See [SKILL.md](SKILL.md) for full configuration reference.

## 友链

[linux.do](https://linux.do)

## License

MIT
