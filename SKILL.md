# Camera YOLO Logger Skill

Termux 摄像头 → YOLOv8 物体检测 → CSV / JSON / Webhook / HTTP API。

## Quick Start
```bash
camera-yolo
```
拍照 → YOLO 识别 → 输出到 stdout + CSV 日志。保留最新照片在 `Image/camera_yolo_temp.jpg`。

## For AI Agents — How to Use This Tool

### 1. Installation (one-time)
```bash
# 克隆仓库
git clone <repo-url> termux-camera-yolo
cd termux-camera-yolo

# 创建 venv 并安装依赖
uv sync

# 安装全局命令 (创建 ~/.local/bin/camera-yolo wrapper)
bash install.sh
```

如果 Termux 通过 pkg 安装了 numpy/pillow/onnxruntime，先创建使用系统包的 venv:
```bash
uv venv --system-site-packages .venv && uv sync && bash install.sh
```

**install.sh 做了什么**：在 `~/.local/bin/camera-yolo` 创建 shell wrapper，自动 cd 到项目目录并执行 `uv run camera-yolo "$@"`。安装后 `camera-yolo` 全局可用，**无需在项目目录内运行**。

安装后若提示 command not found，执行 `source ~/.bashrc` 或 `source ~/.zshrc`。

### 2. One-shot Detection
```bash
# 基础 (向后兼容, text 输出)
camera-yolo

# JSON 输出 (推荐 AI Agent 使用)
camera-yolo --json

# 类过滤 — 仅检测指定类别
camera-yolo --json --classes person car dog

# 运动预过滤 — 静态场景跳过 YOLO, 大幅节省功耗
camera-yolo --json --motion

# 完整示例
camera-yolo --json --classes person --confidence 0.5 --motion
```

### 3. JSON Output Format
```json
{
  "version": "1.1.0",
  "timestamp": "2026-05-08T12:34:56Z",
  "capture": {
    "success": true,
    "path": "Image/camera_yolo_temp.jpg",
    "elapsed_ms": 423.5,
    "image_size": [1920, 1080]
  },
  "detection": {
    "success": true,
    "objects": [
      {
        "class": "person",
        "class_id": 0,
        "confidence": 0.92,
        "bbox": {"x1": 100, "y1": 200, "x2": 300, "y2": 500}
      }
    ],
    "summary": "1个人",
    "elapsed_ms": 850.2,
    "config": {
      "conf_threshold": 0.45,
      "iou_threshold": 0.5,
      "model": "yolov8n.onnx",
      "class_filter": ["person"]
    }
  }
}
```
**Exit code**: 0 on success, 1 on camera failure.

### 4. Continuous Monitoring Mode
```bash
# 基础监控 (每 5s 检测一次)
camera-yolo --monitor --json

# 完整功耗优化: 运动检测 + 自适应间隔 + 存档
camera-yolo --monitor --json --motion --archive --archive-dir captures

# 快速监控 + Webhook 通知
camera-yolo --monitor --json \
  --interval 2.0 --interval-min 1.0 \
  --webhook-url http://localhost:8080/notify \
  --webhook-trigger-classes person
```

自适应间隔行为:
- 检测到物体 → 最短间隔 (--interval-min, 默认 1s)
- 无物体 → 中等间隔 (--interval, 默认 5s)
- 运动过滤无运动 → 最长间隔 (--interval-max, 默认 30s) — 最大功耗节省

### 5. HTTP API Server (for AI Agents)
```bash
camera-yolo --server --server-port 5000
```

API 端点:
```bash
curl http://localhost:5000/detect              # GET/POST — 触发一次检测
curl http://localhost:5000/detect?format=summary  # 仅返回文本摘要
curl http://localhost:5000/status              # 健康检查 + 运行统计
curl http://localhost:5000/log?limit=50        # 最近 CSV 日志
curl http://localhost:5000/config              # GET 当前配置
curl -X POST http://localhost:5000/config \
  -H "Content-Type: application/json" \
  -d '{"confidence": 0.6, "classes": ["person"]}'  # POST 修改运行时配置
curl http://localhost:5000/stream              # MJPEG 实时预览 (需 IP Webcam)
```

### 6. Configuration

优先级: **CLI 参数 > 环境变量 > TOML 配置文件 > 默认值**

**环境变量** (向后兼容):
```bash
export CAMERA_PHOTO_CMD=termux-camera-photo
export CAMERA_PHOTO_PATH=Image/camera_yolo_temp.jpg
export CAMERA_LOG_FILE=camera_log.csv
export CAMERA_MODEL_PATH=yolov8n.onnx
export MODEL_DOWNLOAD_URL="https://huggingface.co/.../yolov8n.onnx"
export DETECT_CONF_THRESH=0.45
export DETECT_IOU_THRESH=0.5
export CLASS_FILTER=person,car,dog
export MOTION_ENABLED=true
export MOTION_THRESHOLD=0.05
export WEBBOOK_URL=http://localhost:8080/notify
export WEBBOOK_TRIGGER_CLASSES=person,car
export ARCHIVE_DIR=captures
export IP_WEBCAM_URL=http://192.168.1.100:8080/shot.jpg
export CAMERA_CAPTURE_BACKEND=ipwebcam
```

**TOML 配置文件** (`camera_yolo_logger.toml`):
```toml
[capture]
cmd = "termux-camera-photo"
path = "Image/camera_yolo_temp.jpg"
timeout = 30
retry = 3
backend = "termux"

[detection]
model = "yolov8n.onnx"
conf_threshold = 0.45
iou_threshold = 0.5
classes = ["person", "car"]

[motion]
enabled = true
threshold = 0.05
resize = 160

[monitor]
interval = 5.0
archive = true
archive_dir = "captures"

[webhook]
url = "http://localhost:8080/notify"
trigger_classes = ["person"]
```

### 7. Camera Requirements
- Termux with `termux-api` package installed
- Camera permission granted to Termux
- For real-time preview (15+ FPS): install IP Webcam app from Play Store, set `capture_backend="ipwebcam"` and `ip_webcam_url`

### 8. Model Management
- *Default*: looks for `yolov8n.onnx` in project root
- *Auto-download*: set `MODEL_DOWNLOAD_URL` to a raw ONNX model URL
- Input precision (float16/float32) is auto-detected — any YOLOv8 ONNX model works
- To use a custom model trained for specific objects, place it at `yolov8n.onnx` or set `CAMERA_MODEL_PATH`

### 9. Reading CSV Results
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
├── schemas.py    — BBox, Detection, DetectionResult, CaptureResult dataclass
├── config.py     — Settings + 分层加载 (CLI/env/TOML/defaults)
├── utils.py      — FileLock (fcntl), timed 装饰器
├── capture.py    — 拍照: termux-camera-photo / IP Webcam, 超时重试
├── detect.py     — YOLOv8 ONNX 推理, Detector class, 数字裁剪
├── motion.py     — 运动检测预过滤器 (帧差分 MSE)
├── monitor.py    — 连续监控循环, 自适应间隔, 存档
├── notify.py     — Webhook 通知 (检测到触发类 → POST JSON)
├── server.py     — Flask HTTP API (/detect, /status, /log, /stream)
├── main.py       — CLI 入口 (argparse, 模式分发)
└── __init__.py   — 版本号
```
