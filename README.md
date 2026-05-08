> **🤖 Vibecoding 产物** — 本仓库由 AI 辅助生成（vibecoding），大量代码通过自然语言描述自动产出。

# Camera YOLO Logger — Minimal Termux Edition

单命令 Termux 摄像头拍照 → YOLOv8 ONNX 物体检测 → CSV / JSON / Webhook / HTTP API。

专为 Android + Termux 环境设计，供 AI Agent (Claude Code / Codex / ZeroClaw) 通过 SKILL.md 调用，也可独立使用。

## Quick Start

```bash
cd termux-camera-yolo
uv sync
bash install.sh     # 创建全局命令 ~/.local/bin/camera-yolo
camera-yolo
```

拍照 → YOLO 检测 → 输出结果到 stdout + `camera_log.csv`。最新照片保留在 `Image/camera_yolo_temp.jpg`。

如果你的 Termux 已通过 `pkg` 安装了 numpy/pillow/onnxruntime：
```bash
uv venv --system-site-packages .venv && uv sync && bash install.sh
```

**install.sh** 在 `~/.local/bin/camera-yolo` 创建 shell wrapper，此后 `camera-yolo` 在任何目录下都可直接调用，无需 `uv run`。

## Modes

| 模式 | 命令 | 说明 |
|------|------|------|
| 一次性 text | `camera-yolo` | 基础用法 (向后兼容) |
| 一次性 JSON | `camera-yolo --json` | 结构化输出，供 AI Agent 使用 |
| 类过滤 | `camera-yolo --json --classes person car dog` | 仅检测指定类别 |
| 运动预过滤 | `camera-yolo --json --motion` | 静态场景跳过 YOLO 推理，降低功耗 |
| 连续监控 | `camera-yolo --monitor --json --motion --archive` | 持续运行，检测到物体自动存档 |
| HTTP 服务器 | `camera-yolo --server --server-port 5000` | 提供 REST API 供外部调用 |

## CLI Flags

### 通用选项
| 标志 | 默认值 | 说明 |
|------|--------|------|
| `--config`, `-c` | — | TOML 配置文件路径 |
| `--setup` | — | 交互式配置向导（首次使用推荐） |

### 拍照选项
| 标志 | 默认值 | 说明 |
|------|--------|------|
| `--photo-cmd` | `termux-camera-photo` | 拍照命令 |
| `--photo-path` | `Image/camera_yolo_temp.jpg` | 照片保存路径 |
| `--capture-timeout` | `30` | 拍照超时秒数 |
| `--capture-retry` | `3` | 拍照失败重试次数 |
| `--capture-backend` | `termux` | 拍照后端：`termux` / `ipwebcam` / `url` |
| `--ip-webcam-url` | — | IP Webcam 抓帧 URL |

### 检测选项
| 标志 | 默认值 | 说明 |
|------|--------|------|
| `--model` | `yolov8n.onnx` | ONNX 模型路径 |
| `--model-download-url` | — | 自动下载模型的 URL |
| `--confidence`, `--conf` | `0.45` | 置信度阈值 |
| `--iou` | `0.5` | NMS IoU 阈值 |
| `--classes` | — | 类过滤 (如 `person car dog`) |

### 输出选项
| 标志 | 默认值 | 说明 |
|------|--------|------|
| `--json` | `false` | 启用 JSON 格式输出 |
| `--log-file` | `camera_log.csv` | CSV 日志路径 |
| `--csv-max-records` | `1000` | CSV 最大记录条数，超出后自动裁剪 |
| `--verbose`, `-v` | `false` | 详细输出到 stderr |

### 运动检测 (功耗优化)
| 标志 | 默认值 | 说明 |
|------|--------|------|
| `--motion` | `false` | 启用运动检测预过滤 |
| `--motion-threshold` | `0.05` | 运动检测灵敏度 (MSE，越小越敏感) |
| `--motion-resize` | `160` | 灰度缩放尺寸 (越小越快) |

### 连续监控
| 标志 | 默认值 | 说明 |
|------|--------|------|
| `--monitor` | `false` | 启用连续监控模式 |
| `--interval` | `5.0` | 监控间隔秒数 |
| `--interval-min` | `1.0` | 检测到物体时的最小间隔 |
| `--interval-max` | `30.0` | 无运动时的最大间隔 |
| `--archive` | `false` | 检测到物体时存档照片 |
| `--archive-dir` | `archive` | 存档目录 |

### Webhook 通知
| 标志 | 说明 |
|------|------|
| `--webhook-url` | Webhook 接收 URL |
| `--webhook-trigger-classes` | 触发通知的目标类别 |
| `--webhook-cooldown` | 冷却时间 (秒，默认 10) |

### HTTP 服务器
| 标志 | 默认值 | 说明 |
|------|--------|------|
| `--server` | `false` | 启动 HTTP API 服务器 |
| `--server-host` | `127.0.0.1` | 监听地址 |
| `--server-port` | `5000` | 监听端口 |

## JSON Output

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
      "class_filter": null,
      "motion_detection": false
    }
  }
}
```

## HTTP API

```bash
camera-yolo --server --server-port 5000
```

| 方法 | 端点 | 说明 |
|------|------|------|
| GET/POST | `/detect` | 触发一次检测，返回 JSON |
| GET | `/detect?format=summary` | 仅返回文本摘要 |
| GET | `/status` | 健康检查 + 运行统计 |
| GET | `/log?limit=50` | 最近 N 条 CSV 日志 |
| GET | `/config` | 查看当前配置 |
| POST | `/config` | 运行时修改配置 (`{"confidence": 0.6, "classes": ["person"]}`) |
| GET | `/stream` | MJPEG 实时预览 (需配置 IP Webcam) |

## 实时预览

安装 "IP Webcam" App (Play Store)，启动后配置：
```bash
camera-yolo --server --capture-backend ipwebcam --ip-webcam-url http://192.168.1.100:8080/shot.jpg
```
浏览器访问 `http://localhost:5000/stream` 查看 MJPEG 实时画面。

## 功耗优化

- **运动预过滤** (`--motion`)：帧差分判断场景变化，静态时跳过 YOLO 推理 — 监控模式下节省 ~90% 功耗
- **自适应间隔**：检测到物体→1s，正常→5s，无运动→30s
- **类过滤** (`--classes`)：使用自定义模型仅检测目标类别，减少无关计算

## 模型

- 默认使用 `yolov8n.onnx` (项目根目录)
- 支持自动下载：设置 `MODEL_DOWNLOAD_URL` 或 `--model-download-url`
- 输入精度 (float16/float32) 自动检测 — 任何 YOLOv8 ONNX 导出模型均可使用
- 需要自定义检测内容时，训练自己的 YOLOv8 模型导出 ONNX 即可

## 首次运行

首次执行 `camera-yolo` 时，会自动在项目目录生成默认配置文件 `camera_yolo_logger.toml`。照片默认保存在项目目录下的 `Image/` 文件夹。

如需自定义设置，可运行交互式配置向导：
```bash
camera-yolo --setup
```

向导会引导你设置：照片保存路径、CSV 最大记录数（默认 1000，超出自动裁剪）、检测类别、置信度阈值。

## 配置文件

除 CLI 标志和环境变量外，还支持 TOML 配置文件 (`camera_yolo_logger.toml`)：

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

[output]
log_file = "camera_log.csv"
max_records = 1000

[motion]
enabled = true
threshold = 0.05

[monitor]
interval = 5.0
archive = true
archive_dir = "captures"

[webhook]
url = "http://localhost:8080/notify"
trigger_classes = ["person"]

[server]
host = "0.0.0.0"
port = 5000
```

配置文件搜索路径：`./camera_yolo_logger.toml` → `~/.config/camera-yolo-logger.toml`

优先级: **CLI 参数 > 环境变量 > TOML 配置文件 > 默认值**

### 向后兼容的环境变量

```bash
export CAMERA_PHOTO_CMD=termux-camera-photo
export DETECT_CONF_THRESH=0.45
export DETECT_IOU_THRESH=0.5
export CAMERA_MODEL_PATH=yolov8n.onnx
export MODEL_DOWNLOAD_URL="https://huggingface.co/.../yolov8n.onnx"
export CLASS_FILTER=person,car,dog
export MOTION_ENABLED=true
export WEBBOOK_URL=http://localhost:8080/notify
export WEBBOOK_TRIGGER_CLASSES=person,car
```

## 项目结构

```
termux-camera-yolo/
├── camera_yolo_logger/
│   ├── __init__.py     # 版本号
│   ├── __main__.py     # python -m 入口
│   ├── schemas.py      # 数据结构 (BBox, Detection, DetectionResult, CaptureResult)
│   ├── config.py       # 配置管理 (Settings + CLI/env/TOML 分层加载)
│   ├── utils.py        # 工具 (FileLock 并发锁, timed 装饰器)
│   ├── setup.py        # 首次运行配置 (自动生成 TOML, --setup 向导, CSV 裁剪)
│   ├── capture.py      # 拍照 (termux-camera-photo / IP Webcam, 超时重试)
│   ├── detect.py       # YOLO ONNX 推理 (Detector 类, NMS, 数字变焦)
│   ├── motion.py       # 运动检测预过滤器 (帧差分 MSE)
│   ├── monitor.py      # 连续监控循环 (自适应间隔, 存档)
│   ├── notify.py       # Webhook 通知
│   ├── server.py       # Flask HTTP API
│   └── main.py         # CLI 入口 (argparse, 模式分发)
├── tests/
│   ├── conftest.py         # 共享 fixtures
│   ├── test_schemas.py     # 数据结构测试 (17)
│   ├── test_config.py      # 配置管理测试 (20)
│   ├── test_utils.py       # 工具测试 (9)
│   ├── test_capture.py     # 拍照测试 (11)
│   ├── test_detect.py      # 检测测试 (21)
│   ├── test_motion.py      # 运动检测测试 (8)
│   ├── test_main.py        # CLI 入口测试 (14)
│   ├── test_monitor.py     # 监控测试 (16)
│   ├── test_notify.py      # 通知测试 (9)
│   └── test_server.py      # HTTP API 测试 (10)
├── install.sh              # 一键安装脚本 (创建全局 camera-yolo 命令)
├── camera_yolo_logger.toml  # 示例配置文件
├── pyproject.toml           # 项目元数据 + 依赖 + pytest 配置
├── requirements.txt
├── SKILL.md                 # AI Agent 集成指南
└── README.md
```

## 测试

```bash
# 运行全部 135 项测试
uv run pytest tests/ -v

# 或在安装了 pytest 的环境中
python3 -m pytest tests/ -v

# 运行单个模块
uv run pytest tests/test_detect.py -v
```

## 环境要求

- Termux (Android) + `termux-api` 包
- Python ≥ 3.10
- 系统依赖（通过 `pkg` 安装）：`opencv-python`, `onnxruntime`（可选，uv 也可安装）
- ONNX 模型文件 (默认 `yolov8n.onnx`)

## 友链

[linux.do](https://linux.do)

## License

MIT
