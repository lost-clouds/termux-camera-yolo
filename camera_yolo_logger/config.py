"""配置管理 — Settings dataclass + 分层加载（CLI > 环境变量 > TOML > 默认值）。"""
from __future__ import annotations

import os
import sys
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ── 默认值常量 ──────────────────────────────────────────────
DEFAULT_PHOTO_CMD = "termux-camera-photo"
DEFAULT_PHOTO_PATH = "Image/camera_yolo_temp.jpg"
DEFAULT_MODEL_PATH = "yolov8n.onnx"
DEFAULT_LOG_FILE = "camera_log.csv"
DEFAULT_CONF_THRESH = 0.45
DEFAULT_IOU_THRESH = 0.5
DEFAULT_CAPTURE_TIMEOUT = 30
DEFAULT_CAPTURE_RETRY = 3
DEFAULT_CAPTURE_RETRY_DELAY = 1.0
DEFAULT_MONITOR_INTERVAL = 5.0
DEFAULT_MONITOR_INTERVAL_MIN = 1.0
DEFAULT_MONITOR_INTERVAL_MAX = 30.0
DEFAULT_MOTION_THRESHOLD = 0.05
DEFAULT_MOTION_RESIZE = 160
DEFAULT_SERVER_HOST = "127.0.0.1"
DEFAULT_SERVER_PORT = 5000
DEFAULT_ARCHIVE_DIR = "archive"
DEFAULT_WEBHOOK_COOLDOWN = 10.0

# ── Settings dataclass ──────────────────────────────────────


@dataclass
class Settings:
    # Capture
    photo_cmd: str = DEFAULT_PHOTO_CMD
    photo_path: str = DEFAULT_PHOTO_PATH
    capture_timeout: int = DEFAULT_CAPTURE_TIMEOUT
    capture_retry: int = DEFAULT_CAPTURE_RETRY
    capture_retry_delay: float = DEFAULT_CAPTURE_RETRY_DELAY
    capture_backend: str = "termux"
    ip_webcam_url: str = ""

    # Detection
    model_path: str = DEFAULT_MODEL_PATH
    model_download_url: str = ""
    conf_thresh: float = DEFAULT_CONF_THRESH
    iou_thresh: float = DEFAULT_IOU_THRESH
    class_filter: list[str] | None = None

    # Output
    log_file: str = DEFAULT_LOG_FILE
    output_format: str = "text"
    verbose: bool = False
    csv_max_records: int = 1000

    # Motion pre-filter
    motion_detection: bool = False
    motion_threshold: float = DEFAULT_MOTION_THRESHOLD
    motion_resize: int = DEFAULT_MOTION_RESIZE

    # Monitor
    monitor: bool = False
    monitor_interval: float = DEFAULT_MONITOR_INTERVAL
    monitor_interval_min: float = DEFAULT_MONITOR_INTERVAL_MIN
    monitor_interval_max: float = DEFAULT_MONITOR_INTERVAL_MAX
    archive_enabled: bool = False
    archive_dir: str = DEFAULT_ARCHIVE_DIR

    # Webhook
    webhook_url: str = ""
    webhook_trigger_classes: list[str] | None = None
    webhook_cooldown: float = DEFAULT_WEBHOOK_COOLDOWN

    # Server
    server: bool = False
    server_host: str = DEFAULT_SERVER_HOST
    server_port: int = DEFAULT_SERVER_PORT

    def to_snapshot(self) -> dict:
        """供 DetectionResult.config_snapshot 使用，返回非敏感配置摘要。"""
        return {
            "conf_threshold": self.conf_thresh,
            "iou_threshold": self.iou_thresh,
            "model": self.model_path,
            "class_filter": self.class_filter,
            "motion_detection": self.motion_detection,
        }


# ── 分层加载 ────────────────────────────────────────────────


def _env_to_dict() -> dict[str, Any]:
    """读取旧版 CAMERA_* / DETECT_* 环境变量。"""
    d: dict[str, Any] = {}
    mapping = {
        "CAMERA_PHOTO_CMD": ("photo_cmd", str),
        "CAMERA_PHOTO_PATH": ("photo_path", str),
        "CAMERA_LOG_FILE": ("log_file", str),
        "CAMERA_MODEL_PATH": ("model_path", str),
        "MODEL_DOWNLOAD_URL": ("model_download_url", str),
        "DETECT_CONF_THRESH": ("conf_thresh", float),
        "DETECT_IOU_THRESH": ("iou_thresh", float),
        "MOTION_ENABLED": ("motion_detection", lambda v: v.lower() in ("1", "true", "yes")),
        "MOTION_THRESHOLD": ("motion_threshold", float),
        "MONITOR_INTERVAL": ("monitor_interval", float),
        "WEBBOOK_URL": ("webhook_url", str),
        "WEBBOOK_TRIGGER_CLASSES": ("webhook_trigger_classes", lambda v: [c.strip() for c in v.split(",") if c.strip()]),
        "CLASS_FILTER": ("class_filter", lambda v: [c.strip() for c in v.split(",") if c.strip()]),
        "CAMERA_OUTPUT_FORMAT": ("output_format", str),
        "ARCHIVE_DIR": ("archive_dir", str),
        "IP_WEBCAM_URL": ("ip_webcam_url", str),
        "CAMERA_CAPTURE_BACKEND": ("capture_backend", str),
    }
    for env_key, (field_name, converter) in mapping.items():
        val = os.environ.get(env_key)
        if val is not None and val != "":
            try:
                d[field_name] = converter(val)
            except (ValueError, TypeError):
                pass
    return d


def _toml_to_dict(path: str | None) -> dict[str, Any]:
    """从 TOML 文件读取配置。使用 Python 3.11+ 内置 tomllib。

    path=None: 搜索默认路径
    path="":   不加载任何文件
    path=...:  加载指定文件
    """
    if path == "":
        return {}
    config_paths = [path] if path else [
        "camera_yolo_logger.toml",
        os.path.expanduser("~/.config/camera-yolo-logger.toml"),
    ]

    # TOML 友好键名 → Settings 内部字段名的映射（按 section 组织）
    section_defs = {
        "capture": {
            "fields": ["photo_cmd", "photo_path", "capture_timeout", "capture_retry",
                       "capture_retry_delay", "capture_backend", "ip_webcam_url"],
            "aliases": {"cmd": "photo_cmd", "path": "photo_path",
                        "timeout": "capture_timeout", "retry": "capture_retry",
                        "retry_delay": "capture_retry_delay", "backend": "capture_backend",
                        "ip_webcam_url": "ip_webcam_url"},
        },
        "detection": {
            "fields": ["model_path", "model_download_url", "conf_thresh", "iou_thresh",
                       "class_filter"],
            "aliases": {"model": "model_path", "model_download_url": "model_download_url",
                        "conf_threshold": "conf_thresh", "iou_threshold": "iou_thresh",
                        "classes": "class_filter"},
        },
        "output": {
            "fields": ["log_file", "output_format", "verbose", "csv_max_records"],
            "aliases": {"log_file": "log_file", "format": "output_format",
                        "verbose": "verbose", "max_records": "csv_max_records"},
        },
        "motion": {
            "fields": ["motion_detection", "motion_threshold", "motion_resize"],
            "aliases": {"enabled": "motion_detection", "threshold": "motion_threshold",
                        "resize": "motion_resize"},
        },
        "monitor": {
            "fields": ["monitor_interval", "monitor_interval_min", "monitor_interval_max",
                       "archive_enabled", "archive_dir"],
            "aliases": {"interval": "monitor_interval", "interval_min": "monitor_interval_min",
                        "interval_max": "monitor_interval_max", "archive": "archive_enabled",
                        "archive_dir": "archive_dir"},
        },
        "webhook": {
            "fields": ["webhook_url", "webhook_trigger_classes", "webhook_cooldown"],
            "aliases": {"url": "webhook_url", "trigger_classes": "webhook_trigger_classes",
                        "cooldown": "webhook_cooldown"},
        },
        "server": {
            "fields": ["server_host", "server_port"],
            "aliases": {"host": "server_host", "port": "server_port"},
        },
    }

    d: dict[str, Any] = {}
    for p in config_paths:
        if p and Path(p).is_file():
            try:
                if sys.version_info >= (3, 11):
                    import tomllib
                    raw = tomllib.loads(Path(p).read_text(encoding="utf-8"))
                else:
                    import tomli as tomllib
                    raw = tomllib.loads(Path(p).read_text(encoding="utf-8"))
            except ImportError:
                continue
            except Exception:
                continue

            for section, sec_def in section_defs.items():
                if section not in raw:
                    continue
                for f in sec_def["fields"]:
                    if f in raw[section]:
                        d[f] = raw[section][f]
                for alias, field in sec_def["aliases"].items():
                    if alias in raw[section] and field not in d:
                        d[field] = raw[section][alias]
            break
    return d


def _cli_to_dict(args: Namespace | None) -> dict[str, Any]:
    """从 argparse 解析结果提取有效字段。"""
    if args is None:
        return {}
    d: dict[str, Any] = {}
    cli_fields = [
        "photo_cmd", "photo_path", "capture_timeout", "capture_retry",
        "capture_retry_delay", "capture_backend", "ip_webcam_url",
        "model_path", "model_download_url", "conf_thresh", "iou_thresh",
        "class_filter", "log_file", "output_format", "verbose", "csv_max_records",
        "motion_detection", "motion_threshold", "motion_resize",
        "monitor", "monitor_interval", "monitor_interval_min", "monitor_interval_max",
        "archive_enabled", "archive_dir",
        "webhook_url", "webhook_trigger_classes", "webhook_cooldown",
        "server", "server_host", "server_port",
        "config_file",
    ]
    # CLI 参数名 → Settings 字段名映射（兼容 argparse dest 命名差异）
    cli_aliases = {
        "classes": "class_filter",
        "confidence": "conf_thresh",
        "conf": "conf_thresh",
        "model": "model_path",
        "model_download_url": "model_download_url",
        "photo_cmd": "photo_cmd",
        "json_output": "output_format",
        "interval": "monitor_interval",
        "interval_min": "monitor_interval_min",
        "interval_max": "monitor_interval_max",
    }
    for f in cli_fields:
        val = getattr(args, f, None)
        if val is not None and val != []:
            d[f] = val
        elif f in cli_aliases and str(cli_aliases[f]).endswith(f):
            pass  # 主名已检查过
    # CLI 别名映射
    for alias, field in cli_aliases.items():
        if field not in d:
            val = getattr(args, alias, None)
            if val is not None and val != []:
                if alias == "json_output" and val:
                    d["output_format"] = "json"
                elif field == "conf_thresh" and alias in ("confidence", "conf"):
                    d[field] = val
                elif field == "class_filter" and alias == "classes":
                    d[field] = val
                elif field == "model_path" and alias == "model":
                    d[field] = val
                elif field in ("monitor_interval", "monitor_interval_min", "monitor_interval_max"):
                    if alias == "interval" and field == "monitor_interval":
                        d[field] = val
                    elif alias == "interval_min" and field == "monitor_interval_min":
                        d[field] = val
                    elif alias == "interval_max" and field == "monitor_interval_max":
                        d[field] = val
    return d


def load_settings(config_path: str | None = None,
                  cli_args: Namespace | None = None) -> Settings:
    """加载配置，优先级：CLI > 环境变量 > TOML 文件 > 默认值。

    Args:
        config_path: TOML 配置文件路径 (None=搜索默认路径, ""=跳过)。
        cli_args: argparse 解析后的命令行参数。
    """
    merged: dict[str, Any] = {}

    # 1. TOML 文件
    merged.update(_toml_to_dict(config_path))

    # 2. 环境变量
    merged.update(_env_to_dict())

    # 3. CLI 参数（最高优先级）
    merged.update(_cli_to_dict(cli_args))

    # 用合并后的值构造 Settings
    field_names = {f.name for f in __import__("dataclasses").fields(Settings)}
    kwargs = {k: v for k, v in merged.items() if k in field_names}
    return Settings(**kwargs)
