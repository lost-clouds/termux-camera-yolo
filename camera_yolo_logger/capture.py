"""拍照模块 — 专注 Termux，使用 termux-camera-photo 拍照。"""
import os
import subprocess
from pathlib import Path

PHOTO_PATH = Path(os.environ.get("CAMERA_PHOTO_PATH", "Image/camera_yolo_temp.jpg"))
PHOTO_CMD  = os.environ.get("CAMERA_PHOTO_CMD", "termux-camera-photo")

def cleanup_photo() -> None:
    PHOTO_PATH.unlink(missing_ok=True)

def capture() -> Path | None:
    """拍照并保存到 PHOTO_PATH，返回路径或 None。"""
    PHOTO_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleanup_photo()
    r = subprocess.run([PHOTO_CMD, str(PHOTO_PATH)], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"拍照失败: {r.stderr.strip()}", file=__import__("sys").stderr)
        return None
    return PHOTO_PATH
