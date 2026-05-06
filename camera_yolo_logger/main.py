"""CLI 入口 — 单一职责：清理→拍照→识别→输出。"""
import os, sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from camera_yolo_logger.capture import capture, cleanup_photo
from camera_yolo_logger.detect import detect

LOG_FILE = Path(os.environ.get("CAMERA_LOG_FILE", "camera_log.csv"))

def main() -> int:
    """执行一次检测并记录到 CSV。"""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    img_path = capture()
    if img_path is None:
        print(f"{ts},拍照失败")
        _log(ts, "拍照失败")
        return 1

    result = detect(img_path)
    cleanup_photo()
    print(f"{ts},{result}")
    _log(ts, result)
    return 0

def _log(ts: str, description: str) -> None:
    import csv
    need_header = not LOG_FILE.exists() or LOG_FILE.stat().st_size == 0
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(["timestamp", "detected"])
        w.writerow([ts, description])

if __name__ == "__main__":
    raise SystemExit(main())
