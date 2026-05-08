"""首次运行配置 — 自动生成默认配置文件、交互式设置向导、CSV 裁剪。"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

from camera_yolo_logger.config import Settings

CONFIG_PATHS = [
    Path("camera_yolo_logger.toml"),
    Path(os.path.expanduser("~/.config/camera-yolo-logger.toml")),
]

SETUP_DONE_MARKER = ".camera_yolo_setup_done"


def is_first_run(project_dir: Path | None = None) -> bool:
    """检查是否首次运行（配置文件不存在 + marker 文件不存在）。"""
    # 有配置文件 → 不是首次
    for p in CONFIG_PATHS:
        if p.exists():
            return False
    # 有 marker 文件 → 不是首次
    pd = project_dir or Path.cwd()
    if (pd / SETUP_DONE_MARKER).exists():
        return False
    return True


def generate_default_config(project_dir: Path | None = None,
                            photo_dir: str | None = None,
                            csv_max: int = 1000) -> Path:
    """生成默认配置文件。

    Args:
        project_dir: 项目根目录
        photo_dir: 照片保存目录（None=项目目录下 Image/）
        csv_max: CSV 最大记录条数
    Returns:
        生成的配置文件路径
    """
    pd = project_dir or Path.cwd()
    config_path = pd / "camera_yolo_logger.toml"

    if photo_dir is None:
        photo_dir = str(pd / "Image")

    # 确保照片目录存在
    Path(photo_dir).mkdir(parents=True, exist_ok=True)

    content = f"""# Camera YOLO Logger — 配置文件
# 自动生成于首次运行。可直接编辑此文件。

[capture]
cmd = "termux-camera-photo"
path = "{photo_dir}/camera_yolo_temp.jpg"

[detection]
model = "yolov8n.onnx"
conf_threshold = 0.45
iou_threshold = 0.5
classes = []

[output]
log_file = "camera_log.csv"
format = "text"
max_records = {csv_max}

[motion]
enabled = false
threshold = 0.05

[monitor]
interval = 5.0
archive = false
archive_dir = "archive"

[webhook]
url = ""

[server]
host = "127.0.0.1"
port = 5000
"""
    config_path.write_text(content, encoding="utf-8")
    # 写 marker 文件防止重复弹出
    (pd / SETUP_DONE_MARKER).write_text("")
    return config_path


def run_setup_wizard(project_dir: Path | None = None) -> Settings:
    """交互式配置向导（camera-yolo --setup 时调用）。"""
    pd = project_dir or Path.cwd()

    print("Camera YOLO Logger — 首次配置向导\n")

    # 照片保存目录
    default_photo = str(pd / "Image")
    photo = input(f"照片保存目录 [{default_photo}]: ").strip()
    if not photo:
        photo = default_photo

    # CSV 最大记录数
    default_max = "1000"
    csv_max_str = input(f"CSV 最大记录条数 [{default_max}]: ").strip()
    try:
        csv_max = int(csv_max_str) if csv_max_str else 1000
    except ValueError:
        csv_max = 1000

    # 类过滤
    default_classes = ""
    classes_str = input(f"仅检测的类别（逗号分隔，留空=全部）[{default_classes}]: ").strip()
    class_filter = [c.strip() for c in classes_str.split(",") if c.strip()] or None

    # 置信度阈值
    default_conf = "0.45"
    conf_str = input(f"置信度阈值 [{default_conf}]: ").strip()
    try:
        conf_thresh = float(conf_str) if conf_str else 0.45
    except ValueError:
        conf_thresh = 0.45

    # 生成配置
    config_path = generate_default_config(pd, photo_dir=photo, csv_max=csv_max)

    # 更新用户自定义项
    if class_filter or conf_thresh != 0.45:
        existing = config_path.read_text(encoding="utf-8")
        if class_filter:
            classes_line = f'classes = {class_filter}'
            existing = existing.replace("classes = []", classes_line)
        if conf_thresh != 0.45:
            existing = existing.replace("conf_threshold = 0.45",
                                        f"conf_threshold = {conf_thresh}")
        config_path.write_text(existing, encoding="utf-8")

    print(f"\n配置已保存到: {config_path}")
    print("之后可直接编辑该文件，或重新运行 camera-yolo --setup\n")

    from camera_yolo_logger.config import load_settings
    return load_settings()


def trim_csv(log_path: Path, max_records: int) -> int:
    """裁剪 CSV 文件，保留 header + 最近 max_records 条记录。

    Returns:
        删除的记录条数
    """
    if max_records <= 0:
        return 0
    if not log_path.exists():
        return 0

    try:
        with open(log_path, "r", newline="") as f:
            rows = list(csv.reader(f))
    except Exception:
        return 0

    if len(rows) <= 1:  # 只有 header 或空
        return 0

    header = rows[0]
    data = rows[1:]

    if len(data) <= max_records:
        return 0

    trimmed = data[-max_records:]
    removed = len(data) - len(trimmed)

    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(trimmed)

    return removed


def ensure_first_run(settings: Settings, project_dir: Path | None = None) -> Settings:
    """确保首次运行完成初始化（自动生成默认配置）。

    非交互式，直接使用默认值创建配置。
    用于 AI Agent 调用场景。
    """
    if not is_first_run(project_dir):
        return settings

    pd = project_dir or Path.cwd()
    config_path = generate_default_config(pd)
    print(f"[camera-yolo] 默认配置已创建: {config_path}", file=sys.stderr)
    print(f"[camera-yolo] 可通过 camera-yolo --setup 重新配置", file=sys.stderr)

    # 重新加载以获取新生成的配置
    from camera_yolo_logger.config import load_settings
    return load_settings()
