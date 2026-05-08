"""共享 fixtures — 临时目录、测试图片、Settings 实例。"""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from camera_yolo_logger.config import Settings


@pytest.fixture
def tmp_dir():
    """临时目录，测试结束后自动清理。"""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_image(tmp_dir):
    """生成 480x640 测试 JPEG（含红绿蓝条纹用于运动检测测试）。"""
    from PIL import Image
    import numpy as np

    arr = np.zeros((480, 640, 3), dtype=np.uint8)
    arr[:, :, 0] = 255  # Red channel full
    img = Image.fromarray(arr)
    path = tmp_dir / "sample.jpg"
    img.save(str(path))
    return path


@pytest.fixture
def sample_image2(tmp_dir):
    """生成与 sample_image 不同的测试 JPEG（模拟运动后的帧）。"""
    from PIL import Image
    import numpy as np

    arr = np.zeros((480, 640, 3), dtype=np.uint8)
    arr[:, :, 1] = 255  # Green channel full
    img = Image.fromarray(arr)
    path = tmp_dir / "sample2.jpg"
    img.save(str(path))
    return path


@pytest.fixture
def default_settings():
    return Settings()


@pytest.fixture
def custom_settings():
    return Settings(
        conf_thresh=0.6,
        iou_thresh=0.4,
        class_filter=["person", "car"],
        motion_detection=True,
        output_format="json",
        monitor_interval=3.0,
    )
