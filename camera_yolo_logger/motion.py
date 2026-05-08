"""运动检测预过滤器 — 帧差分，静态场景跳过 YOLO 推理以节省功耗。"""
from __future__ import annotations

from pathlib import Path

import numpy as np


class MotionDetector:
    """基于帧差分的运动检测器。

    将新帧与参考帧在低分辨率灰度图上做 MSE 比较。
    复杂度 ~O(resize²)，远低于 YOLO 推理，适合作为预过滤器。

    用法:
        md = MotionDetector(resize=160, threshold=0.05)
        if md.has_motion("frame.jpg"):
            run_yolo()
            md.update("frame.jpg")  # 更新参考帧
        # 无运动 → 跳过 YOLO
    """

    def __init__(self, resize: int = 160, threshold: float = 0.05):
        self.resize = resize
        self.threshold = threshold
        self._reference: np.ndarray | None = None

    def _to_grayscale(self, image_path: str | Path) -> np.ndarray:
        """加载图片 → 灰度 → 缩放到 (resize, resize * aspect)。"""
        try:
            from PIL import Image
        except ImportError:
            raise ImportError("运动检测需要 Pillow 库")
        img = Image.open(str(image_path)).convert("L")
        w, h = img.size
        scale = self.resize / min(w, h)
        new_size = (int(w * scale), int(h * scale))
        if new_size[0] < 1 or new_size[1] < 1:
            raise ValueError(f"图片太小: {w}x{h}")
        return np.array(img.resize(new_size, Image.BILINEAR), dtype=np.float32)

    def score(self, image_path: str | Path) -> float:
        """计算运动分数（MSE vs 参考帧）。首帧返回 0.0 并自动设参考。"""
        try:
            frame = self._to_grayscale(image_path)
        except Exception:
            return 0.0
        if self._reference is None or self._reference.shape != frame.shape:
            self._reference = frame
            return 0.0
        diff = frame - self._reference
        return float(np.mean(diff * diff))

    def has_motion(self, image_path: str | Path, threshold: float | None = None) -> bool:
        """判断是否有运动。首次调用自动用当前帧作为参考。"""
        return self.score(image_path) > (threshold or self.threshold)

    def update(self, image_path: str | Path) -> None:
        """手动更新参考帧（检测到运动后调用，重置基线）。"""
        try:
            self._reference = self._to_grayscale(image_path)
        except Exception:
            pass

    def reset(self) -> None:
        """清除参考帧（场景切换时调用）。"""
        self._reference = None
