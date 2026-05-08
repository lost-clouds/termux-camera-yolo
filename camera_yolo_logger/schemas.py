"""共享数据结构 — BBox, Detection, DetectionResult, CaptureResult, MonitorStats。"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def area(self) -> int:
        return (self.x2 - self.x1) * (self.y2 - self.y1)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Detection:
    class_name: str
    class_id: int
    confidence: float
    bbox: BBox

    def to_dict(self) -> dict:
        return {
            "class": self.class_name,
            "class_id": self.class_id,
            "confidence": round(self.confidence, 4),
            "bbox": self.bbox.to_dict(),
        }


@dataclass
class DetectionResult:
    success: bool
    objects: list[Detection]
    summary: str = ""
    elapsed_ms: float = 0.0
    image_path: str | None = None
    image_size: tuple[int, int] | None = None
    error: str | None = None
    config_snapshot: dict | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "success": self.success,
            "objects": [o.to_dict() for o in self.objects],
            "summary": self.summary,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }
        if self.image_path:
            d["image_path"] = self.image_path
        if self.image_size:
            d["image_size"] = list(self.image_size)
        if self.error:
            d["error"] = self.error
        if self.config_snapshot:
            d["config"] = self.config_snapshot
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def __bool__(self) -> bool:
        return self.success


@dataclass
class CaptureResult:
    success: bool
    path: Path | None = None
    error: str | None = None
    elapsed_ms: float = 0.0
    image_size: tuple[int, int] | None = None

    def to_dict(self) -> dict:
        d: dict = {
            "success": self.success,
            "elapsed_ms": round(self.elapsed_ms, 1),
        }
        if self.path:
            d["path"] = str(self.path)
        if self.image_size:
            d["image_size"] = list(self.image_size)
        if self.error:
            d["error"] = self.error
        return d

    def __bool__(self) -> bool:
        return self.success


@dataclass
class MonitorStats:
    iterations: int = 0
    skipped_motion: int = 0
    skipped_capture: int = 0
    detections_total: int = 0
    capture_time_ms: float = 0.0
    detect_time_ms: float = 0.0
    start_time: float = field(default_factory=__import__("time").time)
    last_detection_time: float = 0.0

    def to_dict(self) -> dict:
        import time
        uptime = time.time() - self.start_time
        return {
            "uptime_s": round(uptime, 1),
            "iterations": self.iterations,
            "skipped_motion": self.skipped_motion,
            "skipped_capture": self.skipped_capture,
            "detections_total": self.detections_total,
            "avg_capture_ms": round(self.capture_time_ms / max(1, self.iterations), 1),
            "avg_detect_ms": round(self.detect_time_ms / max(1, self.iterations - self.skipped_motion - self.skipped_capture), 1),
            "last_detection_time": self.last_detection_time,
        }
