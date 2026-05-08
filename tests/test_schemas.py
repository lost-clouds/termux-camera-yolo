"""测试 schemas.py — 数据结构创建、序列化、__bool__。"""
from __future__ import annotations

import json

from camera_yolo_logger.schemas import BBox, CaptureResult, Detection, DetectionResult, MonitorStats


class TestBBox:
    def test_create(self):
        b = BBox(10, 20, 100, 200)
        assert b.x1 == 10
        assert b.y1 == 20
        assert b.x2 == 100
        assert b.y2 == 200

    def test_area(self):
        b = BBox(0, 0, 100, 50)
        assert b.area == 5000

    def test_area_zero(self):
        b = BBox(5, 5, 5, 5)
        assert b.area == 0

    def test_to_dict(self):
        b = BBox(1, 2, 3, 4)
        d = b.to_dict()
        assert d == {"x1": 1, "y1": 2, "x2": 3, "y2": 4}


class TestDetection:
    def test_create(self):
        b = BBox(10, 20, 100, 200)
        d = Detection(class_name="person", class_id=0, confidence=0.95, bbox=b)
        assert d.class_name == "person"
        assert d.class_id == 0
        assert d.confidence == 0.95
        assert d.bbox == b

    def test_to_dict(self):
        d = Detection("car", 2, 0.8765, BBox(1, 2, 3, 4))
        result = d.to_dict()
        assert result["class"] == "car"
        assert result["class_id"] == 2
        assert result["confidence"] == 0.8765
        assert result["bbox"] == {"x1": 1, "y1": 2, "x2": 3, "y2": 4}


class TestDetectionResult:
    def test_success_empty(self):
        r = DetectionResult(success=True, objects=[], summary="未识别到物体")
        assert bool(r) is True
        assert r.summary == "未识别到物体"
        assert len(r.objects) == 0

    def test_success_with_objects(self):
        d = Detection("person", 0, 0.9, BBox(0, 0, 10, 10))
        r = DetectionResult(success=True, objects=[d], summary="1个人", elapsed_ms=500.0)
        assert bool(r) is True
        assert r.elapsed_ms == 500.0

    def test_failure(self):
        r = DetectionResult(success=False, objects=[], summary="识别出错: test", error="test")
        assert bool(r) is False
        assert r.error == "test"

    def test_to_dict_minimal(self):
        r = DetectionResult(success=True, objects=[], summary="无")
        d = r.to_dict()
        assert d["success"] is True
        assert d["objects"] == []
        assert d["summary"] == "无"
        assert "error" not in d
        assert "image_path" not in d

    def test_to_dict_full(self):
        d = Detection("dog", 16, 0.88, BBox(50, 60, 150, 200))
        r = DetectionResult(
            success=True, objects=[d], summary="1只狗", elapsed_ms=800.0,
            image_path="/tmp/img.jpg", image_size=(1920, 1080),
            config_snapshot={"conf_threshold": 0.5},
        )
        result = r.to_dict()
        assert result["image_path"] == "/tmp/img.jpg"
        assert result["image_size"] == [1920, 1080]
        assert result["config"] == {"conf_threshold": 0.5}
        assert result["objects"][0]["class"] == "dog"

    def test_to_json(self):
        d = Detection("cat", 15, 0.75, BBox(1, 2, 3, 4))
        r = DetectionResult(success=True, objects=[d], summary="1只猫")
        j = r.to_json()
        parsed = json.loads(j)
        assert parsed["success"] is True
        assert parsed["objects"][0]["class"] == "cat"


class TestCaptureResult:
    def test_success(self):
        from pathlib import Path
        r = CaptureResult(success=True, path=Path("/tmp/img.jpg"),
                          image_size=(640, 480), elapsed_ms=300.0)
        assert bool(r) is True
        assert r.path == Path("/tmp/img.jpg")
        assert r.image_size == (640, 480)

    def test_failure(self):
        r = CaptureResult(success=False, error="拍照超时")
        assert bool(r) is False
        assert r.error == "拍照超时"

    def test_to_dict(self):
        from pathlib import Path
        r = CaptureResult(success=True, path=Path("/tmp/img.jpg"),
                          image_size=(100, 200))
        d = r.to_dict()
        assert d["success"] is True
        assert d["path"] == "/tmp/img.jpg"
        assert d["image_size"] == [100, 200]


class TestMonitorStats:
    def test_defaults(self):
        s = MonitorStats()
        assert s.iterations == 0
        assert s.skipped_motion == 0
        assert s.detections_total == 0

    def test_to_dict(self):
        s = MonitorStats(iterations=10, skipped_motion=5, detections_total=3,
                         capture_time_ms=5000, detect_time_ms=2000)
        d = s.to_dict()
        assert d["iterations"] == 10
        assert d["skipped_motion"] == 5
        assert d["detections_total"] == 3
        assert "uptime_s" in d
        assert d["avg_capture_ms"] == 500.0
