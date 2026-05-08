"""测试 server.py — Flask HTTP API 所有端点。"""
from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from camera_yolo_logger.config import Settings
from camera_yolo_logger.server import create_app


@pytest.fixture
def app():
    s = Settings(
        log_file="/tmp/test_log.csv",
        model_path="/tmp/model.onnx",
    )
    # 创建假的模型文件和日志文件
    import pathlib
    pathlib.Path(s.model_path).write_text("dummy")
    pathlib.Path(s.log_file).write_text("timestamp,detected\n2026-01-01 00:00:00,test\n")

    app = create_app(s)
    app.config["TESTING"] = True
    return app


@pytest.fixture
def client(app):
    return app.test_client()


class TestStatusEndpoint:
    def test_status_ok(self, client):
        r = client.get("/status")
        assert r.status_code == 200
        data = r.get_json()
        assert data["status"] == "ok"
        assert data["version"] == "1.1.0"
        assert "uptime_s" in data
        assert "config" in data

    def test_model_info_in_status(self, client):
        r = client.get("/status")
        data = r.get_json()
        assert data["model"] == "/tmp/model.onnx"
        assert data["model_exists"] is True


class TestLogEndpoint:
    def test_log_returns_entries(self, client):
        r = client.get("/log")
        assert r.status_code == 200
        data = r.get_json()
        assert "entries" in data
        assert len(data["entries"]) > 0

    def test_log_with_limit(self, client):
        r = client.get("/log?limit=1")
        data = r.get_json()
        assert len(data["entries"]) == 1


class TestConfigEndpoint:
    def test_get_config(self, client):
        r = client.get("/config")
        assert r.status_code == 200
        data = r.get_json()
        assert "conf_threshold" in data
        assert "model" in data

    def test_post_config_update(self, client):
        r = client.post("/config",
                        data=json.dumps({"confidence": 0.8, "classes": ["person"]}),
                        content_type="application/json")
        assert r.status_code == 200
        data = r.get_json()
        assert data["status"] == "updated"


class TestDetectEndpoint:
    def test_detect_success(self):
        """模拟拍照 + 检测成功的完整流程。mock 必须在 app 创建前生效。"""
        from camera_yolo_logger.schemas import BBox, CaptureResult, Detection, DetectionResult
        from pathlib import Path

        with patch("camera_yolo_logger.server.Detector.from_settings") as mock_det:
            mock_detector = MagicMock()
            mock_detector.detect.return_value = DetectionResult(
                success=True,
                objects=[Detection("person", 0, 0.95, BBox(0, 0, 10, 10))],
                summary="1个人",
            )
            mock_det.return_value = mock_detector

            s = Settings(log_file="/tmp/test_log.csv", model_path="/tmp/model.onnx")
            Path("/tmp/model.onnx").write_text("dummy")
            Path("/tmp/test_log.csv").write_text("timestamp,detected\n")
            app = create_app(s)
            app.config["TESTING"] = True
            client = app.test_client()

            with patch("camera_yolo_logger.server.capture") as mock_cap:
                mock_cap.return_value = CaptureResult(
                    success=True, path=Path("/tmp/img.jpg"), image_size=(100, 100),
                )
                r = client.get("/detect")

        assert r.status_code == 200
        data = r.get_json()
        assert data["detection"]["objects"][0]["class"] == "person"

    def test_detect_capture_failure(self, client):
        from pathlib import Path
        from camera_yolo_logger.schemas import CaptureResult

        with patch("camera_yolo_logger.server.capture") as mock_cap:
            mock_cap.return_value = CaptureResult(
                success=False, error="Camera not available",
            )
            r = client.get("/detect")
        assert r.status_code == 500
        data = r.get_json()
        assert data["success"] is False

    def test_detect_summary_format(self):
        """mock 必须在 app 创建前生效。"""
        from pathlib import Path
        from camera_yolo_logger.schemas import BBox, CaptureResult, Detection, DetectionResult

        with patch("camera_yolo_logger.server.Detector.from_settings") as mock_det:
            mock_detector = MagicMock()
            mock_detector.detect.return_value = DetectionResult(
                success=True, objects=[], summary="1个人, 1辆车",
            )
            mock_det.return_value = mock_detector

            s = Settings(log_file="/tmp/test_log.csv", model_path="/tmp/model.onnx")
            Path("/tmp/model.onnx").write_text("dummy")
            Path("/tmp/test_log.csv").write_text("timestamp,detected\n")
            app = create_app(s)
            app.config["TESTING"] = True
            client = app.test_client()

            with patch("camera_yolo_logger.server.capture") as mock_cap:
                mock_cap.return_value = CaptureResult(
                    success=True, path=Path("/tmp/img.jpg"),
                )
                r = client.get("/detect?format=summary")
        assert r.status_code == 200
        data = r.get_json()
        assert data["result"] == "1个人, 1辆车"


class TestStreamEndpoint:
    def test_stream_no_ip_webcam(self, client):
        r = client.get("/stream")
        assert r.status_code == 400
        data = r.get_json()
        assert "未配置" in data["error"]
