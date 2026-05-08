"""测试 main.py — argparse CLI、格式化函数、CSV 日志、run_once。"""
from __future__ import annotations

import csv
import json
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from camera_yolo_logger.main import (
    _format_json,
    _format_text,
    _log_to_csv,
    build_parser,
    run_once,
)
from camera_yolo_logger.config import Settings
from camera_yolo_logger.schemas import BBox, CaptureResult, Detection, DetectionResult


class TestBuildParser:
    def test_default_args_text_output(self):
        p = build_parser()
        args = p.parse_args([])
        assert args.json_output is False
        assert args.monitor is False
        assert args.server is False

    def test_json_flag(self):
        p = build_parser()
        args = p.parse_args(["--json"])
        assert args.json_output is True

    def test_classes(self):
        p = build_parser()
        args = p.parse_args(["--classes", "person", "car", "dog"])
        assert args.classes == ["person", "car", "dog"]

    def test_monitor_with_interval(self):
        p = build_parser()
        args = p.parse_args(["--monitor", "--interval", "2.5"])
        assert args.monitor is True
        assert args.interval == 2.5

    def test_server_with_port(self):
        p = build_parser()
        args = p.parse_args(["--server", "--server-port", "8080"])
        assert args.server is True
        assert args.server_port == 8080

    def test_motion(self):
        p = build_parser()
        args = p.parse_args(["--motion", "--motion-threshold", "0.03"])
        assert args.motion_detection is True
        assert args.motion_threshold == 0.03

    def test_archive(self):
        p = build_parser()
        args = p.parse_args(["--archive", "--archive-dir", "my_archive"])
        assert args.archive_enabled is True
        assert args.archive_dir == "my_archive"

    def test_webhook(self):
        p = build_parser()
        args = p.parse_args([
            "--webhook-url", "http://localhost:8080/notify",
            "--webhook-trigger-classes", "person", "car",
            "--webhook-cooldown", "15.0",
        ])
        assert args.webhook_url == "http://localhost:8080/notify"
        assert args.webhook_trigger_classes == ["person", "car"]
        assert args.webhook_cooldown == 15.0

    def test_capture_backend(self):
        p = build_parser()
        args = p.parse_args(["--capture-backend", "ipwebcam",
                             "--ip-webcam-url", "http://192.168.1.100:8080/shot.jpg"])
        assert args.capture_backend == "ipwebcam"
        assert args.ip_webcam_url == "http://192.168.1.100:8080/shot.jpg"

    def test_confidence_and_iou(self):
        p = build_parser()
        args = p.parse_args(["--confidence", "0.8", "--iou", "0.3"])
        assert args.confidence == 0.8
        assert args.iou == 0.3


class TestFormatFunctions:
    def test_format_text(self):
        det = Detection("person", 0, 0.95, BBox(0, 0, 10, 10))
        result = DetectionResult(success=True, objects=[det], summary="1个人")
        out = _format_text("2026-05-08 12:00:00", result)
        assert out == "2026-05-08 12:00:00,1个人"

    def test_format_text_no_objects(self):
        result = DetectionResult(success=True, objects=[], summary="未识别到物体")
        out = _format_text("2026-05-08 12:00:00", result)
        assert ",未识别到物体" in out

    def test_format_json(self):
        det = Detection("car", 2, 0.88, BBox(10, 20, 100, 80))
        detect_result = DetectionResult(
            success=True, objects=[det], summary="1辆车", elapsed_ms=500.0,
            image_path="/tmp/img.jpg",
        )
        cap_result = CaptureResult(
            success=True, path=Path("/tmp/img.jpg"),
            image_size=(640, 480), elapsed_ms=300.0,
        )
        out = _format_json("2026-05-08T12:00:00Z", cap_result, detect_result)
        parsed = json.loads(out)
        assert parsed["version"] == "1.1.0"
        assert parsed["capture"]["success"] is True
        assert parsed["detection"]["objects"][0]["class"] == "car"


class TestLogToCSV:
    def test_creates_new_file(self, tmp_dir):
        log_path = tmp_dir / "test_log.csv"
        _log_to_csv("2026-05-08 12:00:00", "1个人", log_path)
        assert log_path.exists()

        with open(log_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
        assert rows[0] == ["timestamp", "detected"]
        assert rows[1] == ["2026-05-08 12:00:00", "1个人"]

    def test_appends_to_existing(self, tmp_dir):
        log_path = tmp_dir / "test_log.csv"
        _log_to_csv("t1", "d1", log_path)
        _log_to_csv("t2", "d2", log_path)

        with open(log_path) as f:
            rows = list(csv.reader(f))
        assert len(rows) == 3  # header + 2 rows
        assert rows[1] == ["t1", "d1"]
        assert rows[2] == ["t2", "d2"]


class TestRunOnce:
    def test_capture_failure_returns_1(self, tmp_dir):
        s = Settings(photo_path=str(tmp_dir / "img.jpg"),
                     photo_cmd="nonexistent-cmd-12345",
                     capture_retry=1, capture_timeout=2,
                     log_file=str(tmp_dir / "log.csv"))
        with patch("subprocess.run", side_effect=FileNotFoundError):
            rc = run_once(s)
        assert rc == 1

    def test_successful_flow_text_mode(self, tmp_dir):
        """模拟拍照成功 + 检测成功的完整流程（text 输出模式）。"""
        img_path = tmp_dir / "img.jpg"
        import numpy as np
        from PIL import Image
        Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(str(img_path))

        s = Settings(
            photo_path=str(img_path), log_file=str(tmp_dir / "log.csv"),
            output_format="text", conf_thresh=0.9,
            capture_retry=1, capture_timeout=10,
        )

        # 模拟无检测情况 (高阈值)
        fake_model = tmp_dir / "fake.onnx"
        fake_model.write_text("dummy")

        with patch("camera_yolo_logger.main.capture") as mock_cap:
            mock_cap.return_value = CaptureResult(
                success=True, path=img_path, image_size=(100, 100),
            )
            with patch("camera_yolo_logger.detect.Detector._get_model_path",
                       return_value=fake_model):
                with patch("camera_yolo_logger.detect.Detector._get_session") as mock_sess:
                    sess = MagicMock()
                    sess.get_inputs.return_value = [MagicMock(name="images")]
                    sess.get_inputs.return_value[0].name = "images"
                    fake_output = np.zeros((1, 84, 8400), dtype=np.float32)
                    sess.run.return_value = [fake_output]
                    mock_sess.return_value = sess

                    rc = run_once(s)
        assert rc == 0

    def test_successful_json_mode(self, tmp_dir):
        """模拟拍照成功 + 检测成功的完整流程（JSON 输出模式）。"""
        img_path = tmp_dir / "img.jpg"
        import numpy as np
        from PIL import Image
        Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8)).save(str(img_path))

        s = Settings(
            photo_path=str(img_path), log_file=str(tmp_dir / "log.csv"),
            output_format="json", conf_thresh=0.9,
            capture_retry=1, capture_timeout=10,
        )

        with patch("camera_yolo_logger.main.capture") as mock_cap:
            mock_cap.return_value = CaptureResult(
                success=True, path=img_path, image_size=(100, 100),
            )
            with patch("camera_yolo_logger.detect.Detector._get_model_path",
                       return_value=img_path):  # exists() returns True
                with patch("camera_yolo_logger.detect.Detector._get_session") as mock_sess:
                    sess = MagicMock()
                    sess.get_inputs.return_value = [MagicMock(name="images")]
                    sess.get_inputs.return_value[0].name = "images"
                    fake_output = np.zeros((1, 84, 8400), dtype=np.float32)
                    sess.run.return_value = [fake_output]
                    mock_sess.return_value = sess

                    rc = run_once(s)
        assert rc == 0
