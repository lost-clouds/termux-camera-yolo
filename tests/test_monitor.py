"""测试 monitor.py — Monitor 初始化、间隔计算、信号处理、统计。"""
from __future__ import annotations

import signal
import time
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest

from camera_yolo_logger.config import Settings
from camera_yolo_logger.monitor import Monitor
from camera_yolo_logger.schemas import BBox, CaptureResult, Detection, DetectionResult, MonitorStats


class TestMonitorInit:
    def test_basic_init(self):
        s = Settings()
        m = Monitor(s)
        assert m.running is False
        assert m._signal_received is False
        assert m.stats.iterations == 0
        assert m.motion is None

    def test_init_with_motion(self):
        s = Settings(motion_detection=True, motion_threshold=0.03, motion_resize=200)
        m = Monitor(s)
        assert m.motion is not None
        assert m.motion.threshold == 0.03
        assert m.motion.resize == 200

    def test_init_with_webhook(self):
        s = Settings(webhook_url="http://example.com/hook",
                     webhook_trigger_classes=["person"])
        m = Monitor(s)
        assert m._notifier is not None

    def test_init_without_webhook(self):
        s = Settings(webhook_url="")
        m = Monitor(s)
        assert m._notifier is None


class TestIntervalCalculation:
    def test_detected_objects_min_interval(self):
        s = Settings(monitor_interval_min=1.0, monitor_interval=5.0,
                     monitor_interval_max=30.0)
        m = Monitor(s)
        det = DetectionResult(success=True,
                              objects=[Detection("person", 0, 0.9, BBox(0, 0, 10, 10))],
                              summary="1个人")
        assert m._calc_interval(det) == 1.0

    def test_no_objects_no_motion_mid_interval(self):
        s = Settings(monitor_interval_min=1.0, monitor_interval=5.0,
                     monitor_interval_max=30.0)
        m = Monitor(s)
        det = DetectionResult(success=True, objects=[], summary="无")
        assert m._calc_interval(det) == 5.0

    def test_no_objects_with_motion_max_interval(self):
        s = Settings(monitor_interval_min=1.0, monitor_interval=5.0,
                     monitor_interval_max=30.0, motion_detection=True)
        m = Monitor(s)
        det = DetectionResult(success=True, objects=[], summary="无")
        assert m._calc_interval(det) == 30.0


class TestMonitorStats:
    def test_stats_initial(self):
        s = MonitorStats()
        d = s.to_dict()
        assert d["iterations"] == 0
        assert d["skipped_motion"] == 0
        assert d["detections_total"] == 0
        assert d["uptime_s"] >= 0

    def test_stats_after_iterations(self):
        s = MonitorStats(iterations=20, skipped_motion=10, detections_total=5,
                         capture_time_ms=10000, detect_time_ms=5000)
        d = s.to_dict()
        assert d["iterations"] == 20
        assert d["avg_capture_ms"] == 500.0
        assert d["avg_detect_ms"] == 500.0  # (20-10) = 10 detections, 5000ms / 10


class TestSignalHandling:
    def test_stop_sets_running_false(self):
        s = Settings()
        m = Monitor(s)
        m.running = True
        m.stop()
        assert m.running is False

    def test_handle_signal_stops(self):
        s = Settings()
        m = Monitor(s)
        m.running = True
        m._handle_signal(signal.SIGINT, None)
        assert m._signal_received is True
        assert m.running is False


class TestTimestampHelpers:
    def test_now_str_format(self):
        ts = Monitor._now_str()
        # Should match YYYY-MM-DD HH:MM:SS
        import re
        assert re.match(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}", ts)

    def test_now_iso_format(self):
        ts = Monitor._now_iso()
        assert "T" in ts
        assert "-" in ts


class TestMonitorRun:
    def test_run_stops_on_signal(self):
        """验证收到信号后循环停止。"""
        s = Settings(monitor_interval=0.01, capture_retry=1,
                     capture_timeout=1, photo_cmd="nonexistent")
        m = Monitor(s)
        # 在第一次迭代后发送信号
        with patch("camera_yolo_logger.monitor.capture") as mock_cap:
            mock_cap.return_value = CaptureResult(
                success=False, error="simulated failure",
            )
            # 使用 side effect 在第二次迭代时设置信号
            def stop_after_first(*a, **kw):
                m._signal_received = True
                m.running = False

            with patch.object(m, "_adaptive_sleep", side_effect=stop_after_first):
                m.run()
        # 应该只迭代了 1 次（失败后触发 adaptive_sleep，side effect 停止循环）
        assert m.stats.iterations >= 1

    def test_run_archive_on_detection(self, tmp_dir):
        """检测到物体时存档。"""
        archive_dir = tmp_dir / "archive"
        s = Settings(monitor_interval=0.01, capture_retry=1, capture_timeout=1,
                     photo_cmd="nonexistent", archive_enabled=True,
                     archive_dir=str(archive_dir), photo_path=str(tmp_dir / "img.jpg"))
        m = Monitor(s)
        det = DetectionResult(
            success=True,
            objects=[Detection("person", 0, 0.9, BBox(0, 0, 10, 10))],
            summary="1个人",
        )
        # 创建假照片
        (tmp_dir / "img.jpg").write_text("fake image data")

        m._archive(tmp_dir / "img.jpg", det)
        # 应该创建了存档
        archives = list(archive_dir.glob("**/*.jpg"))
        assert len(archives) >= 1
