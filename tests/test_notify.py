"""测试 notify.py — Webhook 通知逻辑、冷却控制。"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from camera_yolo_logger.notify import Notifier
from camera_yolo_logger.schemas import BBox, Detection, DetectionResult


def _make_detection(*classes: str) -> DetectionResult:
    objects = [
        Detection(name, i, 0.95, BBox(0, 0, 10, 10))
        for i, name in enumerate(classes)
    ]
    return DetectionResult(success=True, objects=objects, summary="test")


class TestNotifier:
    def test_should_notify_no_url(self):
        n = Notifier(webhook_url="")
        assert n.should_notify(_make_detection("person")) is False

    def test_should_notify_no_objects(self):
        n = Notifier(webhook_url="http://example.com/hook")
        empty = DetectionResult(success=True, objects=[], summary="无")
        assert n.should_notify(empty) is False

    def test_should_notify_any_object(self):
        """没有 trigger_classes 时，任何检测都触发。"""
        n = Notifier(webhook_url="http://example.com/hook")
        assert n.should_notify(_make_detection("car")) is True

    def test_should_notify_matching_trigger(self):
        n = Notifier(webhook_url="http://example.com/hook",
                     trigger_classes=["person", "car"])
        assert n.should_notify(_make_detection("person")) is True

    def test_should_notify_non_matching_trigger(self):
        n = Notifier(webhook_url="http://example.com/hook",
                     trigger_classes=["person"])
        assert n.should_notify(_make_detection("dog", "cat")) is False

    def test_should_notify_partial_match(self):
        """多个检测中有一个匹配即触发。"""
        n = Notifier(webhook_url="http://example.com/hook",
                     trigger_classes=["person"])
        assert n.should_notify(_make_detection("person", "car")) is True

    def test_cooldown_prevents_notify(self, monkeypatch):
        """冷却时间内不触发。"""
        n = Notifier(webhook_url="http://example.com/hook", cooldown=10.0)
        # 第一次
        assert n.should_notify(_make_detection("person")) is True
        n._last_notify = __import__("time").time()  # 模拟刚发送过
        # 立即第二次 → 冷却中
        assert n.should_notify(_make_detection("person")) is False

    def test_notify_sends_post(self):
        """notify() 发送 POST 请求。"""
        n = Notifier(webhook_url="http://example.com/hook")

        with patch("urllib.request.urlopen") as mock_urlopen:
            mock_resp = MagicMock()
            mock_urlopen.return_value = mock_resp

            success = n.notify(_make_detection("person"))
        assert success is True
        mock_urlopen.assert_called_once()

    def test_notify_network_error(self):
        """网络错误时返回 False。"""
        n = Notifier(webhook_url="http://example.com/hook")

        with patch("urllib.request.urlopen", side_effect=OSError("conn refused")):
            success = n.notify(_make_detection("person"))
        assert success is False
