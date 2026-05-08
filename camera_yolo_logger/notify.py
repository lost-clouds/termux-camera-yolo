"""Webhook 通知模块 — 检测到触发类时 POST JSON 到指定 URL。"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone

from camera_yolo_logger.schemas import DetectionResult

logger = logging.getLogger(__name__)


class Notifier:
    """检测到目标物体时通过 Webhook 发送通知。

    用法:
        n = Notifier(
            webhook_url="http://localhost:8080/notify",
            trigger_classes=["person", "car"],
            cooldown=10.0,
        )
        if n.should_notify(detection_result):
            n.notify(detection_result)
    """

    def __init__(
        self,
        webhook_url: str = "",
        trigger_classes: list[str] | None = None,
        cooldown: float = 10.0,
    ):
        self.webhook_url = webhook_url
        self.trigger_classes = set(trigger_classes or [])
        self.cooldown = cooldown
        self._last_notify: float = 0.0

    def should_notify(self, detection: DetectionResult) -> bool:
        """判断是否应发送通知。"""
        if not self.webhook_url:
            return False
        if not detection.objects:
            return False
        if self.trigger_classes:
            detected = {o.class_name for o in detection.objects}
            if not detected & self.trigger_classes:
                return False
        if time.time() - self._last_notify < self.cooldown:
            return False
        return True

    def notify(self, detection: DetectionResult) -> bool:
        """POST JSON 到 Webhook URL。返回是否成功。"""
        payload = json.dumps({
            "event": "detection",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "objects": [o.to_dict() for o in detection.objects],
            "summary": detection.summary,
            "image_path": detection.image_path,
        }, ensure_ascii=False).encode("utf-8")

        try:
            from urllib.request import Request, urlopen
            req = Request(
                self.webhook_url,
                data=payload,
                headers={"Content-Type": "application/json; charset=utf-8"},
            )
            urlopen(req, timeout=10)
            self._last_notify = time.time()
            return True
        except Exception as exc:
            logger.warning(f"Webhook 发送失败: {exc}")
            return False
