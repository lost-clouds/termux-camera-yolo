"""连续监控模块 — capture→motion-check→detect→archive→log 循环。"""
from __future__ import annotations

import json
import shutil
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from camera_yolo_logger.config import Settings
from camera_yolo_logger.capture import capture, capture_from_url
from camera_yolo_logger.detect import Detector
from camera_yolo_logger.motion import MotionDetector
from camera_yolo_logger.schemas import DetectionResult, MonitorStats


class Monitor:
    """连续监控循环。

    每轮迭代:
      1. 拍照 (含重试)
      2. 运动检测 (如启用) → 跳过 YOLO
      3. YOLO 检测
      4. 存档 (检测到物体时)
      5. Webhook 通知 (匹配触发类别时)
      6. CSV 日志
      7. 自适应休眠

    信号处理: SIGINT / SIGTERM → 优雅退出并打印统计。
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.detector = Detector.from_settings(settings)
        self.motion: MotionDetector | None = None
        if settings.motion_detection:
            self.motion = MotionDetector(settings.motion_resize, settings.motion_threshold)
        self.running = False
        self.stats = MonitorStats()
        self._signal_received = False
        self._log_path = Path(settings.log_file)

        # Webhook (延迟导入)
        self._notifier = None
        if settings.webhook_url:
            from camera_yolo_logger.notify import Notifier
            self._notifier = Notifier(
                webhook_url=settings.webhook_url,
                trigger_classes=settings.webhook_trigger_classes,
                cooldown=settings.webhook_cooldown,
            )

    # ── 主循环 ──────────────────────────────────────────

    def run(self) -> None:
        self.running = True
        self._setup_signals()
        s = self.settings

        print(f"连续监控启动 — 间隔 {s.monitor_interval}s, "
              f"运动检测: {'开' if s.motion_detection else '关'}, "
              f"存档: {'开' if s.archive_enabled else '关'}",
              file=sys.stderr)

        while self.running and not self._signal_received:
            t_iter = time.monotonic()
            self.stats.iterations += 1

            # 1. 拍照
            if s.capture_backend in ("ipwebcam", "url") and s.ip_webcam_url:
                cap_result = capture_from_url(s.ip_webcam_url, s)
            else:
                cap_result = capture(s)
            self.stats.capture_time_ms += cap_result.elapsed_ms

            if not cap_result.success:
                self.stats.skipped_capture += 1
                if s.output_format == "json":
                    self._print_json({"timestamp": self._now_iso(), "status": "capture_failed",
                                      "error": cap_result.error})
                else:
                    print(f"{self._now_str()},拍照失败: {cap_result.error}")
                self._adaptive_sleep(s.monitor_interval)
                continue

            # 2. 运动检测
            if self.motion:
                if not self.motion.has_motion(str(cap_result.path)):
                    self.stats.skipped_motion += 1
                    if s.output_format == "json":
                        self._print_json({"timestamp": self._now_iso(), "status": "no_motion"})
                    self._adaptive_sleep(s.monitor_interval_max)
                    continue
                self.motion.update(str(cap_result.path))

            # 3. YOLO 检测
            det_result, det_elapsed = self._timed_detect(str(cap_result.path))
            det_result.config_snapshot = s.to_snapshot()
            det_result.image_path = str(cap_result.path)
            self.stats.detect_time_ms += det_elapsed

            # 4. 存档
            if det_result.objects and s.archive_enabled:
                self._archive(cap_result.path, det_result)

            # 5. Webhook
            if self._notifier and self._notifier.should_notify(det_result):
                self._notifier.notify(det_result)

            # 6. 日志 & 输出
            self._log_to_csv(det_result)
            if det_result.objects:
                self.stats.detections_total += len(det_result.objects)
                self.stats.last_detection_time = time.time()

            if s.output_format == "json":
                self._print_json({
                    "timestamp": self._now_iso(),
                    "capture": cap_result.to_dict(),
                    "detection": det_result.to_dict(),
                })
            else:
                print(f"{self._now_str()},{det_result.summary}")

            # 7. 自适应休眠
            elapsed = time.monotonic() - t_iter
            sleep_time = self._calc_interval(det_result) - elapsed
            if sleep_time > 0:
                self._interruptible_sleep(sleep_time)

        self._shutdown()

    def stop(self) -> None:
        self.running = False

    # ── 内部方法 ────────────────────────────────────────

    def _calc_interval(self, detection: DetectionResult) -> float:
        s = self.settings
        if detection.objects:
            return s.monitor_interval_min
        if self.motion:
            return s.monitor_interval_max
        return s.monitor_interval

    def _adaptive_sleep(self, base_interval: float) -> None:
        self._interruptible_sleep(base_interval)

    def _interruptible_sleep(self, duration: float) -> None:
        """中断式休眠：每 0.5s 检查信号。"""
        while duration > 0 and self.running and not self._signal_received:
            step = min(0.5, duration)
            time.sleep(step)
            duration -= step

    def _timed_detect(self, image_path: str):
        t0 = time.perf_counter()
        result = self.detector.detect(image_path)
        return result, (time.perf_counter() - t0) * 1000

    def _archive(self, photo_path: Path, detection: DetectionResult) -> None:
        classes = sorted(set(o.class_name for o in detection.objects))
        date_str = datetime.now().strftime("%Y-%m-%d")
        time_str = datetime.now().strftime("%H-%M-%S")
        class_str = "_".join(classes[:5])  # 最多 5 个类名
        dest_dir = Path(self.settings.archive_dir) / date_str
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{time_str}_{class_str}.jpg"
        try:
            shutil.copy2(str(photo_path), str(dest))
            if self.settings.verbose:
                print(f"已存档: {dest}", file=sys.stderr)
        except OSError as exc:
            if self.settings.verbose:
                print(f"存档失败: {exc}", file=sys.stderr)

    def _log_to_csv(self, detection: DetectionResult) -> None:
        import csv
        from camera_yolo_logger.setup import trim_csv
        need_header = not self._log_path.exists() or self._log_path.stat().st_size == 0
        try:
            with open(self._log_path, "a", newline="") as f:
                w = csv.writer(f)
                if need_header:
                    w.writerow(["timestamp", "detected"])
                w.writerow([self._now_str(), detection.summary])
            trim_csv(self._log_path, self.settings.csv_max_records)
        except OSError:
            pass

    def _print_json(self, data: dict) -> None:
        print(json.dumps(data, ensure_ascii=False), flush=True)

    # ── 信号处理 ────────────────────────────────────────

    def _setup_signals(self) -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, self._handle_signal)
            except Exception:
                pass  # 非主线程可能无法注册

    def _handle_signal(self, signum, frame) -> None:
        print(f"\n收到信号 {signum}，正在退出...", file=sys.stderr)
        self._signal_received = True
        self.stop()

    def _shutdown(self) -> None:
        print(f"监控已停止 — {self.stats.to_dict()}", file=sys.stderr)

    # ── 辅助 ────────────────────────────────────────────

    @staticmethod
    def _now_str() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _now_iso() -> str:
        return datetime.now().isoformat()
