"""CLI 入口 — 单一命令：camera-yolo [flags]

模式:
    camera-yolo                   一次性检测，text 输出（向后兼容）
    camera-yolo --json            一次性检测，JSON 输出
    camera-yolo --monitor         连续监控模式
    camera-yolo --server          启动 HTTP API 服务器
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from camera_yolo_logger.config import Settings, load_settings
from camera_yolo_logger.capture import capture, capture_from_url
from camera_yolo_logger.detect import Detector
from camera_yolo_logger.schemas import DetectionResult, CaptureResult


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="camera-yolo",
        description="Termux 摄像头 YOLO 物体检测 — 拍照 → 识别 → 记录",
    )
    # 配置文件
    p.add_argument("--config", "-c", help="TOML 配置文件路径")

    # 拍照
    cap = p.add_argument_group("拍照")
    cap.add_argument("--photo-cmd", help="拍照命令 (默认: termux-camera-photo)")
    cap.add_argument("--photo-path", help="照片保存路径 (默认: Image/camera_yolo_temp.jpg)")
    cap.add_argument("--capture-timeout", type=int, help="拍照超时秒数")
    cap.add_argument("--capture-retry", type=int, help="拍照重试次数")
    cap.add_argument("--capture-backend", choices=["termux", "ipwebcam", "url"],
                     help="拍照后端: termux / ipwebcam / url")
    cap.add_argument("--ip-webcam-url", help="IP Webcam 抓帧 URL")

    # 检测
    det = p.add_argument_group("检测")
    det.add_argument("--model", help="ONNX 模型路径")
    det.add_argument("--model-download-url", help="模型下载 URL")
    det.add_argument("--confidence", "--conf", type=float, help="置信度阈值")
    det.add_argument("--iou", type=float, help="NMS IoU 阈值")
    det.add_argument("--classes", nargs="+", help="类过滤 (如: person car dog)")

    # 输出
    out = p.add_argument_group("输出")
    out.add_argument("--json", dest="json_output", action="store_true", help="JSON 格式输出")
    out.add_argument("--log-file", help="CSV 日志文件路径")
    out.add_argument("--verbose", "-v", action="store_true", help="详细输出到 stderr")

    # 运动检测
    mot = p.add_argument_group("运动检测 (功耗优化)")
    mot.add_argument("--motion", dest="motion_detection", action="store_true",
                     help="启用运动检测预过滤")
    mot.add_argument("--motion-threshold", type=float, help="运动检测灵敏度 (MSE, 默认 0.05)")
    mot.add_argument("--motion-resize", type=int, help="运动检测缩放尺寸")

    # 连续监控
    mon = p.add_argument_group("连续监控")
    mon.add_argument("--monitor", action="store_true", help="启用连续监控模式")
    mon.add_argument("--interval", type=float, help="监控间隔 (秒)")
    mon.add_argument("--interval-min", type=float, help="检测到物体时的最小间隔")
    mon.add_argument("--interval-max", type=float, help="无运动时的最大间隔")
    mon.add_argument("--archive", dest="archive_enabled", action="store_true",
                     help="检测到物体时存档照片")
    mon.add_argument("--archive-dir", help="存档目录")

    # Webhook
    wh = p.add_argument_group("Webhook 通知")
    wh.add_argument("--webhook-url", help="Webhook URL (检测到物体时 POST JSON)")
    wh.add_argument("--webhook-trigger-classes", nargs="+", help="触发 Webhook 的目标类别")
    wh.add_argument("--webhook-cooldown", type=float, help="通知冷却时间 (秒)")

    # 服务器
    srv = p.add_argument_group("HTTP 服务器")
    srv.add_argument("--server", action="store_true", help="启动 HTTP API 服务器")
    srv.add_argument("--server-host", help="监听地址")
    srv.add_argument("--server-port", type=int, help="监听端口")

    return p


# ── 输出格式化 ─────────────────────────────────────────────


def _format_text(ts: str, result: DetectionResult) -> str:
    return f"{ts},{result.summary}"


def _format_json(ts: str, capture_result: CaptureResult,
                 detect_result: DetectionResult) -> str:
    return json.dumps({
        "version": "1.1.0",
        "timestamp": ts,
        "capture": capture_result.to_dict(),
        "detection": detect_result.to_dict(),
    }, ensure_ascii=False)


# ── CSV 日志 ────────────────────────────────────────────────


def _log_to_csv(ts: str, description: str, log_file: Path) -> None:
    need_header = not log_file.exists() or log_file.stat().st_size == 0
    with open(log_file, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(["timestamp", "detected"])
        w.writerow([ts, description])


# ── 一次性检测 ─────────────────────────────────────────────


def run_once(settings: Settings) -> int:
    """执行单次检测：拍照 → 识别 → 输出。返回 exit code。"""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    # 1. 拍照
    if settings.capture_backend in ("ipwebcam", "url") and settings.ip_webcam_url:
        cap_result = capture_from_url(settings.ip_webcam_url, settings)
    else:
        cap_result = capture(settings)

    if not cap_result.success:
        err = cap_result.error or "拍照失败"
        if settings.output_format == "json":
            print(_format_json(ts, cap_result,
                               DetectionResult(success=False, objects=[], summary=err, error=err)))
        else:
            print(f"{ts},拍照失败: {err}")
        _log_to_csv(ts, f"拍照失败: {err}", Path(settings.log_file))
        return 1

    # 2. 运动预过滤
    if settings.motion_detection:
        from camera_yolo_logger.motion import MotionDetector
        md = MotionDetector(settings.motion_resize, settings.motion_threshold)
        if not md.has_motion(str(cap_result.path)):
            if settings.output_format == "json":
                print(json.dumps({"timestamp": ts, "status": "no_motion"}, ensure_ascii=False))
            else:
                print(f"{ts},无运动")
            _log_to_csv(ts, "无运动", Path(settings.log_file))
            return 0
        md.update(str(cap_result.path))

    # 3. YOLO 检测
    detector = Detector.from_settings(settings)
    det_result = detector.detect(str(cap_result.path))
    det_result.config_snapshot = settings.to_snapshot()

    # 4. 输出
    if settings.output_format == "json":
        print(_format_json(ts, cap_result, det_result))
    else:
        print(_format_text(ts, det_result))
    _log_to_csv(ts, det_result.summary, Path(settings.log_file))

    return 0


# ── 主入口 ──────────────────────────────────────────────────


def main() -> int:
    args = build_parser().parse_args()
    config_path = args.config_file if hasattr(args, "config_file") else None
    settings = load_settings(config_path=config_path, cli_args=args)

    # 兼容旧版环境变量：如果 CAMERA_LOG_FILE 被设置但 CLI 没有覆盖
    if os.environ.get("CAMERA_LOG_FILE") and not getattr(args, "log_file", None):
        settings.log_file = os.environ["CAMERA_LOG_FILE"]

    if settings.verbose:
        import logging
        logging.basicConfig(level=logging.DEBUG, format="%(levelname)s: %(message)s",
                            stream=sys.stderr)

    if settings.server:
        from camera_yolo_logger.server import run_server
        run_server(settings)
        return 0

    if settings.monitor:
        from camera_yolo_logger.monitor import Monitor
        monitor = Monitor(settings)
        monitor.run()
        return 0

    # 默认：一次性检测
    return run_once(settings)


if __name__ == "__main__":
    raise SystemExit(main())
