"""HTTP API 服务器 — 为 AI Agent 提供 RESTful 接口。"""
from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, Response, jsonify, request

from camera_yolo_logger.config import Settings
from camera_yolo_logger.capture import capture, capture_from_url
from camera_yolo_logger.detect import Detector

_start_time = time.time()


def create_app(settings: Settings) -> Flask:
    app = Flask(__name__)
    detector = Detector.from_settings(settings)

    @app.route("/detect", methods=["GET", "POST"])
    def detect_endpoint():
        """触发一次拍照 + 检测，返回 JSON 结果。"""
        if settings.capture_backend in ("ipwebcam", "url") and settings.ip_webcam_url:
            cap_result = capture_from_url(settings.ip_webcam_url, settings)
        else:
            cap_result = capture(settings)

        if not cap_result.success:
            return jsonify({"error": cap_result.error, "success": False}), 500

        det_result = detector.detect(str(cap_result.path))
        det_result.config_snapshot = settings.to_snapshot()

        if request.args.get("format") == "summary":
            return jsonify({"result": det_result.summary})

        return jsonify({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "capture": cap_result.to_dict(),
            "detection": det_result.to_dict(),
        })

    @app.route("/status")
    def status():
        """健康检查 + 运行状态。"""
        model_path = Path(settings.model_path)
        return jsonify({
            "status": "ok",
            "version": "1.1.0",
            "model": str(settings.model_path),
            "model_exists": model_path.exists(),
            "model_size_mb": round(model_path.stat().st_size / 1e6, 2) if model_path.exists() else 0,
            "uptime_s": round(time.time() - _start_time, 1),
            "config": settings.to_snapshot(),
        })

    @app.route("/log")
    def log_endpoint():
        """返回最近 CSV 日志（默认 100 条）。"""
        limit = request.args.get("limit", 100, type=int)
        log_path = Path(settings.log_file)
        if not log_path.exists():
            return jsonify({"entries": []})
        entries = []
        try:
            with open(log_path) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    entries.append(row)
            entries = entries[-limit:]
        except Exception:
            pass
        return jsonify({"entries": entries, "total": len(entries)})

    @app.route("/config", methods=["GET"])
    def get_config():
        return jsonify(settings.to_snapshot())

    @app.route("/config", methods=["POST"])
    def update_config():
        """运行时修改配置（conf_thresh, iou_thresh, class_filter）。"""
        data = request.get_json(silent=True) or {}
        if "confidence" in data:
            detector.conf_thresh = float(data["confidence"])
        if "iou" in data:
            detector.iou_thresh = float(data["iou"])
        if "classes" in data:
            detector.class_filter = data["classes"]
        if "webhook_url" in data:
            settings.webhook_url = data["webhook_url"]
        return jsonify({"status": "updated", "config": settings.to_snapshot()})

    @app.route("/stream")
    def stream():
        """代理 IP Webcam MJPEG 流（需配置 ip_webcam_url）。"""
        if not settings.ip_webcam_url:
            return jsonify({"error": "未配置 IP Webcam URL"}), 400

        stream_url = settings.ip_webcam_url.replace("/shot.jpg", "/video").replace(
            "/photo.jpg", "/video")

        def generate():
            from urllib.request import urlopen
            try:
                resp = urlopen(stream_url, timeout=5)
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    yield chunk
            except Exception as exc:
                yield f"data: Stream error: {exc}\n\n".encode()

        return Response(
            generate(),
            mimetype="multipart/x-mixed-replace; boundary=--boundary",
        )

    return app


def run_server(settings: Settings) -> None:
    """启动 Flask HTTP API 服务器。"""
    app = create_app(settings)
    print(f"Camera-YOLO API 服务器启动: http://{settings.server_host}:{settings.server_port}")
    print(f"  GET  /detect   — 触发一次检测")
    print(f"  GET  /status   — 健康检查")
    print(f"  GET  /log      — CSV 日志")
    print(f"  GET  /stream   — IP Webcam 实时流 (需配置)")
    print(f"  POST /config   — 运行时修改设置")
    app.run(
        host=settings.server_host,
        port=settings.server_port,
        debug=False,
        use_reloader=False,
    )
