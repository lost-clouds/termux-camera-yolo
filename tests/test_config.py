"""测试 config.py — Settings 默认值、环境变量、TOML、CLI 分层加载。"""
from __future__ import annotations

import os
import tempfile
from argparse import Namespace
from pathlib import Path

from camera_yolo_logger.config import Settings, _env_to_dict, _toml_to_dict, _cli_to_dict, load_settings


class TestSettingsDefaults:
    def test_defaults(self):
        s = Settings()
        assert s.photo_cmd == "termux-camera-photo"
        assert s.photo_path == "Image/camera_yolo_temp.jpg"
        assert s.conf_thresh == 0.45
        assert s.iou_thresh == 0.5
        assert s.capture_timeout == 30
        assert s.capture_retry == 3
        assert s.model_path == "yolov8n.onnx"
        assert s.log_file == "camera_log.csv"
        assert s.output_format == "text"
        assert s.motion_detection is False
        assert s.monitor is False
        assert s.server is False
        assert s.class_filter is None
        assert s.webhook_url == ""

    def test_custom_init(self):
        s = Settings(conf_thresh=0.8, output_format="json", class_filter=["person"])
        assert s.conf_thresh == 0.8
        assert s.output_format == "json"
        assert s.class_filter == ["person"]

    def test_to_snapshot(self):
        s = Settings(conf_thresh=0.6, class_filter=["cat"])
        snap = s.to_snapshot()
        assert snap["conf_threshold"] == 0.6
        assert snap["class_filter"] == ["cat"]
        assert snap["model"] == "yolov8n.onnx"
        assert snap["motion_detection"] is False


class TestEnvLoading:
    def test_empty_env(self, monkeypatch):
        for key in list(os.environ):
            if key.startswith(("CAMERA_", "DETECT_", "MODEL_", "MOTION_",
                               "MONITOR_", "WEBBOOK_", "CLASS_", "ARCHIVE_",
                               "IP_WEBCAM", "CAMERA_CAPTURE")):
                monkeypatch.delenv(key, raising=False)
        d = _env_to_dict()
        assert d == {}

    def test_legacy_env_vars(self, monkeypatch):
        monkeypatch.setenv("CAMERA_PHOTO_CMD", "custom-cmd")
        monkeypatch.setenv("DETECT_CONF_THRESH", "0.7")
        monkeypatch.setenv("DETECT_IOU_THRESH", "0.3")
        monkeypatch.setenv("CAMERA_MODEL_PATH", "my_model.onnx")
        monkeypatch.setenv("MODEL_DOWNLOAD_URL", "https://example.com/model.onnx")
        monkeypatch.setenv("CAMERA_LOG_FILE", "mylog.csv")
        monkeypatch.setenv("CAMERA_PHOTO_PATH", "photos/test.jpg")

        d = _env_to_dict()
        assert d["photo_cmd"] == "custom-cmd"
        assert d["conf_thresh"] == 0.7
        assert d["iou_thresh"] == 0.3
        assert d["model_path"] == "my_model.onnx"
        assert d["model_download_url"] == "https://example.com/model.onnx"
        assert d["log_file"] == "mylog.csv"
        assert d["photo_path"] == "photos/test.jpg"

    def test_motion_env_var(self, monkeypatch):
        monkeypatch.setenv("MOTION_ENABLED", "true")
        monkeypatch.setenv("MOTION_THRESHOLD", "0.03")
        d = _env_to_dict()
        assert d["motion_detection"] is True
        assert d["motion_threshold"] == 0.03

    def test_class_filter_env(self, monkeypatch):
        monkeypatch.setenv("CLASS_FILTER", "person,car,dog")
        d = _env_to_dict()
        assert d["class_filter"] == ["person", "car", "dog"]

    def test_webhook_trigger_classes_env(self, monkeypatch):
        monkeypatch.setenv("WEBBOOK_TRIGGER_CLASSES", "person, car")
        d = _env_to_dict()
        assert d["webhook_trigger_classes"] == ["person", "car"]


class TestTOMLoading:
    def test_empty_no_files(self):
        d = _toml_to_dict("")
        assert d == {}

    def test_valid_toml(self, tmp_dir):
        toml_path = tmp_dir / "test.toml"
        toml_path.write_text("""
[capture]
cmd = "test-cmd"
timeout = 15

[detection]
conf_threshold = 0.8
classes = ["person", "cat"]

[motion]
enabled = true
threshold = 0.03

[monitor]
interval = 10.0
archive = true

[webhook]
url = "http://example.com/hook"
trigger_classes = ["person"]

[server]
host = "0.0.0.0"
port = 8080
""")
        d = _toml_to_dict(str(toml_path))
        assert d["photo_cmd"] == "test-cmd"
        assert d["capture_timeout"] == 15
        assert d["conf_thresh"] == 0.8
        assert d["class_filter"] == ["person", "cat"]
        assert d["motion_detection"] is True
        assert d["motion_threshold"] == 0.03
        assert d["monitor_interval"] == 10.0
        assert d["archive_enabled"] is True
        assert d["webhook_url"] == "http://example.com/hook"
        assert d["webhook_trigger_classes"] == ["person"]
        assert d["server_host"] == "0.0.0.0"
        assert d["server_port"] == 8080


class TestCLILoading:
    def test_empty_args(self):
        d = _cli_to_dict(None)
        assert d == {}

    def test_empty_namespace(self):
        d = _cli_to_dict(Namespace())
        assert d == {}

    def test_json_flag(self):
        d = _cli_to_dict(Namespace(json_output=True))
        assert d["output_format"] == "json"

    def test_classes(self):
        d = _cli_to_dict(Namespace(classes=["person", "car"]))
        assert d["class_filter"] == ["person", "car"]

    def test_monitor_with_interval(self):
        d = _cli_to_dict(Namespace(monitor=True, monitor_interval=2.5))
        assert d["monitor"] is True
        assert d["monitor_interval"] == 2.5

    def test_server(self):
        d = _cli_to_dict(Namespace(server=True, server_port=9999))
        assert d["server"] is True
        assert d["server_port"] == 9999


class TestLoadSettings:
    def test_default(self):
        s = load_settings()
        assert isinstance(s, Settings)
        assert s.conf_thresh == 0.45

    def test_env_overrides_default(self, monkeypatch):
        monkeypatch.setenv("DETECT_CONF_THRESH", "0.99")
        s = load_settings()
        assert s.conf_thresh == 0.99

    def test_cli_overrides_env(self, monkeypatch):
        monkeypatch.setenv("DETECT_CONF_THRESH", "0.5")
        s = load_settings(cli_args=Namespace(conf_thresh=0.88))
        assert s.conf_thresh == 0.88

    def test_full_priority_chain(self, tmp_dir, monkeypatch):
        """TOML → Env → CLI 的完整优先级。"""
        toml_path = tmp_dir / "cfg.toml"
        toml_path.write_text('[detection]\nconf_threshold = 0.3')
        monkeypatch.setenv("DETECT_CONF_THRESH", "0.6")

        # CLI should win
        s = load_settings(config_path=str(toml_path),
                          cli_args=Namespace(conf_thresh=0.9))
        assert s.conf_thresh == 0.9
