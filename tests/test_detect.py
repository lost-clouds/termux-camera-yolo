"""测试 detect.py — NMS、COCO 列表、Detector 类（mock ONNX）。"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from camera_yolo_logger.detect import COCO_80, Detector, _download_model, _nms, detect
from camera_yolo_logger.schemas import BBox, Detection, DetectionResult


class TestCOCO80:
    def test_length(self):
        assert len(COCO_80) == 80

    def test_first_is_person(self):
        assert COCO_80[0] == "person"

    def test_last_is_toothbrush(self):
        assert COCO_80[79] == "toothbrush"

    def test_common_classes(self):
        assert "car" in COCO_80
        assert "dog" in COCO_80
        assert "cat" in COCO_80
        assert "cell phone" in COCO_80
        assert "laptop" in COCO_80


class TestNMS:
    def test_empty(self):
        keep = _nms([], [], 0.5, 0.5)
        assert keep == []

    def test_single_detection(self):
        keep = _nms([[0, 0, 100, 100]], [0.9], 0.5, 0.5)
        assert keep == [0]

    def test_below_threshold(self):
        keep = _nms([[0, 0, 100, 100]], [0.3], 0.5, 0.5)
        assert keep == []

    def test_two_separate_boxes(self):
        boxes = [[0, 0, 50, 50], [200, 200, 300, 300]]
        scores = [0.9, 0.8]
        keep = _nms(boxes, scores, 0.5, 0.5)
        assert len(keep) == 2

    def test_two_overlapping_same_class_suppressed(self):
        boxes = [[0, 0, 100, 100], [10, 10, 110, 110]]
        scores = [0.9, 0.8]
        keep = _nms(boxes, scores, 0.5, 0.5, class_ids=[0, 0])
        assert len(keep) == 1

    def test_two_overlapping_different_class_kept(self):
        boxes = [[0, 0, 100, 100], [10, 10, 110, 110]]
        scores = [0.9, 0.8]
        keep = _nms(boxes, scores, 0.5, 0.5, class_ids=[0, 1])
        assert len(keep) == 2


class TestDetector:
    @pytest.fixture
    def detector(self):
        return Detector(conf_thresh=0.5, iou_thresh=0.5)

    def test_init_defaults(self):
        d = Detector()
        assert d.conf_thresh == 0.45
        assert d.iou_thresh == 0.5
        assert d.class_filter is None
        assert d.model_path == Path("yolov8n.onnx")

    def test_init_custom(self):
        d = Detector(
            model_path="custom.onnx",
            model_download_url="https://example.com/model.onnx",
            conf_thresh=0.7,
            iou_thresh=0.3,
            class_filter=["person", "cat"],
        )
        assert d.model_path == Path("custom.onnx")
        assert d.model_download_url == "https://example.com/model.onnx"
        assert d.conf_thresh == 0.7
        assert d.iou_thresh == 0.3
        assert d.class_filter == ["person", "cat"]

    def test_from_settings(self):
        from camera_yolo_logger.config import Settings
        s = Settings(
            model_path="settings_model.onnx",
            model_download_url="http://dl.example.com",
            conf_thresh=0.55,
            iou_thresh=0.45,
            class_filter=["dog"],
        )
        d = Detector.from_settings(s)
        assert d.model_path == Path("settings_model.onnx")
        assert d.conf_thresh == 0.55
        assert d.class_filter == ["dog"]

    def test_detect_model_not_found(self, detector):
        detector.model_path = Path("/nonexistent/model.onnx")
        result = detector.detect("fake_image.jpg")
        assert result.success is False
        assert "模型文件不存在" in result.summary

    def test_detect_with_mock_onnx(self, tmp_dir, sample_image):
        """模拟 ONNX Runtime 返回数据，验证完整检测流水线。"""
        # 创建假的模型文件使 exists() 通过
        fake_model = tmp_dir / "fake.onnx"
        fake_model.write_text("dummy")
        detector = Detector(conf_thresh=0.5, iou_thresh=0.5)
        detector._get_model_path = lambda: fake_model

        with patch("camera_yolo_logger.detect.Detector._get_session") as mock_sess:
            sess = MagicMock()
            sess.get_inputs.return_value = [MagicMock(name="images")]
            sess.get_inputs.return_value[0].name = "images"
            # 模拟 YOLOv8 输出: (1, 84, 8400)
            fake_output = np.zeros((1, 84, 8400), dtype=np.float32)
            fake_output[0, :4, 0] = [0.5, 0.5, 0.1, 0.1]  # cx,cy,w,h (归一化)
            fake_output[0, 4 + 0, 0] = 0.95  # person score
            sess.run.return_value = [fake_output]
            mock_sess.return_value = sess

            result = detector.detect(str(sample_image))
        assert result.success is True
        assert len(result.objects) > 0

    def test_detect_no_objects_found(self, tmp_dir, sample_image):
        """模拟 ONNX 返回无高置信度检测。"""
        fake_model = tmp_dir / "fake.onnx"
        fake_model.write_text("dummy")
        detector = Detector(conf_thresh=0.9, iou_thresh=0.5)
        detector._get_model_path = lambda: fake_model

        with patch("camera_yolo_logger.detect.Detector._get_session") as mock_sess:
            sess = MagicMock()
            sess.get_inputs.return_value = [MagicMock(name="images")]
            sess.get_inputs.return_value[0].name = "images"
            fake_output = np.zeros((1, 84, 8400), dtype=np.float32)
            sess.run.return_value = [fake_output]
            mock_sess.return_value = sess

            result = detector.detect(str(sample_image))
        assert result.success is True
        assert len(result.objects) == 0
        assert "未识别" in result.summary

    def test_detect_summary_backwards_compat(self, tmp_dir, sample_image):
        """detect_summary() 返回字符串（向后兼容）。"""
        detector = Detector(conf_thresh=0.9, iou_thresh=0.5)
        detector._get_model_path = lambda: Path("fake.onnx")

        with patch("camera_yolo_logger.detect.Detector._get_session") as mock_sess:
            sess = MagicMock()
            sess.get_inputs.return_value = [MagicMock(name="images")]
            sess.get_inputs.return_value[0].name = "images"
            fake_output = np.zeros((1, 84, 8400), dtype=np.float32)
            sess.run.return_value = [fake_output]
            mock_sess.return_value = sess

            summary = detector.detect_summary(str(sample_image))
        assert isinstance(summary, str)
        assert len(summary) > 0


class TestModuleLevelDetect:
    def test_returns_string_backwards_compat(self, tmp_dir, sample_image):
        """模块级 detect() 返回字符串（向后兼容）。"""
        with patch("camera_yolo_logger.detect.Detector._get_model_path") as mock_path:
            mock_path.return_value = Path("/nonexistent/model.onnx")
            result = detect(str(sample_image))
        assert isinstance(result, str)


# 需要 mock import
from unittest.mock import MagicMock, patch
