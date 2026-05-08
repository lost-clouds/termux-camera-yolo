"""测试 motion.py — MotionDetector 帧差分运动检测。"""
from __future__ import annotations

import numpy as np
import pytest

from camera_yolo_logger.motion import MotionDetector


class TestMotionDetector:
    @pytest.fixture
    def md(self):
        return MotionDetector(resize=160, threshold=0.05)

    def test_first_frame_score_zero(self, md, sample_image):
        """首帧运动分数为 0，自动设为参考帧。"""
        score = md.score(sample_image)
        assert score == 0.0

    def test_first_frame_no_motion(self, md, sample_image):
        assert md.has_motion(sample_image) is False

    def test_identical_frames_no_motion(self, md, sample_image):
        md.score(sample_image)  # set reference
        score = md.score(sample_image)
        assert score == 0.0
        assert md.has_motion(sample_image) is False

    def test_different_frames_motion_detected(self, md, sample_image, sample_image2):
        """两帧不同（红 vs 绿）→ 检测到运动。"""
        md.score(sample_image)  # set reference = 红色
        score = md.score(sample_image2)  # compare with 绿色
        assert score > 0.0
        assert md.has_motion(sample_image2) is True

    def test_update_changes_reference(self, md, sample_image, sample_image2):
        """update() 更新参考帧后，相同帧不再有运动。"""
        md.score(sample_image)  # ref = 红色
        assert md.has_motion(sample_image2) is True  # 绿色 vs 红色 → 运动
        md.update(sample_image2)  # ref = 绿色
        assert md.has_motion(sample_image2) is False  # 绿色 vs 绿色 → 无运动

    def test_reset_clears_reference(self, md, sample_image):
        md.score(sample_image)  # set reference
        md.reset()
        # 重置后再次调用 score 会设为新参考
        score = md.score(sample_image)
        assert score == 0.0

    def test_custom_threshold(self, md, sample_image, sample_image2):
        """高阈值下即使帧不同也可能不触发运动。"""
        md.score(sample_image)
        # 红 vs 绿 MSE 很大，设置极高阈值
        assert md.has_motion(sample_image2, threshold=99999.0) is False

    def test_different_shape_auto_reset(self, md, tmp_dir):
        """不同尺寸的帧自动重置参考。"""
        from PIL import Image
        # 100x100
        arr1 = np.zeros((100, 100, 3), dtype=np.uint8)
        p1 = tmp_dir / "s1.jpg"
        Image.fromarray(arr1).save(str(p1))
        # 200x200
        arr2 = np.zeros((200, 200, 3), dtype=np.uint8)
        p2 = tmp_dir / "s2.jpg"
        Image.fromarray(arr2).save(str(p2))

        md.score(p1)  # reference = 100x100
        # 不同尺寸 → 自动重置
        score = md.score(p2)
        assert score == 0.0
