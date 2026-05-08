"""测试 capture.py — 拍照重试、超时、capture_from_url、cleanup_photo。"""
from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from camera_yolo_logger.capture import (
    _single_capture,
    capture,
    capture_from_url,
    cleanup_photo,
)
from camera_yolo_logger.config import Settings


def _fake_success(cmd, path, timeout):
    """创建模拟成功拍照的 subprocess.CompletedProcess。"""
    return subprocess.CompletedProcess(args=[cmd, str(path)], returncode=0,
                                       stdout="", stderr="")


def _fake_failure(cmd, path, timeout):
    """创建模拟拍照失败的 subprocess.CompletedProcess。"""
    return subprocess.CompletedProcess(args=[cmd, str(path)], returncode=1,
                                       stdout="", stderr="camera error")


class TestSingleCapture:
    def test_success(self, tmp_dir, sample_image):
        path = tmp_dir / "capture.jpg"
        mock_img = MagicMock()
        mock_img.size = (640, 480)
        mock_pil = MagicMock()
        mock_pil.open.return_value.__enter__.return_value = mock_img
        with patch("subprocess.run", side_effect=lambda *a, **kw: _fake_success("cmd", path, 30)):
            with patch("PIL.Image.open", mock_pil.open):
                result = _single_capture("fake-cmd", path, 30)
        assert result.success is True
        assert result.path == path
        assert result.image_size == (640, 480)

    def test_failure_nonzero(self, tmp_dir):
        path = tmp_dir / "capture.jpg"
        with patch("subprocess.run", side_effect=lambda *a, **kw: _fake_failure("cmd", path, 30)):
            result = _single_capture("fake-cmd", path, 30)
        assert result.success is False
        assert "camera error" in (result.error or "")

    def test_timeout(self, tmp_dir):
        path = tmp_dir / "capture.jpg"
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 5)):
            result = _single_capture("fake-cmd", path, 5)
        assert result.success is False
        assert "超时" in (result.error or "")

    def test_file_not_found(self, tmp_dir):
        path = tmp_dir / "capture.jpg"
        with patch("subprocess.run", side_effect=FileNotFoundError):
            result = _single_capture("nonexistent-cmd", path, 30)
        assert result.success is False
        assert "不存在" in (result.error or "")


class TestCaptureWithRetry:
    def test_success_first_attempt(self, tmp_dir):
        path = tmp_dir / "capture.jpg"
        s = Settings(photo_cmd="fake-cmd", photo_path=str(path),
                     capture_retry=3, capture_timeout=10)
        with patch("subprocess.run", side_effect=lambda *a, **kw: _fake_success("cmd", path, 10)):
            with patch("PIL.Image.open", return_value=MagicMock(size=(100, 100))):
                result = capture(s)
        assert result.success is True
        assert result.elapsed_ms >= 0

    def test_retry_on_failure_then_succeed(self, tmp_dir):
        path = tmp_dir / "capture.jpg"
        s = Settings(photo_cmd="fake-cmd", photo_path=str(path),
                     capture_retry=3, capture_timeout=10, capture_retry_delay=0.01)
        call_count = [0]

        def flaky_run(*a, **kw):
            call_count[0] += 1
            if call_count[0] < 3:
                return _fake_failure("cmd", path, 10)
            return _fake_success("cmd", path, 10)

        with patch("subprocess.run", side_effect=flaky_run):
            with patch("PIL.Image.open", return_value=MagicMock(size=(100, 100))):
                result = capture(s)
        assert result.success is True
        assert call_count[0] == 3

    def test_all_retries_fail(self, tmp_dir):
        path = tmp_dir / "capture.jpg"
        s = Settings(photo_cmd="fake-cmd", photo_path=str(path),
                     capture_retry=2, capture_timeout=10, capture_retry_delay=0.01)
        with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("cmd", 5)):
            result = capture(s)
        assert result.success is False


class TestCaptureFromURL:
    def test_success(self, tmp_dir):
        path = tmp_dir / "url_capture.jpg"
        s = Settings(photo_path=str(path))
        with patch("urllib.request.urlopen") as mock_urlopen:
            from PIL import Image
            import io
            import numpy as np
            arr = np.zeros((100, 100, 3), dtype=np.uint8)
            buf = io.BytesIO()
            Image.fromarray(arr).save(buf, format="JPEG")
            buf.seek(0)
            mock_resp = MagicMock()
            mock_resp.read.return_value = buf.read()
            mock_urlopen.return_value.__enter__.return_value = mock_resp

            result = capture_from_url("http://example.com/shot.jpg", s)
        assert result.success is True
        assert result.path == path

    def test_network_error(self, tmp_dir):
        path = tmp_dir / "url_capture.jpg"
        s = Settings(photo_path=str(path))
        with patch("urllib.request.urlopen", side_effect=OSError("network error")):
            result = capture_from_url("http://example.com/shot.jpg", s)
        assert result.success is False
        assert "网络" in (result.error or "") or "network" in (result.error or "").lower() or "抓取失败" in (result.error or "")


class TestCleanupPhoto:
    def test_cleanup_existing(self, tmp_dir):
        path = tmp_dir / "photo.jpg"
        path.write_text("dummy")
        s = Settings(photo_path=str(path))
        assert path.exists()
        cleanup_photo(s)
        assert not path.exists()

    def test_cleanup_nonexistent(self, tmp_dir):
        path = tmp_dir / "nonexistent.jpg"
        s = Settings(photo_path=str(path))
        cleanup_photo(s)  # 不应抛出异常
