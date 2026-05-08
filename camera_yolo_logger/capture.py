"""拍照模块 — termux-camera-photo / IP Webcam, 超时重试, FileLock 并发保护。"""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

from camera_yolo_logger.config import Settings
from camera_yolo_logger.schemas import CaptureResult
from camera_yolo_logger.utils import FileLock


def _single_capture(cmd: str, path: Path, timeout: int) -> CaptureResult:
    """单次拍照，返回 CaptureResult。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        r = subprocess.run(
            [cmd, str(path)],
            capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return CaptureResult(success=False, error=f"拍照超时 ({timeout}s)")
    except FileNotFoundError:
        return CaptureResult(success=False, error=f"命令不存在: {cmd}")
    except Exception as exc:
        return CaptureResult(success=False, error=f"拍照异常: {exc}")

    if r.returncode != 0:
        err = r.stderr.strip() or r.stdout.strip() or f"exit code {r.returncode}"
        return CaptureResult(success=False, error=err)

    try:
        from PIL import Image
        with Image.open(str(path)) as im:
            w, h = im.size
        return CaptureResult(success=True, path=path, image_size=(w, h))
    except Exception:
        # 文件可能已写入但无法读取尺寸
        return CaptureResult(success=True, path=path)


def capture(settings: Settings | None = None) -> CaptureResult:
    """拍照（含超时 + 重试 + FileLock）。

    向后兼容旧接口：返回 CaptureResult dataclass（旧代码访问 .path 和 None 判断仍可工作）。
    """
    s = settings or Settings()
    path = Path(s.photo_path)
    max_retries = max(1, s.capture_retry)
    lock = FileLock(path, timeout=10.0)

    last_error = ""

    for attempt in range(max_retries):
        if attempt > 0:
            delay = s.capture_retry_delay * (2 ** (attempt - 1))
            time.sleep(delay)

        t0 = time.perf_counter()
        try:
            with lock:
                result = _single_capture(s.photo_cmd, path, s.capture_timeout)
        except TimeoutError:
            last_error = "无法获取拍照锁（可能有其他进程正在使用相机）"
            continue

        result.elapsed_ms = (time.perf_counter() - t0) * 1000
        if result.success:
            return result
        last_error = result.error or "未知错误"

    return CaptureResult(success=False, error=last_error or "拍照失败（已达最大重试次数）")


def capture_from_url(url: str, settings: Settings | None = None) -> CaptureResult:
    """从 HTTP URL 获取帧（IP Webcam / 网络摄像头后端）。

    URL 示例:
        http://192.168.1.100:8080/shot.jpg    (IP Webcam 单帧)
        http://192.168.1.100:8080/photo.jpg   (其他网络摄像头)
    """
    s = settings or Settings()
    path = Path(s.photo_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    try:
        from urllib.request import urlopen, Request
        req = Request(url, headers={"User-Agent": "camera-yolo-logger/1.1"})
        with urlopen(req, timeout=10) as resp:
            data = resp.read()
        path.write_bytes(data)
        try:
            from PIL import Image
            with Image.open(str(path)) as im:
                w, h = im.size
            return CaptureResult(success=True, path=path, image_size=(w, h),
                                 elapsed_ms=(time.perf_counter() - t0) * 1000)
        except Exception:
            return CaptureResult(success=True, path=path,
                                 elapsed_ms=(time.perf_counter() - t0) * 1000)
    except Exception as exc:
        return CaptureResult(success=False,
                             error=f"URL 抓取失败: {exc}",
                             elapsed_ms=(time.perf_counter() - t0) * 1000)


def cleanup_photo(settings: Settings | None = None) -> None:
    """删除最新照片（供需要清理的场景使用）。"""
    s = settings or Settings()
    Path(s.photo_path).unlink(missing_ok=True)
