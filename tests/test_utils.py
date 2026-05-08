"""测试 utils.py — FileLock, timed 装饰器。"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from camera_yolo_logger.utils import FileLock, timed


class TestFileLock:
    def test_acquire_release(self, tmp_dir):
        lock = FileLock(tmp_dir / "test.jpg")
        assert lock.acquire() is True
        lock.release()

    def test_context_manager(self, tmp_dir):
        with FileLock(tmp_dir / "test.jpg") as _lock:
            pass  # 正常获取和释放

    def test_concurrent_lock_blocked(self, tmp_dir):
        """同一文件的第二个锁应超时失败。"""
        lock1 = FileLock(tmp_dir / "test.jpg")
        assert lock1.acquire() is True
        try:
            lock2 = FileLock(tmp_dir / "test.jpg", timeout=0.5)
            with pytest.raises(TimeoutError):
                with lock2:
                    pass
        finally:
            lock1.release()

    def test_lockfile_created(self, tmp_dir):
        lock_path = tmp_dir / "photo.jpg"
        with FileLock(lock_path):
            lockfile = Path(str(lock_path) + ".lock")
            assert lockfile.exists()
        # 释放后 lockfile 仍然存在（但已解锁）

    def test_different_files_no_conflict(self, tmp_dir):
        """不同文件路径的锁不应冲突。"""
        lock1 = FileLock(tmp_dir / "a.jpg")
        lock2 = FileLock(tmp_dir / "b.jpg")
        assert lock1.acquire() is True
        try:
            assert lock2.acquire() is True
        finally:
            lock2.release()
        lock1.release()

    def test_release_twice_safe(self, tmp_dir):
        lock = FileLock(tmp_dir / "test.jpg")
        lock.acquire()
        lock.release()
        lock.release()  # 不应抛出异常


class TestTimed:
    def test_returns_result_and_elapsed(self):
        @timed
        def fast():
            return 42

        result, elapsed = fast()
        assert result == 42
        assert elapsed >= 0

    def test_elapsed_increases_with_sleep(self):
        @timed
        def slow():
            time.sleep(0.05)
            return "done"

        result, elapsed = slow()
        assert result == "done"
        assert elapsed >= 40  # milliseconds

    def test_preserves_function_name(self):
        @timed
        def my_func():
            pass

        assert my_func.__name__ == "my_func"
