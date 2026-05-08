"""共享工具 — FileLock 进程间锁, timed 装饰器。"""
from __future__ import annotations

import fcntl
import functools
import os
import time
from pathlib import Path


class FileLock:
    """基于 fcntl.flock 的跨进程文件锁（Linux/Android/Termux 可用）。"""

    def __init__(self, lock_path: str | Path, timeout: float = 10.0):
        self._lock_path = Path(lock_path)
        self._timeout = timeout
        self._fd: int | None = None

    def acquire(self) -> bool:
        """获取锁，超时返回 False。锁文件放在 lock_path 同目录下。"""
        lockfile = self._lock_path.with_suffix(self._lock_path.suffix + ".lock")
        lockfile.parent.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self._timeout
        self._fd = os.open(str(lockfile), os.O_CREAT | os.O_RDWR, 0o644)
        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return True
            except BlockingIOError:
                if time.monotonic() > deadline:
                    os.close(self._fd)
                    self._fd = None
                    return False
                time.sleep(0.1)

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            except (OSError, ValueError):
                pass
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def __enter__(self):
        if not self.acquire():
            raise TimeoutError(f"Failed to acquire lock on {self._lock_path} within {self._timeout}s")
        return self

    def __exit__(self, *args):
        self.release()
        return False


def timed(func):
    """装饰器：返回 (result, elapsed_ms) 元组。"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        t0 = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = (time.perf_counter() - t0) * 1000
        return result, elapsed
    return wrapper
