"""共享工具 — FileLock 进程间锁, 时间戳, CSV 日志。"""
from __future__ import annotations

import csv
import fcntl
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


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


# ── 时间戳 ──────────────────────────────────────────────


def utc_now_str(fmt: str = "%Y-%m-%d %H:%M:%S") -> str:
    """返回当前 UTC 时间格式化字符串（CSV 日志用）。"""
    return datetime.now(timezone.utc).strftime(fmt)


def utc_now_iso() -> str:
    """返回当前 UTC 时间 ISO 8601 字符串。"""
    return datetime.now(timezone.utc).isoformat()


# ── CSV 日志 ────────────────────────────────────────────


def log_detection_to_csv(
    ts: str,
    description: str,
    log_path: Path,
    max_records: int = 0,
    verbose: bool = False,
) -> None:
    """追加一条检测记录到 CSV 日志文件。

    Args:
        ts: 时间戳字符串。
        description: 检测摘要文本。
        log_path: CSV 文件路径。
        max_records: 写入后裁剪到该条数（0=不裁剪）。
        verbose: 裁剪时输出信息到 stderr。
    """
    need_header = not log_path.exists() or log_path.stat().st_size == 0
    with open(log_path, "a", newline="") as f:
        w = csv.writer(f)
        if need_header:
            w.writerow(["timestamp", "detected"])
        w.writerow([ts, description])
    if max_records > 0:
        removed = trim_csv(log_path, max_records)
        if removed > 0 and verbose:
            print(f"CSV 裁剪: 删除 {removed} 条旧记录", file=sys.stderr)


def trim_csv(log_path: Path, max_records: int) -> int:
    """裁剪 CSV 文件，保留 header + 最近 max_records 条记录。

    Returns:
        删除的记录条数
    """
    if max_records <= 0:
        return 0
    if not log_path.exists():
        return 0

    try:
        with open(log_path, "r", newline="") as f:
            rows = list(csv.reader(f))
    except Exception:
        logger.warning("Failed to read CSV for trimming: %s", log_path)
        return 0

    if len(rows) <= 1:  # 只有 header 或空
        return 0

    header = rows[0]
    data = rows[1:]

    if len(data) <= max_records:
        return 0

    trimmed = data[-max_records:]
    removed = len(data) - len(trimmed)

    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(trimmed)

    return removed
