"""
core/telemetry.py
=================
Async, non-blocking telemetry logger for the BFMC 2026 autonomous stack.

  TelemetryLogger.log(**fields)      — queues one CSV row (rate-limited to 1 Hz)
  TelemetryLogger.start_recording()  — opens cv2.VideoWriter in background thread
  TelemetryLogger.write_frame(frame) — non-blocking frame enqueue
  TelemetryLogger.stop_recording()   — gracefully drains + releases VideoWriter
  TelemetryLogger.stop()             — full shutdown (call in on_close)

All I/O runs in a single daemon worker thread so the 20 Hz control loop
is never blocked by disk writes.
"""

import csv
import os
import queue
import threading
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from config import (
    LOG_DIRECTORY,
    LOG_CSV_INTERVAL_S,
    LOG_CSV_FIELDS,
    LOG_VIDEO_CODEC,
    LOG_VIDEO_FPS,
    LOG_VIDEO_RES,
    LOG_MAX_QUEUE_SIZE,
)

_SENTINEL = object()   # poison pill for worker shutdown


class TelemetryLogger:
    def __init__(self):
        os.makedirs(LOG_DIRECTORY, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._csv_path   = os.path.join(LOG_DIRECTORY, f"telemetry_{ts}.csv")
        self._csv_queue: queue.Queue = queue.Queue(maxsize=LOG_MAX_QUEUE_SIZE)
        self._frame_queue: queue.Queue = queue.Queue(maxsize=512)

        self._last_log_time = 0.0
        self._video_writer: Optional[cv2.VideoWriter] = None
        self._video_lock  = threading.Lock()
        self._recording   = False
        self._video_path  = ""

        # Write CSV header immediately (synchronous — happens once at startup)
        with open(self._csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=LOG_CSV_FIELDS, extrasaction="ignore")
            writer.writeheader()

        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True, name="telemetry-worker")
        self._worker_thread.start()

    # ─────────────────────────────────────────────────────────
    #  Public API
    # ─────────────────────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        return self._recording

    @property
    def csv_path(self) -> str:
        return self._csv_path

    def log(self, **fields) -> None:
        """Queue a telemetry row. Drops silently if rate limit or queue is full."""
        now = time.monotonic()
        if now - self._last_log_time < LOG_CSV_INTERVAL_S:
            return
        self._last_log_time = now

        row = {"timestamp": datetime.now().strftime("%H:%M:%S.%f")[:-3]}
        row.update(fields)

        try:
            self._csv_queue.put_nowait(row)
        except queue.Full:
            pass  # drop — never block the control loop

    def start_recording(self, path: str = "") -> str:
        """Open a new VideoWriter. Returns the file path being written to."""
        with self._video_lock:
            if self._recording:
                return self._video_path   # already running

            if not path:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(LOG_DIRECTORY, f"camera_{ts}.avi")

            fourcc = cv2.VideoWriter_fourcc(*LOG_VIDEO_CODEC)
            w, h   = LOG_VIDEO_RES
            writer = cv2.VideoWriter(path, fourcc, LOG_VIDEO_FPS, (w, h))
            if not writer.isOpened():
                raise RuntimeError(f"[Telemetry] Could not open VideoWriter at {path}")

            self._video_writer = writer
            self._video_path   = path
            self._recording    = True
            return path

    def write_frame(self, frame: np.ndarray) -> None:
        """Non-blocking frame enqueue. Drops if queue is full or not recording."""
        if not self._recording:
            return
        # Resize to recording resolution if the frame doesn't match
        h, w = LOG_VIDEO_RES[1], LOG_VIDEO_RES[0]
        if frame.shape[1] != w or frame.shape[0] != h:
            frame = cv2.resize(frame, (w, h))
        try:
            self._frame_queue.put_nowait(frame.copy())
        except queue.Full:
            pass

    def stop_recording(self) -> None:
        """Gracefully drain the frame queue and release the VideoWriter."""
        with self._video_lock:
            self._recording = False

        # Drain remaining frames before releasing (best-effort, max 2 s)
        deadline = time.monotonic() + 2.0
        while not self._frame_queue.empty() and time.monotonic() < deadline:
            time.sleep(0.01)

        with self._video_lock:
            if self._video_writer is not None:
                self._video_writer.release()
                self._video_writer = None

    def stop(self) -> None:
        """Full shutdown: stop recording, poison the worker, join thread."""
        self.stop_recording()
        self._stop_event.set()
        try:
            self._csv_queue.put_nowait(_SENTINEL)
        except queue.Full:
            pass
        self._worker_thread.join(timeout=5.0)

    # ─────────────────────────────────────────────────────────
    #  Background worker (single daemon thread)
    # ─────────────────────────────────────────────────────────

    def _worker(self) -> None:
        with open(self._csv_path, "a", newline="") as f:
            csv_writer = csv.DictWriter(f, fieldnames=LOG_CSV_FIELDS, extrasaction="ignore")

            while not self._stop_event.is_set():
                # ── CSV drain ────────────────────────────────
                try:
                    item = self._csv_queue.get(timeout=0.05)
                    if item is _SENTINEL:
                        break
                    csv_writer.writerow(item)
                    f.flush()
                except queue.Empty:
                    pass

                # ── Frame drain ──────────────────────────────
                try:
                    frame = self._frame_queue.get_nowait()
                    with self._video_lock:
                        if self._video_writer is not None and self._recording:
                            self._video_writer.write(frame)
                except queue.Empty:
                    pass
