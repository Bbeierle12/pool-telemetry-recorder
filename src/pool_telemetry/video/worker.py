from __future__ import annotations

import time
from typing import Optional

import cv2
from PyQt6 import QtCore

from .sources import VideoSettings, VideoSource, open_capture


class VideoWorker(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(object, float)
    status = QtCore.pyqtSignal(str)
    ended = QtCore.pyqtSignal()

    def __init__(
        self,
        source: VideoSource,
        settings: Optional[VideoSettings] = None,
        target_fps: Optional[float] = None,
        playback_real_time: bool = False,
        parent: Optional[QtCore.QObject] = None,
    ) -> None:
        super().__init__(parent)
        self._source = source
        self._settings = settings
        self._target_fps = target_fps
        self._playback_real_time = playback_real_time
        self._running = True

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        cap = open_capture(self._source, self._settings)
        if not cap.isOpened():
            self.status.emit("Video source failed to open")
            self.ended.emit()
            return

        frame_delay = None
        if self._target_fps and self._target_fps > 0:
            frame_delay = 1.0 / self._target_fps
        elif self._playback_real_time:
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps and fps > 0:
                frame_delay = 1.0 / fps

        self.status.emit("Video source connected")
        while self._running:
            ok, frame = cap.read()
            if not ok:
                break
            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            self.frame_ready.emit(frame, timestamp_ms)
            if frame_delay:
                time.sleep(frame_delay)

        cap.release()
        self.status.emit("Video source stopped")
        self.ended.emit()
