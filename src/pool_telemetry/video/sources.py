from __future__ import annotations

from dataclasses import dataclass

import cv2


Resolution = tuple[int, int]


@dataclass
class VideoSettings:
    resolution: Resolution | None = None
    framerate: float | None = None


VideoSource = int | str


def open_capture(source: VideoSource, settings: VideoSettings | None = None) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if settings and isinstance(source, int):
        if settings.resolution:
            width, height = settings.resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        if settings.framerate:
            cap.set(cv2.CAP_PROP_FPS, float(settings.framerate))
    return cap


def parse_resolution(value: str) -> Resolution | None:
    lookup = {
        "1080p": (1920, 1080),
        "720p": (1280, 720),
        "4k": (3840, 2160),
    }
    if value in lookup:
        return lookup[value]
    # Handle custom WxH format (case-insensitive)
    lower_value = value.lower()
    if "x" in lower_value:
        parts = lower_value.split("x")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return int(parts[0]), int(parts[1])
    return None
