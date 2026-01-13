from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import cv2


Resolution = Tuple[int, int]


@dataclass
class VideoSettings:
    resolution: Optional[Resolution] = None
    framerate: Optional[float] = None


VideoSource = Union[int, str]


def open_capture(source: VideoSource, settings: Optional[VideoSettings] = None) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(source)
    if settings and isinstance(source, int):
        if settings.resolution:
            width, height = settings.resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        if settings.framerate:
            cap.set(cv2.CAP_PROP_FPS, float(settings.framerate))
    return cap


def parse_resolution(value: str) -> Optional[Resolution]:
    lookup = {
        "1080p": (1920, 1080),
        "720p": (1280, 720),
        "4k": (3840, 2160),
    }
    if value in lookup:
        return lookup[value]
    if "x" in value:
        parts = value.lower().split("x")
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            return int(parts[0]), int(parts[1])
    return None
