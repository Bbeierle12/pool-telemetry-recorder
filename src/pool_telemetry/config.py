from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def default_data_dir() -> Path:
    return Path.home() / ".pool_telemetry"


def default_config_path() -> Path:
    return default_data_dir() / "config.json"


@dataclass
class ApiKeys:
    gemini: Optional[str] = None
    anthropic: Optional[str] = None


@dataclass
class GoProConfig:
    connection_mode: str = "usb_webcam"
    wifi_ip: Optional[str] = None
    resolution: str = "1080p"
    framerate: int = 60
    stabilization: bool = True


@dataclass
class VideoImportConfig:
    default_directory: str = "~/Videos"
    supported_formats: List[str] = field(default_factory=lambda: ["mp4", "mov", "mkv", "avi"])
    max_file_size_gb: int = 10


@dataclass
class GeminiConfig:
    model: str = "gemini-2.0-flash-live"
    frame_sample_rate_ms: int = 33
    reconnect_attempts: int = 3
    reconnect_delay_ms: int = 1000
    system_prompt: str = (
        "You are a pool telemetry extractor. Return structured JSON events for shots, "
        "collisions, cushions, pockets, fouls, and table state updates."
    )


@dataclass
class StorageConfig:
    data_directory: str = str(default_data_dir())
    save_key_frames: bool = True
    save_raw_events: bool = True
    frame_quality: int = 85
    max_storage_gb: int = 50
    auto_cleanup_days: int = 90


@dataclass
class UiConfig:
    theme: str = "dark"
    video_preview_size: str = "medium"
    show_raw_json: bool = False
    show_trajectory_overlay: bool = True
    event_log_max_lines: int = 500


@dataclass
class CostTrackingConfig:
    enabled: bool = True
    warn_threshold_usd: float = 5.0
    stop_threshold_usd: Optional[float] = None


@dataclass
class AppConfig:
    version: str = "1.0.0"
    api_keys: ApiKeys = field(default_factory=ApiKeys)
    gopro: GoProConfig = field(default_factory=GoProConfig)
    video_import: VideoImportConfig = field(default_factory=VideoImportConfig)
    gemini: GeminiConfig = field(default_factory=GeminiConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    ui: UiConfig = field(default_factory=UiConfig)
    cost_tracking: CostTrackingConfig = field(default_factory=CostTrackingConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _deep_merge(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _config_from_dict(data: Dict[str, Any]) -> AppConfig:
    defaults = AppConfig().to_dict()
    merged = _deep_merge(defaults, data)
    return AppConfig(
        version=merged["version"],
        api_keys=ApiKeys(**merged["api_keys"]),
        gopro=GoProConfig(**merged["gopro"]),
        video_import=VideoImportConfig(**merged["video_import"]),
        gemini=GeminiConfig(**merged["gemini"]),
        storage=StorageConfig(**merged["storage"]),
        ui=UiConfig(**merged["ui"]),
        cost_tracking=CostTrackingConfig(**merged["cost_tracking"]),
    )


def load_config(path: Optional[Path] = None) -> AppConfig:
    config_path = path or default_config_path()
    if not config_path.exists():
        return AppConfig()
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    return _config_from_dict(raw)


def save_config(config: AppConfig, path: Optional[Path] = None) -> Path:
    config_path = path or default_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config.to_dict(), indent=2), encoding="utf-8")
    return config_path
