"""Tests for configuration management."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from pool_telemetry.config import (
    ApiKeys,
    AppConfig,
    CostTrackingConfig,
    GeminiConfig,
    GoProConfig,
    StorageConfig,
    UiConfig,
    VideoImportConfig,
    _config_from_dict,
    _deep_merge,
    load_config,
    save_config,
)


class TestDeepMerge:
    """Tests for _deep_merge function."""

    def test_shallow_merge(self) -> None:
        base = {"a": 1, "b": 2}
        updates = {"b": 3, "c": 4}
        result = _deep_merge(base, updates)
        assert result == {"a": 1, "b": 3, "c": 4}

    def test_nested_merge(self) -> None:
        base = {"outer": {"a": 1, "b": 2}}
        updates = {"outer": {"b": 3, "c": 4}}
        result = _deep_merge(base, updates)
        assert result == {"outer": {"a": 1, "b": 3, "c": 4}}

    def test_deep_nested_merge(self) -> None:
        base = {"l1": {"l2": {"a": 1}}}
        updates = {"l1": {"l2": {"b": 2}}}
        result = _deep_merge(base, updates)
        assert result == {"l1": {"l2": {"a": 1, "b": 2}}}

    def test_overwrite_non_dict_with_dict(self) -> None:
        base = {"key": "string_value"}
        updates = {"key": {"nested": "value"}}
        result = _deep_merge(base, updates)
        assert result == {"key": {"nested": "value"}}

    def test_original_unchanged(self) -> None:
        base = {"a": 1}
        updates = {"b": 2}
        _deep_merge(base, updates)
        assert base == {"a": 1}


class TestAppConfig:
    """Tests for AppConfig dataclass."""

    def test_default_values(self) -> None:
        config = AppConfig()
        assert config.version == "1.0.0"
        assert config.api_keys.gemini is None
        assert config.gopro.resolution == "1080p"
        assert config.gemini.model == "gemini-2.0-flash-live"
        assert config.cost_tracking.enabled is True

    def test_to_dict_roundtrip(self) -> None:
        config = AppConfig()
        config.api_keys.gemini = "test-key"
        config.cost_tracking.warn_threshold_usd = 10.0

        data = config.to_dict()
        restored = _config_from_dict(data)

        assert restored.api_keys.gemini == "test-key"
        assert restored.cost_tracking.warn_threshold_usd == 10.0

    def test_partial_config_uses_defaults(self) -> None:
        partial_data = {
            "version": "2.0.0",
            "api_keys": {"gemini": "my-key"},
        }
        config = _config_from_dict(partial_data)
        assert config.version == "2.0.0"
        assert config.api_keys.gemini == "my-key"
        assert config.api_keys.anthropic is None  # default
        assert config.gopro.resolution == "1080p"  # default


class TestLoadSaveConfig:
    """Tests for load_config and save_config functions."""

    def test_save_and_load_config(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        config = AppConfig()
        config.api_keys.gemini = "test-api-key"
        config.storage.data_directory = str(tmp_path / "data")

        save_config(config, config_path)
        assert config_path.exists()

        loaded = load_config(config_path)
        assert loaded.api_keys.gemini == "test-api-key"
        assert loaded.storage.data_directory == str(tmp_path / "data")

    def test_load_nonexistent_returns_defaults(self, tmp_path: Path) -> None:
        config_path = tmp_path / "nonexistent.json"
        config = load_config(config_path)
        assert config.version == "1.0.0"
        assert config.api_keys.gemini is None

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        config_path = tmp_path / "nested" / "dir" / "config.json"
        config = AppConfig()
        save_config(config, config_path)
        assert config_path.exists()

    def test_config_file_is_valid_json(self, tmp_path: Path) -> None:
        config_path = tmp_path / "config.json"
        config = AppConfig()
        save_config(config, config_path)

        with open(config_path) as f:
            data = json.load(f)

        assert "version" in data
        assert "api_keys" in data
        assert "gemini" in data


class TestConfigDataclasses:
    """Tests for individual config dataclasses."""

    def test_api_keys_defaults(self) -> None:
        keys = ApiKeys()
        assert keys.gemini is None
        assert keys.anthropic is None

    def test_gopro_config_defaults(self) -> None:
        gopro = GoProConfig()
        assert gopro.connection_mode == "usb_webcam"
        assert gopro.resolution == "1080p"
        assert gopro.framerate == 60

    def test_video_import_config_defaults(self) -> None:
        video = VideoImportConfig()
        assert "mp4" in video.supported_formats
        assert video.max_file_size_gb == 10

    def test_gemini_config_defaults(self) -> None:
        gemini = GeminiConfig()
        assert gemini.model == "gemini-2.0-flash-live"
        assert gemini.frame_sample_rate_ms == 33
        assert "telemetry" in gemini.system_prompt.lower()

    def test_storage_config_defaults(self) -> None:
        storage = StorageConfig()
        assert storage.save_key_frames is True
        assert storage.frame_quality == 85

    def test_ui_config_defaults(self) -> None:
        ui = UiConfig()
        assert ui.theme == "dark"
        assert ui.event_log_max_lines == 500

    def test_cost_tracking_config_defaults(self) -> None:
        cost = CostTrackingConfig()
        assert cost.enabled is True
        assert cost.warn_threshold_usd == 5.0
        assert cost.stop_threshold_usd is None
