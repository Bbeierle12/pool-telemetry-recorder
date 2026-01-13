"""Tests for video source utilities."""

from __future__ import annotations

import pytest

from pool_telemetry.video.sources import (
    RESOLUTION_PRESETS,
    VideoSettings,
    is_wifi_source,
    parse_resolution,
)


class TestParseResolution:
    """Tests for parse_resolution function."""

    def test_parses_1080p(self) -> None:
        result = parse_resolution("1080p")
        assert result == (1920, 1080)

    def test_parses_720p(self) -> None:
        result = parse_resolution("720p")
        assert result == (1280, 720)

    def test_parses_4k(self) -> None:
        result = parse_resolution("4k")
        assert result == (3840, 2160)

    def test_parses_4k_uppercase(self) -> None:
        result = parse_resolution("4K")
        assert result == (3840, 2160)

    def test_parses_2_7k(self) -> None:
        result = parse_resolution("2.7k")
        assert result == (2704, 1520)

    def test_parses_2_7k_uppercase(self) -> None:
        result = parse_resolution("2.7K")
        assert result == (2704, 1520)

    def test_parses_480p(self) -> None:
        result = parse_resolution("480p")
        assert result == (854, 480)

    def test_parses_1440p(self) -> None:
        result = parse_resolution("1440p")
        assert result == (2560, 1440)

    def test_parses_custom_resolution(self) -> None:
        result = parse_resolution("1280x720")
        assert result == (1280, 720)

    def test_parses_custom_resolution_uppercase(self) -> None:
        result = parse_resolution("1920X1080")
        assert result == (1920, 1080)

    def test_parses_custom_resolution_mixed_case(self) -> None:
        result = parse_resolution("3840x2160")
        assert result == (3840, 2160)

    def test_handles_whitespace(self) -> None:
        result = parse_resolution("  1080p  ")
        assert result == (1920, 1080)

    def test_returns_none_for_invalid(self) -> None:
        assert parse_resolution("invalid") is None
        assert parse_resolution("123") is None

    def test_returns_none_for_empty(self) -> None:
        assert parse_resolution("") is None
        assert parse_resolution(None) is None  # type: ignore[arg-type]

    def test_returns_none_for_partial_custom(self) -> None:
        assert parse_resolution("1920x") is None
        assert parse_resolution("x1080") is None
        assert parse_resolution("axb") is None

    def test_returns_none_for_negative_values(self) -> None:
        # These should return None because the parts aren't digits
        assert parse_resolution("-1920x1080") is None


class TestResolutionPresets:
    """Tests for resolution preset constants."""

    def test_presets_exist(self) -> None:
        assert "720p" in RESOLUTION_PRESETS
        assert "1080p" in RESOLUTION_PRESETS
        assert "4k" in RESOLUTION_PRESETS
        assert "2.7k" in RESOLUTION_PRESETS

    def test_preset_values(self) -> None:
        assert RESOLUTION_PRESETS["720p"] == (1280, 720)
        assert RESOLUTION_PRESETS["1080p"] == (1920, 1080)
        assert RESOLUTION_PRESETS["4k"] == (3840, 2160)


class TestVideoSettings:
    """Tests for VideoSettings dataclass."""

    def test_default_values(self) -> None:
        settings = VideoSettings()
        assert settings.resolution is None
        assert settings.framerate is None
        assert settings.stabilization is False
        assert settings.buffer_size == 1

    def test_with_resolution(self) -> None:
        settings = VideoSettings(resolution=(1920, 1080))
        assert settings.resolution == (1920, 1080)

    def test_with_framerate(self) -> None:
        settings = VideoSettings(framerate=60.0)
        assert settings.framerate == 60.0

    def test_with_stabilization(self) -> None:
        settings = VideoSettings(stabilization=True)
        assert settings.stabilization is True

    def test_with_buffer_size(self) -> None:
        settings = VideoSettings(buffer_size=5)
        assert settings.buffer_size == 5

    def test_with_all_options(self) -> None:
        settings = VideoSettings(
            resolution=(1280, 720),
            framerate=30.0,
            stabilization=True,
            buffer_size=3,
        )
        assert settings.resolution == (1280, 720)
        assert settings.framerate == 30.0
        assert settings.stabilization is True
        assert settings.buffer_size == 3


class TestIsWifiSource:
    """Tests for is_wifi_source function."""

    def test_device_index_is_not_wifi(self) -> None:
        assert is_wifi_source(0) is False
        assert is_wifi_source(1) is False
        assert is_wifi_source(10) is False

    def test_file_path_is_not_wifi(self) -> None:
        assert is_wifi_source("/path/to/video.mp4") is False
        assert is_wifi_source("C:\\Videos\\test.mp4") is False
        assert is_wifi_source("video.mkv") is False

    def test_http_url_is_wifi(self) -> None:
        assert is_wifi_source("http://10.5.5.9:8080/live") is True
        assert is_wifi_source("http://192.168.1.1/stream") is True

    def test_https_url_is_wifi(self) -> None:
        assert is_wifi_source("https://camera.local/stream") is True

    def test_rtsp_url_is_wifi(self) -> None:
        assert is_wifi_source("rtsp://10.5.5.9:554/live") is True
        assert is_wifi_source("rtsp://user:pass@camera/stream") is True

    def test_udp_url_is_wifi(self) -> None:
        assert is_wifi_source("udp://10.5.5.9:8554") is True
        assert is_wifi_source("udp://239.255.0.1:1234") is True
