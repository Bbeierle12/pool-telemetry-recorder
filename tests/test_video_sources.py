"""Tests for video source utilities."""

from __future__ import annotations

import pytest

from pool_telemetry.video.sources import VideoSettings, parse_resolution


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

    def test_parses_custom_resolution(self) -> None:
        result = parse_resolution("1280x720")
        assert result == (1280, 720)

    def test_parses_custom_resolution_uppercase(self) -> None:
        result = parse_resolution("1920X1080")
        assert result == (1920, 1080)

    def test_parses_custom_resolution_mixed_case(self) -> None:
        result = parse_resolution("3840x2160")
        assert result == (3840, 2160)

    def test_returns_none_for_invalid(self) -> None:
        assert parse_resolution("invalid") is None
        assert parse_resolution("") is None
        assert parse_resolution("123") is None

    def test_returns_none_for_partial_custom(self) -> None:
        assert parse_resolution("1920x") is None
        assert parse_resolution("x1080") is None
        assert parse_resolution("axb") is None

    def test_returns_none_for_negative_values(self) -> None:
        # These should return None because the parts aren't digits
        assert parse_resolution("-1920x1080") is None


class TestVideoSettings:
    """Tests for VideoSettings dataclass."""

    def test_default_values(self) -> None:
        settings = VideoSettings()
        assert settings.resolution is None
        assert settings.framerate is None

    def test_with_resolution(self) -> None:
        settings = VideoSettings(resolution=(1920, 1080))
        assert settings.resolution == (1920, 1080)

    def test_with_framerate(self) -> None:
        settings = VideoSettings(framerate=60.0)
        assert settings.framerate == 60.0

    def test_with_both(self) -> None:
        settings = VideoSettings(resolution=(1280, 720), framerate=30.0)
        assert settings.resolution == (1280, 720)
        assert settings.framerate == 30.0
