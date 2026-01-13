"""Tests for Gemini client configuration."""

from __future__ import annotations

import pytest

from pool_telemetry.gemini.client import GEMINI_BASE_URL, GeminiConnection


class TestGeminiConnection:
    """Tests for GeminiConnection dataclass."""

    def test_default_values(self) -> None:
        conn = GeminiConnection(
            api_key="test-key",
            model="gemini-2.0-flash",
        )
        assert conn.api_key == "test-key"
        assert conn.model == "gemini-2.0-flash"
        assert conn.system_prompt == ""
        assert conn.reconnect_attempts == 3
        assert conn.reconnect_delay_ms == 1000
        assert conn.base_url == GEMINI_BASE_URL

    def test_endpoint_constructed_from_model(self) -> None:
        conn = GeminiConnection(
            api_key="test-key",
            model="gemini-2.0-flash",
        )
        expected = f"{GEMINI_BASE_URL}/gemini-2.0-flash:streamGenerateContent"
        assert conn.endpoint == expected

    def test_endpoint_uses_custom_model(self) -> None:
        conn = GeminiConnection(
            api_key="test-key",
            model="gemini-pro-vision",
        )
        assert "gemini-pro-vision" in conn.endpoint

    def test_endpoint_uses_custom_base_url(self) -> None:
        conn = GeminiConnection(
            api_key="test-key",
            model="test-model",
            base_url="wss://custom.example.com/v1/models",
        )
        assert conn.endpoint == "wss://custom.example.com/v1/models/test-model:streamGenerateContent"

    def test_custom_system_prompt(self) -> None:
        prompt = "You are a helpful assistant."
        conn = GeminiConnection(
            api_key="test-key",
            model="gemini-2.0-flash",
            system_prompt=prompt,
        )
        assert conn.system_prompt == prompt

    def test_custom_reconnect_settings(self) -> None:
        conn = GeminiConnection(
            api_key="test-key",
            model="gemini-2.0-flash",
            reconnect_attempts=5,
            reconnect_delay_ms=2000,
        )
        assert conn.reconnect_attempts == 5
        assert conn.reconnect_delay_ms == 2000


class TestGeminiBaseUrl:
    """Tests for GEMINI_BASE_URL constant."""

    def test_is_websocket_url(self) -> None:
        assert GEMINI_BASE_URL.startswith("wss://")

    def test_is_google_domain(self) -> None:
        assert "googleapis.com" in GEMINI_BASE_URL

    def test_includes_models_path(self) -> None:
        assert "/models" in GEMINI_BASE_URL
