"""Tests for database management."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from pool_telemetry.config import AppConfig
from pool_telemetry.db import (
    SCHEMA_STATEMENTS,
    connect,
    get_db_path,
    init_db,
    insert_event,
)


class TestGetDbPath:
    """Tests for get_db_path function."""

    def test_returns_path_in_data_directory(self, tmp_path: Path) -> None:
        config = AppConfig()
        config.storage.data_directory = str(tmp_path)
        path = get_db_path(config)
        assert path == tmp_path / "database.sqlite"

    def test_path_is_pathlib_path(self, tmp_path: Path) -> None:
        config = AppConfig()
        config.storage.data_directory = str(tmp_path)
        path = get_db_path(config)
        assert isinstance(path, Path)


class TestConnect:
    """Tests for connect function."""

    def test_creates_parent_directories(self, tmp_path: Path) -> None:
        db_path = tmp_path / "nested" / "dir" / "test.db"
        conn = connect(db_path)
        conn.close()
        assert db_path.parent.exists()

    def test_returns_connection_with_row_factory(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = connect(db_path)
        assert conn.row_factory == sqlite3.Row
        conn.close()

    def test_connection_is_usable(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = connect(db_path)
        conn.execute("CREATE TABLE test (id INTEGER)")
        conn.execute("INSERT INTO test VALUES (1)")
        result = conn.execute("SELECT * FROM test").fetchone()
        assert result["id"] == 1
        conn.close()


class TestInitDb:
    """Tests for init_db function."""

    def test_creates_all_tables(self, tmp_path: Path) -> None:
        config = AppConfig()
        config.storage.data_directory = str(tmp_path)
        conn = init_db(config)

        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        table_names = [t["name"] for t in tables]

        assert "sessions" in table_names
        assert "shots" in table_names
        assert "events" in table_names
        assert "fouls" in table_names
        assert "games" in table_names
        assert "key_frames" in table_names
        conn.close()

    def test_creates_all_indexes(self, tmp_path: Path) -> None:
        config = AppConfig()
        config.storage.data_directory = str(tmp_path)
        conn = init_db(config)

        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'"
        ).fetchall()
        index_names = [i["name"] for i in indexes]

        assert "idx_shots_session_id" in index_names
        assert "idx_events_session_id" in index_names
        assert "idx_fouls_session_id" in index_names
        assert "idx_games_session_id" in index_names
        assert "idx_key_frames_session_id" in index_names
        conn.close()

    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        config = AppConfig()
        config.storage.data_directory = str(tmp_path)
        conn = init_db(config)

        result = conn.execute("PRAGMA journal_mode").fetchone()
        assert result[0].lower() == "wal"
        conn.close()

    def test_foreign_keys_enabled(self, tmp_path: Path) -> None:
        config = AppConfig()
        config.storage.data_directory = str(tmp_path)
        conn = init_db(config)

        result = conn.execute("PRAGMA foreign_keys").fetchone()
        assert result[0] == 1
        conn.close()

    def test_idempotent_initialization(self, tmp_path: Path) -> None:
        config = AppConfig()
        config.storage.data_directory = str(tmp_path)

        conn1 = init_db(config)
        conn1.execute(
            "INSERT INTO sessions (id, name, status) VALUES ('test', 'Test', 'active')"
        )
        conn1.commit()
        conn1.close()

        # Re-initialize should not fail or lose data
        conn2 = init_db(config)
        result = conn2.execute("SELECT * FROM sessions WHERE id = 'test'").fetchone()
        assert result["name"] == "Test"
        conn2.close()


class TestInsertEvent:
    """Tests for insert_event function."""

    def test_inserts_event_successfully(self, tmp_path: Path) -> None:
        config = AppConfig()
        config.storage.data_directory = str(tmp_path)
        conn = init_db(config)

        # First create a session
        conn.execute(
            "INSERT INTO sessions (id, name, status) VALUES ('sess1', 'Test', 'active')"
        )
        conn.commit()

        insert_event(
            conn,
            session_id="sess1",
            timestamp_ms=1000,
            event_type="SHOT_START",
            event_data={"shot_number": 1},
        )

        result = conn.execute("SELECT * FROM events WHERE session_id = 'sess1'").fetchone()
        assert result["timestamp_ms"] == 1000
        assert result["event_type"] == "SHOT_START"
        assert json.loads(result["event_data"]) == {"shot_number": 1}
        conn.close()

    def test_event_data_serialized_as_json(self, tmp_path: Path) -> None:
        config = AppConfig()
        config.storage.data_directory = str(tmp_path)
        conn = init_db(config)

        conn.execute(
            "INSERT INTO sessions (id, name, status) VALUES ('sess1', 'Test', 'active')"
        )
        conn.commit()

        complex_data = {
            "balls": [{"id": 1, "position": [100, 200]}],
            "confidence": 0.95,
        }
        insert_event(conn, "sess1", 2000, "TABLE_STATE", complex_data)

        result = conn.execute("SELECT event_data FROM events").fetchone()
        parsed = json.loads(result["event_data"])
        assert parsed["balls"][0]["position"] == [100, 200]
        conn.close()

    def test_received_at_is_set(self, tmp_path: Path) -> None:
        config = AppConfig()
        config.storage.data_directory = str(tmp_path)
        conn = init_db(config)

        conn.execute(
            "INSERT INTO sessions (id, name, status) VALUES ('sess1', 'Test', 'active')"
        )
        conn.commit()

        insert_event(conn, "sess1", 1000, "TEST", {})

        result = conn.execute("SELECT received_at FROM events").fetchone()
        assert result["received_at"] is not None
        # Should be ISO format
        assert "T" in result["received_at"]
        conn.close()


class TestSchemaVersioning:
    """Tests for schema versioning."""

    def test_schema_version_table_created(self, tmp_path: Path) -> None:
        config = AppConfig()
        config.storage.data_directory = str(tmp_path)
        conn = init_db(config)

        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        ).fetchall()
        assert len(tables) == 1
        conn.close()

    def test_schema_version_recorded(self, tmp_path: Path) -> None:
        from pool_telemetry.db import SCHEMA_VERSION

        config = AppConfig()
        config.storage.data_directory = str(tmp_path)
        conn = init_db(config)

        row = conn.execute("SELECT MAX(version) FROM schema_version").fetchone()
        assert row[0] == SCHEMA_VERSION
        conn.close()

    def test_schema_version_has_timestamp(self, tmp_path: Path) -> None:
        config = AppConfig()
        config.storage.data_directory = str(tmp_path)
        conn = init_db(config)

        row = conn.execute("SELECT applied_at FROM schema_version").fetchone()
        assert row["applied_at"] is not None
        assert "T" in row["applied_at"]  # ISO format
        conn.close()


class TestSchemaStatements:
    """Tests for schema definitions."""

    def test_all_statements_are_valid_sql(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(db_path)

        for statement in SCHEMA_STATEMENTS:
            # Should not raise
            conn.execute(statement)

        conn.close()

    def test_sessions_table_has_required_columns(self, tmp_path: Path) -> None:
        config = AppConfig()
        config.storage.data_directory = str(tmp_path)
        conn = init_db(config)

        columns = conn.execute("PRAGMA table_info(sessions)").fetchall()
        column_names = [c["name"] for c in columns]

        required = [
            "id", "name", "created_at", "started_at", "ended_at",
            "source_type", "total_shots", "total_pocketed", "total_fouls",
            "gemini_cost_usd", "status"
        ]
        for col in required:
            assert col in column_names, f"Missing column: {col}"
        conn.close()

    def test_shots_table_has_required_columns(self, tmp_path: Path) -> None:
        config = AppConfig()
        config.storage.data_directory = str(tmp_path)
        conn = init_db(config)

        columns = conn.execute("PRAGMA table_info(shots)").fetchall()
        column_names = [c["name"] for c in columns]

        required = [
            "id", "session_id", "shot_number", "timestamp_start_ms",
            "cue_ball_trajectory", "confidence_overall"
        ]
        for col in required:
            assert col in column_names, f"Missing column: {col}"
        conn.close()
