"""Tests for data export functionality."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from pool_telemetry.config import AppConfig
from pool_telemetry.db import connect, get_db_path, init_db, insert_event
from pool_telemetry.exporter import ExportManager
from pool_telemetry.sessions import SessionManager


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Create an initialized database and return its path."""
    config = AppConfig()
    config.storage.data_directory = str(tmp_path)
    conn = init_db(config)
    conn.close()
    return get_db_path(config)


@pytest.fixture
def session_with_data(db_path: Path) -> str:
    """Create a session with test data and return its ID."""
    session_manager = SessionManager(db_path)
    session = session_manager.start_session("Test Session", "video_file", "/test.mp4")

    with connect(db_path) as conn:
        # Add events
        insert_event(conn, session.session_id, 1000, "SHOT_START", {"shot_number": 1})
        insert_event(conn, session.session_id, 2000, "POCKET", {"ball": 3})
        insert_event(conn, session.session_id, 3000, "SHOT_END", {"shot_number": 1})

        # Add shots
        conn.execute(
            """
            INSERT INTO shots (session_id, shot_number, timestamp_start_ms,
                               timestamp_end_ms, confidence_overall)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session.session_id, 1, 1000, 3000, 0.95),
        )
        conn.execute(
            """
            INSERT INTO shots (session_id, shot_number, timestamp_start_ms,
                               timestamp_end_ms, confidence_overall)
            VALUES (?, ?, ?, ?, ?)
            """,
            (session.session_id, 2, 4000, 6000, 0.88),
        )

        # Add a foul
        conn.execute(
            """
            INSERT INTO fouls (session_id, shot_number, timestamp_ms, foul_type)
            VALUES (?, ?, ?, ?)
            """,
            (session.session_id, 2, 5000, "scratch"),
        )

        conn.commit()

    session_manager.end_session(session.session_id, 2, 1, 1, 0.05)
    return session.session_id


@pytest.fixture
def export_manager(db_path: Path) -> ExportManager:
    """Create an ExportManager instance."""
    return ExportManager(db_path)


class TestExportFullJson:
    """Tests for export_full_json method."""

    def test_exports_all_tables(
        self, export_manager: ExportManager, session_with_data: str, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "export.json"
        export_manager.export_full_json(session_with_data, output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert "session" in data
        assert "shots" in data
        assert "events" in data
        assert "fouls" in data
        assert "games" in data
        assert "key_frames" in data

    def test_session_data_correct(
        self, export_manager: ExportManager, session_with_data: str, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "export.json"
        export_manager.export_full_json(session_with_data, output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert data["session"]["id"] == session_with_data
        assert data["session"]["name"] == "Test Session"
        assert data["session"]["status"] == "completed"

    def test_shots_data_correct(
        self, export_manager: ExportManager, session_with_data: str, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "export.json"
        export_manager.export_full_json(session_with_data, output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert len(data["shots"]) == 2
        assert data["shots"][0]["shot_number"] == 1
        assert data["shots"][1]["shot_number"] == 2

    def test_events_ordered_by_timestamp(
        self, export_manager: ExportManager, session_with_data: str, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "export.json"
        export_manager.export_full_json(session_with_data, output_path)

        with open(output_path) as f:
            data = json.load(f)

        timestamps = [e["timestamp_ms"] for e in data["events"]]
        assert timestamps == sorted(timestamps)

    def test_fouls_included(
        self, export_manager: ExportManager, session_with_data: str, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "export.json"
        export_manager.export_full_json(session_with_data, output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert len(data["fouls"]) == 1
        assert data["fouls"][0]["foul_type"] == "scratch"


class TestExportClaudeJson:
    """Tests for export_claude_json method."""

    def test_exports_subset_of_tables(
        self, export_manager: ExportManager, session_with_data: str, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "export.json"
        export_manager.export_claude_json(session_with_data, output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert "session" in data
        assert "shots" in data
        assert "events" in data
        # These should NOT be in claude export
        assert "fouls" not in data
        assert "games" not in data
        assert "key_frames" not in data

    def test_includes_session_and_shots(
        self, export_manager: ExportManager, session_with_data: str, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "export.json"
        export_manager.export_claude_json(session_with_data, output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert data["session"]["id"] == session_with_data
        assert len(data["shots"]) == 2


class TestExportShotsCsv:
    """Tests for export_shots_csv method."""

    def test_creates_csv_file(
        self, export_manager: ExportManager, session_with_data: str, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "shots.csv"
        export_manager.export_shots_csv(session_with_data, output_path)

        assert output_path.exists()
        assert output_path.suffix == ".csv"

    def test_csv_has_correct_rows(
        self, export_manager: ExportManager, session_with_data: str, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "shots.csv"
        export_manager.export_shots_csv(session_with_data, output_path)

        with open(output_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 2
        assert rows[0]["shot_number"] == "1"
        assert rows[1]["shot_number"] == "2"

    def test_csv_has_header(
        self, export_manager: ExportManager, session_with_data: str, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "shots.csv"
        export_manager.export_shots_csv(session_with_data, output_path)

        with open(output_path) as f:
            header = f.readline()

        assert "shot_number" in header
        assert "session_id" in header
        assert "confidence_overall" in header

    def test_empty_shots_creates_empty_file(
        self, export_manager: ExportManager, db_path: Path, tmp_path: Path
    ) -> None:
        # Create session with no shots
        session_manager = SessionManager(db_path)
        session = session_manager.start_session("Empty", "video_file", "/test.mp4")

        output_path = tmp_path / "empty.csv"
        export_manager.export_shots_csv(session.session_id, output_path)

        assert output_path.exists()
        assert output_path.read_text() == ""


class TestExportEventsJsonl:
    """Tests for export_events_jsonl method."""

    def test_creates_jsonl_file(
        self, export_manager: ExportManager, session_with_data: str, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "events.jsonl"
        export_manager.export_events_jsonl(session_with_data, output_path)

        assert output_path.exists()

    def test_one_json_object_per_line(
        self, export_manager: ExportManager, session_with_data: str, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "events.jsonl"
        export_manager.export_events_jsonl(session_with_data, output_path)

        with open(output_path) as f:
            lines = f.readlines()

        assert len(lines) == 3  # We inserted 3 events

        for line in lines:
            # Each line should be valid JSON
            data = json.loads(line)
            assert "event_type" in data
            assert "timestamp_ms" in data

    def test_events_ordered_by_timestamp(
        self, export_manager: ExportManager, session_with_data: str, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "events.jsonl"
        export_manager.export_events_jsonl(session_with_data, output_path)

        timestamps = []
        with open(output_path) as f:
            for line in f:
                data = json.loads(line)
                timestamps.append(data["timestamp_ms"])

        assert timestamps == sorted(timestamps)

    def test_empty_events_creates_empty_file(
        self, export_manager: ExportManager, db_path: Path, tmp_path: Path
    ) -> None:
        session_manager = SessionManager(db_path)
        session = session_manager.start_session("Empty", "video_file", "/test.mp4")

        output_path = tmp_path / "empty.jsonl"
        export_manager.export_events_jsonl(session.session_id, output_path)

        assert output_path.exists()
        assert output_path.read_text() == ""


class TestExportNonexistentSession:
    """Tests for exporting sessions that don't exist."""

    def test_full_json_empty_for_nonexistent(
        self, export_manager: ExportManager, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "export.json"
        export_manager.export_full_json("nonexistent-id", output_path)

        with open(output_path) as f:
            data = json.load(f)

        assert data["session"] == {}
        assert data["shots"] == []
        assert data["events"] == []

    def test_csv_empty_for_nonexistent(
        self, export_manager: ExportManager, tmp_path: Path
    ) -> None:
        output_path = tmp_path / "shots.csv"
        export_manager.export_shots_csv("nonexistent-id", output_path)

        assert output_path.read_text() == ""
