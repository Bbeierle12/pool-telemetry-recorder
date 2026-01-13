from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .config import AppConfig

logger = logging.getLogger(__name__)

# Current schema version - increment when schema changes
SCHEMA_VERSION = 1

SCHEMA_STATEMENTS = [
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY,
        applied_at DATETIME
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        name TEXT,
        created_at DATETIME,
        started_at DATETIME,
        ended_at DATETIME,
        source_type TEXT,
        source_path TEXT,
        video_duration_ms INTEGER,
        video_resolution TEXT,
        video_framerate REAL,
        calibration_data TEXT,
        total_shots INTEGER,
        total_pocketed INTEGER,
        total_fouls INTEGER,
        total_games INTEGER,
        gemini_cost_usd REAL,
        status TEXT,
        notes TEXT,
        metadata TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS shots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT REFERENCES sessions(id),
        shot_number INTEGER,
        game_number INTEGER,
        player INTEGER,
        timestamp_start_ms INTEGER,
        timestamp_end_ms INTEGER,
        duration_ms INTEGER,
        table_state_before TEXT,
        table_state_after TEXT,
        cue_stick_data TEXT,
        cue_ball_trajectory TEXT,
        object_ball_trajectories TEXT,
        collisions TEXT,
        pocketing_events TEXT,
        cushion_contacts INTEGER,
        balls_contacted TEXT,
        balls_pocketed TEXT,
        derived_metrics TEXT,
        confidence_overall REAL,
        frames_analyzed INTEGER,
        anomalies TEXT,
        pre_frame_path TEXT,
        post_frame_path TEXT,
        analyzed BOOLEAN DEFAULT 0,
        analysis_data TEXT,
        created_at DATETIME
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT REFERENCES sessions(id),
        timestamp_ms INTEGER,
        event_type TEXT,
        event_data TEXT,
        processed BOOLEAN DEFAULT 0,
        error_message TEXT,
        received_at DATETIME
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS fouls (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT REFERENCES sessions(id),
        shot_id INTEGER REFERENCES shots(id),
        shot_number INTEGER,
        timestamp_ms INTEGER,
        foul_type TEXT,
        details TEXT,
        created_at DATETIME
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS games (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT REFERENCES sessions(id),
        game_number INTEGER,
        game_type TEXT,
        started_at_ms INTEGER,
        ended_at_ms INTEGER,
        winner INTEGER,
        win_condition TEXT,
        player_1_type TEXT,
        player_2_type TEXT,
        final_score TEXT,
        created_at DATETIME
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS key_frames (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id TEXT REFERENCES sessions(id),
        shot_id INTEGER REFERENCES shots(id),
        timestamp_ms INTEGER,
        frame_type TEXT,
        file_path TEXT,
        file_size_bytes INTEGER,
        resolution TEXT,
        created_at DATETIME
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_shots_session_id ON shots(session_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_events_session_id ON events(session_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_fouls_session_id ON fouls(session_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_games_session_id ON games(session_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS idx_key_frames_session_id ON key_frames(session_id)
    """,
]


def get_db_path(config: AppConfig) -> Path:
    """Get the database file path from configuration.

    Args:
        config: Application configuration.

    Returns:
        Path to the SQLite database file.
    """
    return Path(config.storage.data_directory) / "database.sqlite"


def connect(db_path: Path) -> sqlite3.Connection:
    """Open a connection to the SQLite database.

    Creates parent directories if needed. Sets row_factory to sqlite3.Row
    for dict-like row access.

    Args:
        db_path: Path to the database file.

    Returns:
        SQLite connection with Row factory configured.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(config: AppConfig) -> sqlite3.Connection:
    """Initialize the database with schema and pragmas.

    Creates all tables and indexes if they don't exist. Safe to call
    multiple times (idempotent).

    Args:
        config: Application configuration.

    Returns:
        Open database connection (caller should close when done).
    """
    db_path = get_db_path(config)
    conn = connect(db_path)
    _apply_pragmas(conn)
    _initialize_schema(conn, SCHEMA_STATEMENTS)
    _check_schema_version(conn)
    return conn


def insert_event(
    conn: sqlite3.Connection,
    session_id: str,
    timestamp_ms: int,
    event_type: str,
    event_data: dict,
) -> None:
    """Insert an event record into the events table.

    Args:
        conn: Database connection.
        session_id: ID of the session this event belongs to.
        timestamp_ms: Event timestamp in milliseconds.
        event_type: Type of event (e.g., "SHOT_START", "POCKET").
        event_data: Event payload as a dictionary (will be JSON-serialized).
    """
    conn.execute(
        """
        INSERT INTO events (session_id, timestamp_ms, event_type, event_data, received_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            session_id,
            timestamp_ms,
            event_type,
            json.dumps(event_data),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    conn.commit()


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")


def _initialize_schema(conn: sqlite3.Connection, statements: Iterable[str]) -> None:
    for statement in statements:
        conn.execute(statement)
    conn.commit()


def _get_schema_version(conn: sqlite3.Connection) -> int:
    """Get the current schema version from the database.

    Returns:
        Schema version number, or 0 if not set.
    """
    try:
        row = conn.execute(
            "SELECT MAX(version) FROM schema_version"
        ).fetchone()
        return row[0] or 0
    except sqlite3.OperationalError:
        # Table doesn't exist yet
        return 0


def _set_schema_version(conn: sqlite3.Connection, version: int) -> None:
    """Record the schema version in the database."""
    conn.execute(
        "INSERT OR REPLACE INTO schema_version (version, applied_at) VALUES (?, ?)",
        (version, datetime.now(timezone.utc).isoformat()),
    )
    conn.commit()


def _check_schema_version(conn: sqlite3.Connection) -> None:
    """Check and log schema version status."""
    db_version = _get_schema_version(conn)

    if db_version == 0:
        # Fresh database, set initial version
        _set_schema_version(conn, SCHEMA_VERSION)
        logger.info("Initialized database schema version %d", SCHEMA_VERSION)
    elif db_version < SCHEMA_VERSION:
        # Database needs migration (future feature)
        logger.warning(
            "Database schema version %d is older than application version %d. "
            "Migration may be needed.",
            db_version,
            SCHEMA_VERSION,
        )
        # For now, just update the version (no actual migration logic yet)
        _set_schema_version(conn, SCHEMA_VERSION)
    elif db_version > SCHEMA_VERSION:
        logger.warning(
            "Database schema version %d is newer than application version %d. "
            "You may be using an older application version.",
            db_version,
            SCHEMA_VERSION,
        )
