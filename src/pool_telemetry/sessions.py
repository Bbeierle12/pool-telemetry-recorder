from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .db import connect


@dataclass
class SessionInfo:
    session_id: str
    name: str
    created_at: str
    status: str
    source_type: str
    source_path: str


class SessionManager:
    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path

    def start_session(
        self,
        name: Optional[str],
        source_type: str,
        source_path: str,
    ) -> SessionInfo:
        session_id = uuid.uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        session_name = name or f"Session {created_at}"
        with connect(self._db_path) as conn:
            conn.execute(
                """
                INSERT INTO sessions (id, name, created_at, started_at, source_type, source_path, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    session_name,
                    created_at,
                    created_at,
                    source_type,
                    source_path,
                    "recording",
                ),
            )
            conn.commit()
        return SessionInfo(
            session_id=session_id,
            name=session_name,
            created_at=created_at,
            status="recording",
            source_type=source_type,
            source_path=source_path,
        )

    def end_session(
        self,
        session_id: str,
        total_shots: int,
        total_pocketed: int,
        total_fouls: int,
        cost_usd: float,
        notes: Optional[str] = None,
    ) -> None:
        ended_at = datetime.now(timezone.utc).isoformat()
        with connect(self._db_path) as conn:
            conn.execute(
                """
                UPDATE sessions
                SET ended_at = ?, total_shots = ?, total_pocketed = ?, total_fouls = ?,
                    gemini_cost_usd = ?, status = ?, notes = ?
                WHERE id = ?
                """,
                (
                    ended_at,
                    total_shots,
                    total_pocketed,
                    total_fouls,
                    cost_usd,
                    "completed",
                    notes,
                    session_id,
                ),
            )
            conn.commit()

    def list_sessions(self) -> List[Dict[str, Any]]:
        with connect(self._db_path) as conn:
            cursor = conn.execute(
                """
                SELECT id, name, created_at, started_at, ended_at, total_shots, total_pocketed,
                       total_fouls, status
                FROM sessions
                ORDER BY created_at DESC
                """
            )
            return [dict(row) for row in cursor.fetchall()]

    def delete_session(self, session_id: str) -> None:
        with connect(self._db_path) as conn:
            conn.execute("DELETE FROM key_frames WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM fouls WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM shots WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM games WHERE session_id = ?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            conn.commit()
