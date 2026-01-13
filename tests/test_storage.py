"""Tests for storage management."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from pool_telemetry.config import StorageConfig
from pool_telemetry.storage import SessionStorage, StorageManager, StorageStats


@pytest.fixture
def storage_config(tmp_path: Path) -> StorageConfig:
    """Create a storage config with temp directory."""
    return StorageConfig(
        data_directory=str(tmp_path),
        save_key_frames=True,
        save_raw_events=True,
        frame_quality=85,
        max_storage_gb=50,
        auto_cleanup_days=90,
    )


@pytest.fixture
def storage_manager(storage_config: StorageConfig) -> StorageManager:
    """Create a storage manager instance."""
    return StorageManager(storage_config)


@pytest.fixture
def mock_frame() -> np.ndarray:
    """Create a mock video frame."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


class TestStorageManager:
    """Tests for StorageManager class."""

    def test_creates_base_directories(self, storage_manager: StorageManager, tmp_path: Path) -> None:
        assert (tmp_path / "sessions").exists()
        assert (tmp_path / "exports").exists()

    def test_data_directory_property(self, storage_manager: StorageManager, tmp_path: Path) -> None:
        assert storage_manager.data_directory == tmp_path

    def test_sessions_directory_property(self, storage_manager: StorageManager, tmp_path: Path) -> None:
        assert storage_manager.sessions_directory == tmp_path / "sessions"

    def test_exports_directory_property(self, storage_manager: StorageManager, tmp_path: Path) -> None:
        assert storage_manager.exports_directory == tmp_path / "exports"


class TestSessionStorage:
    """Tests for session storage operations."""

    def test_get_session_storage_returns_correct_paths(
        self, storage_manager: StorageManager, tmp_path: Path
    ) -> None:
        session_id = "test_session_123"
        storage = storage_manager.get_session_storage(session_id)

        assert storage.session_id == session_id
        assert storage.root == tmp_path / "sessions" / session_id
        assert storage.frames_dir == tmp_path / "sessions" / session_id / "frames"
        assert storage.thumbnails_dir == tmp_path / "sessions" / session_id / "thumbnails"
        assert storage.metadata_path == tmp_path / "sessions" / session_id / "metadata.json"

    def test_create_session_storage_creates_directories(
        self, storage_manager: StorageManager
    ) -> None:
        session_id = "new_session"
        storage = storage_manager.create_session_storage(session_id)

        assert storage.root.exists()
        assert storage.frames_dir.exists()
        assert storage.thumbnails_dir.exists()
        assert storage.metadata_path.exists()

    def test_create_session_storage_writes_metadata(
        self, storage_manager: StorageManager
    ) -> None:
        session_id = "metadata_test"
        custom_meta = {"source_type": "gopro", "name": "Test Session"}
        storage = storage_manager.create_session_storage(session_id, metadata=custom_meta)

        meta = json.loads(storage.metadata_path.read_text())
        assert meta["session_id"] == session_id
        assert meta["source_type"] == "gopro"
        assert meta["name"] == "Test Session"
        assert "created_at" in meta
        assert meta["frames_saved"] == 0

    def test_delete_session_storage_removes_directory(
        self, storage_manager: StorageManager
    ) -> None:
        session_id = "to_delete"
        storage = storage_manager.create_session_storage(session_id)
        assert storage.root.exists()

        result = storage_manager.delete_session_storage(session_id)
        assert result is True
        assert not storage.root.exists()

    def test_delete_nonexistent_session_returns_false(
        self, storage_manager: StorageManager
    ) -> None:
        result = storage_manager.delete_session_storage("nonexistent")
        assert result is False


class TestFrameSaving:
    """Tests for frame saving operations."""

    def test_save_frame_creates_file(
        self, storage_manager: StorageManager, mock_frame: np.ndarray
    ) -> None:
        session_id = "frame_test"
        storage_manager.create_session_storage(session_id)

        path = storage_manager.save_frame(
            session_id, mock_frame, "pre_shot", shot_number=1, timestamp_ms=1000
        )

        assert path is not None
        assert path.exists()
        assert "shot_0001" in path.name
        assert "pre_shot" in path.name

    def test_save_frame_increments_counter(
        self, storage_manager: StorageManager, mock_frame: np.ndarray
    ) -> None:
        session_id = "counter_test"
        storage_manager.create_session_storage(session_id)

        storage_manager.save_frame(session_id, mock_frame, "key_frame", timestamp_ms=1000)
        storage_manager.save_frame(session_id, mock_frame, "key_frame", timestamp_ms=2000)

        meta = storage_manager.get_session_metadata(session_id)
        assert meta["frames_saved"] == 2

    def test_save_frame_disabled_returns_none(
        self, tmp_path: Path, mock_frame: np.ndarray
    ) -> None:
        config = StorageConfig(data_directory=str(tmp_path), save_key_frames=False)
        manager = StorageManager(config)
        manager.create_session_storage("disabled_test")

        path = manager.save_frame("disabled_test", mock_frame, "test", timestamp_ms=1000)
        assert path is None

    def test_list_session_frames(
        self, storage_manager: StorageManager, mock_frame: np.ndarray
    ) -> None:
        session_id = "list_frames_test"
        storage_manager.create_session_storage(session_id)

        storage_manager.save_frame(session_id, mock_frame, "pre", shot_number=1, timestamp_ms=100)
        storage_manager.save_frame(session_id, mock_frame, "post", shot_number=1, timestamp_ms=200)

        frames = storage_manager.list_session_frames(session_id)
        assert len(frames) == 2


class TestThumbnails:
    """Tests for thumbnail operations."""

    def test_save_thumbnail_creates_file(
        self, storage_manager: StorageManager, mock_frame: np.ndarray
    ) -> None:
        session_id = "thumb_test"
        storage_manager.create_session_storage(session_id)

        path = storage_manager.save_thumbnail(session_id, mock_frame, "session_thumb")

        assert path is not None
        assert path.exists()
        assert path.name == "session_thumb.jpg"

    def test_get_session_thumbnail_returns_path(
        self, storage_manager: StorageManager, mock_frame: np.ndarray
    ) -> None:
        session_id = "get_thumb_test"
        storage_manager.create_session_storage(session_id)
        storage_manager.save_thumbnail(session_id, mock_frame, "session_thumb")

        thumb = storage_manager.get_session_thumbnail(session_id)
        assert thumb is not None
        assert thumb.exists()

    def test_get_session_thumbnail_returns_none_if_missing(
        self, storage_manager: StorageManager
    ) -> None:
        session_id = "no_thumb"
        storage_manager.create_session_storage(session_id)

        thumb = storage_manager.get_session_thumbnail(session_id)
        assert thumb is None


class TestMetadata:
    """Tests for metadata operations."""

    def test_update_session_metadata(self, storage_manager: StorageManager) -> None:
        session_id = "meta_update"
        storage_manager.create_session_storage(session_id)

        storage_manager.update_session_metadata(session_id, {"total_shots": 10})

        meta = storage_manager.get_session_metadata(session_id)
        assert meta["total_shots"] == 10
        assert "updated_at" in meta

    def test_get_session_metadata_returns_none_if_missing(
        self, storage_manager: StorageManager
    ) -> None:
        meta = storage_manager.get_session_metadata("nonexistent")
        assert meta is None

    def test_link_source_video(self, storage_manager: StorageManager) -> None:
        session_id = "video_link"
        storage_manager.create_session_storage(session_id)

        storage_manager.link_source_video(session_id, "/path/to/video.mp4")

        storage = storage_manager.get_session_storage(session_id)
        link_file = storage.root / "source_video.txt"
        assert link_file.exists()
        assert link_file.read_text() == "/path/to/video.mp4"


class TestStorageStats:
    """Tests for storage statistics."""

    def test_get_storage_stats_empty(self, storage_manager: StorageManager) -> None:
        stats = storage_manager.get_storage_stats()
        assert stats.total_bytes == 0
        assert stats.session_count == 0
        assert stats.frame_count == 0

    def test_get_storage_stats_with_sessions(
        self, storage_manager: StorageManager, mock_frame: np.ndarray
    ) -> None:
        storage_manager.create_session_storage("session1")
        storage_manager.create_session_storage("session2")
        storage_manager.save_frame("session1", mock_frame, "test", timestamp_ms=100)

        stats = storage_manager.get_storage_stats()
        assert stats.session_count == 2
        assert stats.frame_count == 1
        assert stats.total_bytes > 0

    def test_storage_stats_total_gb(self) -> None:
        stats = StorageStats(
            total_bytes=1024 * 1024 * 1024,  # 1 GB
            session_count=1,
            frame_count=10,
            oldest_session_date=None,
            newest_session_date=None,
        )
        assert stats.total_gb == 1.0

    def test_check_storage_quota_within(self, storage_manager: StorageManager) -> None:
        within, usage = storage_manager.check_storage_quota()
        assert within is True
        assert usage == 0.0


class TestCleanup:
    """Tests for cleanup operations."""

    def test_cleanup_orphaned_storage(self, storage_manager: StorageManager) -> None:
        # Create storage that's not in the "database"
        storage_manager.create_session_storage("orphan1")
        storage_manager.create_session_storage("orphan2")
        storage_manager.create_session_storage("valid")

        # Only "valid" is in the database
        db_session_ids = {"valid"}
        cleaned = storage_manager.cleanup_orphaned_storage(db_session_ids)

        assert cleaned == 2
        assert not storage_manager.get_session_storage("orphan1").root.exists()
        assert not storage_manager.get_session_storage("orphan2").root.exists()
        assert storage_manager.get_session_storage("valid").root.exists()

    def test_cleanup_old_sessions_respects_cutoff(
        self, tmp_path: Path
    ) -> None:
        config = StorageConfig(
            data_directory=str(tmp_path),
            auto_cleanup_days=0,  # Disabled
        )
        manager = StorageManager(config)
        manager.create_session_storage("test")

        cleaned = manager.cleanup_old_sessions()
        assert cleaned == 0


class TestSessionStorageDataclass:
    """Tests for SessionStorage dataclass."""

    def test_ensure_dirs_creates_all_directories(self, tmp_path: Path) -> None:
        storage = SessionStorage(
            session_id="test",
            root=tmp_path / "sessions" / "test",
            frames_dir=tmp_path / "sessions" / "test" / "frames",
            thumbnails_dir=tmp_path / "sessions" / "test" / "thumbnails",
            metadata_path=tmp_path / "sessions" / "test" / "metadata.json",
        )
        storage.ensure_dirs()

        assert storage.root.exists()
        assert storage.frames_dir.exists()
        assert storage.thumbnails_dir.exists()
