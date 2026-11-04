"""
SQLite database for persisting all tracked detections.

Schema: videos, persons, vehicles, objects tables.
Uses WAL mode for concurrent read/write safety.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Schema revision history:
#   0 -> original relative-second schema
#   1 -> bbox_time_s / bbox_frame columns (ad-hoc migration, kept for compat)
#   2 -> absolute-time columns (start_epoch, *_ts), embeddings.model, new indexes
_LATEST_USER_VERSION = 2

_SCHEMA_SQL = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS videos (
    video_id    TEXT PRIMARY KEY,
    file_path   TEXT NOT NULL,
    sha256      TEXT,
    camera_id   TEXT DEFAULT 'cam_default',
    duration_s  REAL,
    fps         REAL,
    width       INTEGER,
    height      INTEGER,
    status      TEXT DEFAULT 'processing',
    start_epoch REAL,
    start_source TEXT DEFAULT 'unknown',
    tz          TEXT,
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS persons (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id        TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    track_id        INTEGER NOT NULL,
    gender          TEXT DEFAULT 'unknown',
    age_est         REAL,
    activity        TEXT DEFAULT 'unknown',
    pose            TEXT DEFAULT 'unknown',
    orientation     TEXT DEFAULT 'unknown',
    upper_color     TEXT DEFAULT 'unknown',
    lower_color     TEXT DEFAULT 'unknown',
    first_seen_s    REAL,
    last_seen_s     REAL,
    first_seen_ts   REAL,
    last_seen_ts    REAL,
    frame_count     INTEGER DEFAULT 0,
    peak_conf       REAL DEFAULT 0.0,
    bbox_json       TEXT,
    bbox_time_s     REAL,
    bbox_frame      INTEGER,
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(video_id, track_id)
);

CREATE TABLE IF NOT EXISTS vehicles (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id        TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    track_id        INTEGER NOT NULL,
    vehicle_type    TEXT NOT NULL,
    color           TEXT DEFAULT 'unknown',
    color_conf      REAL DEFAULT 0.0,
    first_seen_s    REAL,
    last_seen_s     REAL,
    first_seen_ts   REAL,
    last_seen_ts    REAL,
    frame_count     INTEGER DEFAULT 0,
    peak_conf       REAL DEFAULT 0.0,
    bbox_json       TEXT,
    bbox_time_s     REAL,
    bbox_frame      INTEGER,
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(video_id, track_id)
);

CREATE TABLE IF NOT EXISTS objects (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id                TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    track_id                INTEGER,
    object_type             TEXT NOT NULL,
    color                   TEXT DEFAULT 'unknown',
    associated_person_track INTEGER,
    first_seen_s            REAL,
    last_seen_s             REAL,
    first_seen_ts           REAL,
    last_seen_ts            REAL,
    frame_count             INTEGER DEFAULT 0,
    peak_conf               REAL DEFAULT 0.0,
    bbox_json               TEXT,
    bbox_time_s             REAL,
    bbox_frame              INTEGER,
    created_at              TEXT DEFAULT (datetime('now'))
);
"""

_SCHEMA_TABLES_SQL = _SCHEMA_SQL + """
-- Embeddings table for vector search (SigLIP 2 768-d vectors stored as BLOBs)
CREATE TABLE IF NOT EXISTS embeddings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id        TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    entity_type     TEXT NOT NULL CHECK(entity_type IN ('person','vehicle','object')),
    entity_id       INTEGER NOT NULL,
    track_id        INTEGER,
    timestamp_s     REAL,
    ts              REAL,
    model           TEXT,
    embedding       BLOB NOT NULL,
    crop_path       TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- Crops table for storing representative crop images per entity
CREATE TABLE IF NOT EXISTS crops (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id        TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    entity_type     TEXT NOT NULL CHECK(entity_type IN ('person','vehicle','object')),
    entity_id       INTEGER NOT NULL,
    track_id        INTEGER,
    timestamp_s     REAL,
    ts              REAL,
    crop_path       TEXT NOT NULL,
    created_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(video_id, entity_type, track_id)
);

-- Virtual gate line crossing events table
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id        TEXT NOT NULL REFERENCES videos(video_id) ON DELETE CASCADE,
    event_type      TEXT NOT NULL CHECK(event_type IN ('entry', 'exit', 'crossing')),
    entity_type     TEXT NOT NULL CHECK(entity_type IN ('person', 'vehicle', 'object')),
    track_id        INTEGER NOT NULL,
    line_name       TEXT DEFAULT 'main_gate',
    timestamp_s     REAL NOT NULL,
    ts              REAL,
    confidence      REAL DEFAULT 0.0,
    bbox_json       TEXT,
    metadata_json   TEXT,
    created_at      TEXT DEFAULT (datetime('now'))
);

-- Indexes for fast search queries
CREATE INDEX IF NOT EXISTS idx_persons_video ON persons(video_id);
CREATE INDEX IF NOT EXISTS idx_persons_gender ON persons(gender);
CREATE INDEX IF NOT EXISTS idx_persons_upper_color ON persons(upper_color);
CREATE INDEX IF NOT EXISTS idx_persons_lower_color ON persons(lower_color);
CREATE INDEX IF NOT EXISTS idx_persons_video_time ON persons(video_id, first_seen_s, last_seen_s);
CREATE INDEX IF NOT EXISTS idx_vehicles_video ON vehicles(video_id);
CREATE INDEX IF NOT EXISTS idx_vehicles_type ON vehicles(vehicle_type);
CREATE INDEX IF NOT EXISTS idx_vehicles_color ON vehicles(color);
CREATE INDEX IF NOT EXISTS idx_vehicles_video_time ON vehicles(video_id, first_seen_s, last_seen_s);
CREATE INDEX IF NOT EXISTS idx_objects_video ON objects(video_id);
CREATE INDEX IF NOT EXISTS idx_objects_type ON objects(object_type);
CREATE INDEX IF NOT EXISTS idx_objects_color ON objects(color);
CREATE INDEX IF NOT EXISTS idx_objects_video_time ON objects(video_id, first_seen_s, last_seen_s);
CREATE INDEX IF NOT EXISTS idx_crops_entity ON crops(entity_type, entity_id);
CREATE INDEX IF NOT EXISTS idx_events_video ON events(video_id);
CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_time ON events(timestamp_s);

-- Indexes on legacy-safe columns for embeddings/events
CREATE INDEX IF NOT EXISTS idx_emb_video ON embeddings(video_id);
CREATE INDEX IF NOT EXISTS idx_emb_entity ON embeddings(entity_type, entity_id);
"""

# Indexes that reference migrated columns (ts / model); created in _migrate() after
# those columns exist on legacy databases.
_SCHEMA_TS_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_emb_ts ON embeddings(entity_type, ts);
CREATE INDEX IF NOT EXISTS idx_emb_model ON embeddings(model);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
"""

_ABS_TIME_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_persons_first_seen_ts ON persons(first_seen_ts);
CREATE INDEX IF NOT EXISTS idx_persons_last_seen_ts ON persons(last_seen_ts);
CREATE INDEX IF NOT EXISTS idx_vehicles_first_seen_ts ON vehicles(first_seen_ts);
CREATE INDEX IF NOT EXISTS idx_vehicles_last_seen_ts ON vehicles(last_seen_ts);
CREATE INDEX IF NOT EXISTS idx_objects_first_seen_ts ON objects(first_seen_ts);
CREATE INDEX IF NOT EXISTS idx_objects_last_seen_ts ON objects(last_seen_ts);
""" + _SCHEMA_TS_INDEX_SQL


class CCTVDatabase:
    """SQLite database manager for CCTV detection persistence."""

    def __init__(self, db_path: str | Path = "outputs/cctv.db", read_only: bool = False) -> None:
        self.db_path = Path(db_path)
        self.read_only = read_only
        is_memory = str(self.db_path) == ":memory:"
        if not is_memory:
            self.db_path = self.db_path.resolve()
            if not read_only:
                self.db_path.parent.mkdir(parents=True, exist_ok=True)

        connect_error: sqlite3.OperationalError | None = None
        for attempt in range(5):
            try:
                if read_only and not is_memory:
                    uri = f"file:{self.db_path.as_posix()}?mode=ro"
                    self.conn = sqlite3.connect(uri, uri=True, timeout=30.0, check_same_thread=False)
                else:
                    self.conn = sqlite3.connect(
                        str(self.db_path), timeout=30.0, check_same_thread=False
                    )
                break
            except sqlite3.OperationalError as exc:
                connect_error = exc
                if attempt == 4:
                    raise sqlite3.OperationalError(
                        f"Unable to open SQLite database '{self.db_path}'. "
                        "Ensure the directory exists and the database is not being moved or deleted."
                    ) from exc
                time.sleep(0.25 * (2**attempt))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA busy_timeout = 30000")
        # temp_store = MEMORY must be set before schema init/migration: index
        # rebuilds (e.g. objects) can fail with SQLITE_CANTOPEN when the sorter
        # spills to a temp file.
        self.conn.execute("PRAGMA synchronous = NORMAL")
        self.conn.execute("PRAGMA temp_store = MEMORY")
        self.conn.execute("PRAGMA cache_size = -65536")
        if not read_only:
            self._backup_current_user_version()
            self._init_schema()
        else:
            self.conn.execute("PRAGMA query_only = ON")

    def _backup_current_user_version(self) -> None:
        """Snapshot the database file before any schema migration (v0/v1 only)."""
        version = int(self.conn.execute("PRAGMA user_version").fetchone()[0] or 0)
        if version >= _LATEST_USER_VERSION or str(self.db_path) == ":memory:":
            return
        backup_path = self.db_path.with_name(f"{self.db_path.name}.bak-v{version}")
        try:
            if backup_path.exists():
                return
            with sqlite3.connect(str(backup_path)) as bconn:
                self.conn.backup(bconn)
        except Exception as exc:  # backup must never block ingest
            print(f"[db] WARNING: pre-migration backup failed: {exc}", file=sys.stderr)

    def _migrate(self) -> None:
        """Versioned, idempotent schema migration up to _LATEST_USER_VERSION."""
        if int(self.conn.execute("PRAGMA user_version").fetchone()[0] or 0) >= _LATEST_USER_VERSION:
            return
        # Missing tables mean a fresh database: _SCHEMA_SQL already created the
        # full shape, so only column backfill matters for legacy databases.
        self._ensure_column("videos", "start_epoch", "REAL")
        self._ensure_column("videos", "start_source", "TEXT DEFAULT 'unknown'")
        self._ensure_column("videos", "tz", "TEXT")
        for table in ("persons", "vehicles", "objects"):
            self._ensure_column(table, "first_seen_ts", "REAL")
            self._ensure_column(table, "last_seen_ts", "REAL")
        self._ensure_column("events", "ts", "REAL")
        self._ensure_column("embeddings", "ts", "REAL")
        self._ensure_column("embeddings", "model", "TEXT")
        self._ensure_column("crops", "ts", "REAL")

        # Legacy rows were all produced by the SigLIP 2 encoder; tag them so
        # mixed-model comparisons are impossible in the vector search paths.
        from common.embed import get_model_tag
        self.conn.execute(
            "UPDATE embeddings SET model = ? WHERE model IS NULL", (get_model_tag(),)
        )

        # Backfill absolute timestamps from videos.start_epoch + relative seconds.
        self._backfill_abs_ts()

        self.conn.executescript(_ABS_TIME_INDEX_SQL)
        self.conn.commit()
        self.conn.execute(f"PRAGMA user_version = {_LATEST_USER_VERSION}")

    def _backfill_abs_ts(self, video_id: str | None = None) -> None:
        """Fill *_ts columns from videos.start_epoch + relative seconds.

        Rows whose video has no start_epoch stay NULL. Idempotent: only NULL
        absolute values are written. Called for the whole DB during migration
        and per-video when a start_epoch is learned later (re-ingest with
        --start-time).
        """
        extra, eparams = ("", [])
        if video_id is not None:
            extra, eparams = " AND s.video_id = ?", [video_id]

        def stmt(table: str, null_col: str, sets: str) -> str:
            return f"""UPDATE {table}
                   SET {sets}
                   FROM (SELECT video_id, start_epoch FROM videos) AS s
                   WHERE {table}.video_id = s.video_id
                     AND s.start_epoch IS NOT NULL{extra}
                     AND {table}.{null_col} IS NULL"""

        for sql in (
            stmt("persons", "first_seen_ts",
                 "first_seen_ts = s.start_epoch + persons.first_seen_s, "
                 "last_seen_ts = s.start_epoch + persons.last_seen_s"),
            stmt("vehicles", "first_seen_ts",
                 "first_seen_ts = s.start_epoch + vehicles.first_seen_s, "
                 "last_seen_ts = s.start_epoch + vehicles.last_seen_s"),
            stmt("objects", "first_seen_ts",
                 "first_seen_ts = s.start_epoch + objects.first_seen_s, "
                 "last_seen_ts = s.start_epoch + objects.last_seen_s"),
            stmt("events", "ts", "ts = s.start_epoch + events.timestamp_s"),
            stmt("embeddings", "ts", "ts = s.start_epoch + embeddings.timestamp_s"),
            stmt("crops", "ts", "ts = s.start_epoch + crops.timestamp_s"),
        ):
            self.conn.execute(sql, eparams)

    def _init_schema(self) -> None:
        """Create tables and indexes if they exist nowhere yet, then migrate."""
        self.conn.executescript(_SCHEMA_TABLES_SQL)
        self._migrate()
        self._migrate()
        # Lightweight migrations for databases created before bbox timestamps existed.
        for table in ("persons", "vehicles", "objects"):
            self._ensure_column(table, "bbox_time_s", "REAL")
            self._ensure_column(table, "bbox_frame", "INTEGER")
            # Legacy rows stored the peak-confidence box without knowing which frame
            # it came from. The first-seen timestamp is the safest available anchor.
            self.conn.execute(
                f"UPDATE {table} SET bbox_time_s = first_seen_s "
                f"WHERE bbox_time_s IS NULL AND first_seen_s IS NOT NULL"
            )
            # Recover the frame index when the source FPS is known. Object-only
            # ingestion has always stored the detection timestamp, so this makes
            # those existing boxes render on the corresponding source frame.
            self.conn.execute(
                f"""UPDATE {table}
                    SET bbox_frame = CAST(ROUND(bbox_time_s * v.fps) AS INTEGER)
                    FROM videos v
                    WHERE {table}.video_id = v.video_id
                      AND {table}.bbox_frame IS NULL
                      AND {table}.bbox_time_s IS NOT NULL
                      AND v.fps IS NOT NULL
                      AND v.fps > 0"""
            )
        self.conn.commit()

    def _ensure_column(self, table: str, column: str, declaration: str) -> None:
        """Add a column to an existing table when migrating an older database."""
        columns = {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")}
        if column not in columns:
            self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {declaration}")

    def close(self) -> None:
        """Close the database connection."""
        if self.conn:
            self.conn.close()

    # ── Video operations ──────────────────────────────────────────────

    @staticmethod
    def compute_sha256(file_path: str | Path, chunk_size: int = 8192) -> str:
        """Compute SHA-256 hash of a file."""
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest()

    def register_video(
        self,
        video_id: str,
        file_path: str,
        camera_id: str = "cam_default",
        duration_s: float | None = None,
        fps: float | None = None,
        width: int | None = None,
        height: int | None = None,
        sha256: str | None = None,
        start_epoch: float | None = None,
        start_source: str = "unknown",
        tz: str | None = None,
    ) -> str:
        """Register a video in the database. Returns video_id.

        start_epoch is the UTC epoch second of frame 0 of the video; all
        absolute timestamps are derived from it. start_source records how the
        value was obtained: 'arg' | 'metadata' | 'mtime' | 'live' | 'unknown'.
        """
        self.conn.execute(
            """INSERT INTO videos
               (video_id, file_path, sha256, camera_id, duration_s, fps, width, height,
                start_epoch, start_source, tz, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'processing')
               ON CONFLICT(video_id) DO UPDATE SET
                   file_path = excluded.file_path,
                   sha256 = excluded.sha256,
                   camera_id = excluded.camera_id,
                   duration_s = excluded.duration_s,
                   fps = excluded.fps,
                   width = excluded.width,
                   height = excluded.height,
                   start_epoch = COALESCE(excluded.start_epoch, videos.start_epoch),
                   start_source = CASE
                       WHEN excluded.start_epoch IS NOT NULL THEN excluded.start_source
                       ELSE videos.start_source END,
                   tz = COALESCE(excluded.tz, videos.tz),
                   status = 'processing'""",
            (video_id, file_path, sha256, camera_id, duration_s, fps, width, height,
             start_epoch, start_source, tz),
        )
        self.conn.commit()
        if start_epoch is not None:
            # Videos registered without a start first (legacy rows) get their
            # absolute timestamps filled relative to the newly-known epoch.
            self._backfill_abs_ts(video_id)
            self.conn.commit()
        return video_id

    def update_video_status(self, video_id: str, status: str) -> None:
        """Update video processing status."""
        self.conn.execute(
            "UPDATE videos SET status = ? WHERE video_id = ?",
            (status, video_id),
        )
        self.conn.commit()

    # ── Person operations ─────────────────────────────────────────────

    def upsert_person(
        self,
        video_id: str,
        track_id: int,
        gender: str = "unknown",
        age_est: float | None = None,
        activity: str = "unknown",
        pose: str = "unknown",
        orientation: str = "unknown",
        upper_color: str = "unknown",
        lower_color: str = "unknown",
        first_seen_s: float | None = None,
        last_seen_s: float | None = None,
        frame_count: int = 0,
        peak_conf: float = 0.0,
        bbox_json: str | None = None,
        bbox_time_s: float | None = None,
        bbox_frame: int | None = None,
    ) -> None:
        """Insert or update a tracked person."""
        self.conn.execute(
            """INSERT INTO persons
               (video_id, track_id, gender, age_est, activity, pose, orientation,
                upper_color, lower_color, first_seen_s, last_seen_s, frame_count,
                peak_conf, bbox_json, bbox_time_s, bbox_frame,
                first_seen_ts, last_seen_ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       (SELECT start_epoch FROM videos WHERE video_id = ?) + ?,
                       (SELECT start_epoch FROM videos WHERE video_id = ?) + ?)
               ON CONFLICT(video_id, track_id) DO UPDATE SET
                   gender = CASE WHEN excluded.gender != 'unknown' THEN excluded.gender ELSE persons.gender END,
                   age_est = COALESCE(excluded.age_est, persons.age_est),
                   activity = CASE WHEN excluded.activity != 'unknown' THEN excluded.activity ELSE persons.activity END,
                   pose = CASE WHEN excluded.pose != 'unknown' THEN excluded.pose ELSE persons.pose END,
                   orientation = CASE WHEN excluded.orientation != 'unknown' THEN excluded.orientation ELSE persons.orientation END,
                   upper_color = CASE WHEN excluded.upper_color != 'unknown' THEN excluded.upper_color ELSE persons.upper_color END,
                   lower_color = CASE WHEN excluded.lower_color != 'unknown' THEN excluded.lower_color ELSE persons.lower_color END,
                   first_seen_s = MIN(COALESCE(persons.first_seen_s, excluded.first_seen_s), COALESCE(excluded.first_seen_s, persons.first_seen_s)),
                   last_seen_s = MAX(COALESCE(persons.last_seen_s, excluded.last_seen_s), COALESCE(excluded.last_seen_s, persons.last_seen_s)),
                   first_seen_ts = MIN(COALESCE(persons.first_seen_ts, excluded.first_seen_ts), COALESCE(excluded.first_seen_ts, persons.first_seen_ts)),
                   last_seen_ts = MAX(COALESCE(persons.last_seen_ts, excluded.last_seen_ts), COALESCE(excluded.last_seen_ts, persons.last_seen_ts)),
                   frame_count = persons.frame_count + excluded.frame_count,
                   peak_conf = MAX(persons.peak_conf, excluded.peak_conf),
                   bbox_json = CASE WHEN excluded.peak_conf > persons.peak_conf THEN excluded.bbox_json ELSE persons.bbox_json END,
                   bbox_time_s = CASE WHEN excluded.peak_conf > persons.peak_conf THEN excluded.bbox_time_s ELSE persons.bbox_time_s END,
                   bbox_frame = CASE WHEN excluded.peak_conf > persons.peak_conf THEN excluded.bbox_frame ELSE persons.bbox_frame END
            """,
            (video_id, track_id, gender, age_est, activity, pose, orientation,
             upper_color, lower_color, first_seen_s, last_seen_s, frame_count,
             peak_conf, bbox_json, bbox_time_s, bbox_frame,
             video_id, first_seen_s, video_id, last_seen_s),
        )

    def commit_persons(self) -> None:
        """Commit pending person inserts."""
        self.conn.commit()

    def update_person_gender(self, person_id: int, gender: str, gender_conf: float | None = None) -> None:
        """Update a person's gender estimate by row id."""
        self.conn.execute(
            "UPDATE persons SET gender = ? WHERE id = ?",
            (gender, person_id),
        )

    # ── Vehicle operations ────────────────────────────────────────────

    def upsert_vehicle(
        self,
        video_id: str,
        track_id: int,
        vehicle_type: str,
        color: str = "unknown",
        color_conf: float = 0.0,
        first_seen_s: float | None = None,
        last_seen_s: float | None = None,
        frame_count: int = 0,
        peak_conf: float = 0.0,
        bbox_json: str | None = None,
        bbox_time_s: float | None = None,
        bbox_frame: int | None = None,
    ) -> None:
        """Insert or update a tracked vehicle."""
        self.conn.execute(
            """INSERT INTO vehicles
               (video_id, track_id, vehicle_type, color, color_conf,
                first_seen_s, last_seen_s, frame_count, peak_conf, bbox_json,
                bbox_time_s, bbox_frame, first_seen_ts, last_seen_ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       (SELECT start_epoch FROM videos WHERE video_id = ?) + ?,
                       (SELECT start_epoch FROM videos WHERE video_id = ?) + ?)
               ON CONFLICT(video_id, track_id) DO UPDATE SET
                   vehicle_type = excluded.vehicle_type,
                   color = CASE WHEN excluded.color_conf > vehicles.color_conf THEN excluded.color ELSE vehicles.color END,
                   color_conf = MAX(vehicles.color_conf, excluded.color_conf),
                   first_seen_s = MIN(COALESCE(vehicles.first_seen_s, excluded.first_seen_s), COALESCE(excluded.first_seen_s, vehicles.first_seen_s)),
                   last_seen_s = MAX(COALESCE(vehicles.last_seen_s, excluded.last_seen_s), COALESCE(excluded.last_seen_s, vehicles.last_seen_s)),
                   first_seen_ts = MIN(COALESCE(vehicles.first_seen_ts, excluded.first_seen_ts), COALESCE(excluded.first_seen_ts, vehicles.first_seen_ts)),
                   last_seen_ts = MAX(COALESCE(vehicles.last_seen_ts, excluded.last_seen_ts), COALESCE(excluded.last_seen_ts, vehicles.last_seen_ts)),
                   frame_count = vehicles.frame_count + excluded.frame_count,
                   peak_conf = MAX(vehicles.peak_conf, excluded.peak_conf),
                   bbox_json = CASE WHEN excluded.peak_conf > vehicles.peak_conf THEN excluded.bbox_json ELSE vehicles.bbox_json END,
                   bbox_time_s = CASE WHEN excluded.peak_conf > vehicles.peak_conf THEN excluded.bbox_time_s ELSE vehicles.bbox_time_s END,
                   bbox_frame = CASE WHEN excluded.peak_conf > vehicles.peak_conf THEN excluded.bbox_frame ELSE vehicles.bbox_frame END
            """,
            (video_id, track_id, vehicle_type, color, color_conf,
             first_seen_s, last_seen_s, frame_count, peak_conf, bbox_json,
             bbox_time_s, bbox_frame,
             video_id, first_seen_s, video_id, last_seen_s),
        )

    def commit_vehicles(self) -> None:
        """Commit pending vehicle inserts."""
        self.conn.commit()

    # ── Object operations ─────────────────────────────────────────────

    def insert_object(
        self,
        video_id: str,
        track_id: int | None,
        object_type: str,
        color: str = "unknown",
        associated_person_track: int | None = None,
        first_seen_s: float | None = None,
        last_seen_s: float | None = None,
        frame_count: int = 0,
        peak_conf: float = 0.0,
        bbox_json: str | None = None,
        bbox_time_s: float | None = None,
        bbox_frame: int | None = None,
    ) -> None:
        """Insert a tracked object/clothing item."""
        self.conn.execute(
            """INSERT INTO objects
               (video_id, track_id, object_type, color, associated_person_track,
                first_seen_s, last_seen_s, frame_count, peak_conf, bbox_json,
                bbox_time_s, bbox_frame, first_seen_ts, last_seen_ts)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       (SELECT start_epoch FROM videos WHERE video_id = ?) + ?,
                       (SELECT start_epoch FROM videos WHERE video_id = ?) + ?)""",
            (video_id, track_id, object_type, color, associated_person_track,
             first_seen_s, last_seen_s, frame_count, peak_conf, bbox_json,
             bbox_time_s, bbox_frame,
             video_id, first_seen_s, video_id, last_seen_s),
        )

    def commit_objects(self) -> None:
        """Commit pending object inserts."""
        self.conn.commit()

    # ── Search / Query operations ─────────────────────────────────────

    def search_persons(
        self,
        video_id: str | None = None,
        gender: str | None = None,
        age_min: float | None = None,
        age_max: float | None = None,
        upper_color: str | None = None,
        lower_color: str | None = None,
        time_start: float | None = None,
        time_end: float | None = None,
        ts_from: float | None = None,
        ts_to: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search tracked persons with filters.

        time_start/time_end are relative video seconds (legacy).
        ts_from/ts_to are absolute UTC epoch seconds with overlap semantics:
        a track matches when its span [first_seen_ts, last_seen_ts] intersects
        [ts_from, ts_to]. Rows without absolute time are excluded when either
        absolute bound is given.
        """
        conditions: list[str] = []
        params: list[Any] = []
        if video_id:
            conditions.append("p.video_id = ?")
            params.append(video_id)
        if gender:
            conditions.append("LOWER(p.gender) = LOWER(?)")
            params.append(gender)
        if age_min is not None:
            conditions.append("p.age_est >= ?")
            params.append(age_min)
        if age_max is not None:
            conditions.append("p.age_est <= ?")
            params.append(age_max)
        if upper_color:
            conditions.append("LOWER(p.upper_color) = LOWER(?)")
            params.append(upper_color)
        if lower_color:
            conditions.append("LOWER(p.lower_color) = LOWER(?)")
            params.append(lower_color)
        if ts_from is not None:
            conditions.append("p.last_seen_ts >= ?")
            params.append(ts_from)
        if ts_to is not None:
            conditions.append("p.first_seen_ts <= ?")
            params.append(ts_to)
        if time_start is not None:
            conditions.append("p.last_seen_s >= ?")
            params.append(time_start)
        if time_end is not None:
            conditions.append("p.first_seen_s <= ?")
            params.append(time_end)

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        query = f"""
            SELECT p.*, v.file_path, v.camera_id, v.fps
            FROM persons p
            JOIN videos v ON p.video_id = v.video_id
            {where}
            ORDER BY p.first_seen_s ASC
        """
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def search_vehicles(
        self,
        video_id: str | None = None,
        vehicle_type: str | None = None,
        color: str | None = None,
        time_start: float | None = None,
        time_end: float | None = None,
        ts_from: float | None = None,
        ts_to: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search tracked vehicles with filters (ts_from/ts_to: absolute, overlap)."""
        conditions: list[str] = []
        params: list[Any] = []
        if video_id:
            conditions.append("v2.video_id = ?")
            params.append(video_id)
        if vehicle_type:
            conditions.append("LOWER(v2.vehicle_type) = LOWER(?)")
            params.append(vehicle_type)
        if color:
            conditions.append("LOWER(v2.color) = LOWER(?)")
            params.append(color)
        if ts_from is not None:
            conditions.append("v2.last_seen_ts >= ?")
            params.append(ts_from)
        if ts_to is not None:
            conditions.append("v2.first_seen_ts <= ?")
            params.append(ts_to)
        if time_start is not None:
            conditions.append("v2.last_seen_s >= ?")
            params.append(time_start)
        if time_end is not None:
            conditions.append("v2.first_seen_s <= ?")
            params.append(time_end)

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        query = f"""
            SELECT v2.*, v.file_path, v.camera_id, v.fps
            FROM vehicles v2
            JOIN videos v ON v2.video_id = v.video_id
            {where}
            ORDER BY v2.first_seen_s ASC
        """
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def search_objects(
        self,
        video_id: str | None = None,
        object_type: str | None = None,
        color: str | None = None,
        time_start: float | None = None,
        time_end: float | None = None,
        ts_from: float | None = None,
        ts_to: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search tracked objects/clothing with filters (ts_from/ts_to: absolute, overlap)."""
        conditions: list[str] = []
        params: list[Any] = []
        if video_id:
            conditions.append("o.video_id = ?")
            params.append(video_id)
        if object_type:
            conditions.append("LOWER(o.object_type) LIKE LOWER(?)")
            params.append(f"%{object_type}%")
        if color:
            conditions.append("LOWER(o.color) = LOWER(?)")
            params.append(color)
        if ts_from is not None:
            conditions.append("o.last_seen_ts >= ?")
            params.append(ts_from)
        if ts_to is not None:
            conditions.append("o.first_seen_ts <= ?")
            params.append(ts_to)
        if time_start is not None:
            conditions.append("o.last_seen_s >= ?")
            params.append(time_start)
        if time_end is not None:
            conditions.append("o.first_seen_s <= ?")
            params.append(time_end)

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        query = f"""
            SELECT o.*, v.file_path, v.camera_id, v.fps
            FROM objects o
            JOIN videos v ON o.video_id = v.video_id
            {where}
            ORDER BY o.first_seen_s ASC
        """
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_all_videos(self) -> list[dict[str, Any]]:
        """List all registered videos."""
        rows = self.conn.execute("SELECT * FROM videos ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]

    def get_video_summary(self, video_id: str) -> dict[str, Any]:
        """Get summary counts for a video."""
        person_count = self.conn.execute(
            "SELECT COUNT(*) FROM persons WHERE video_id = ?", (video_id,)
        ).fetchone()[0]
        vehicle_count = self.conn.execute(
            "SELECT COUNT(*) FROM vehicles WHERE video_id = ?", (video_id,)
        ).fetchone()[0]
        object_count = self.conn.execute(
            "SELECT COUNT(*) FROM objects WHERE video_id = ?", (video_id,)
        ).fetchone()[0]
        return {
            "video_id": video_id,
            "persons": person_count,
            "vehicles": vehicle_count,
            "objects": object_count,
        }

    def delete_video(self, video_id: str) -> None:
        """Cascade delete a video and all its detections."""
        self.conn.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))
        self.conn.commit()

    # ── Crop operations ────────────────────────────────────────────────

    def save_crop_record(
        self,
        video_id: str,
        entity_type: str,
        entity_id: int,
        track_id: int | None,
        timestamp_s: float | None,
        crop_path: str,
    ) -> None:
        """Store a representative crop image path for an entity."""
        self.conn.execute(
            """INSERT INTO crops
               (video_id, entity_type, entity_id, track_id, timestamp_s, ts, crop_path)
               VALUES (?, ?, ?, ?, ?, (SELECT start_epoch FROM videos WHERE video_id = ?) + ?, ?)
               ON CONFLICT(video_id, entity_type, track_id) DO UPDATE SET
                   entity_id = excluded.entity_id,
                   timestamp_s = excluded.timestamp_s,
                   ts = excluded.ts,
                   crop_path = excluded.crop_path""",
            (video_id, entity_type, entity_id, track_id, timestamp_s,
             video_id, timestamp_s, crop_path),
        )

    def update_representative_bbox(
        self,
        video_id: str,
        entity_type: str,
        track_id: int,
        bbox_json: str,
        timestamp_s: float,
        frame_index: int,
    ) -> None:
        """Store the bbox from the same frame as the representative crop."""
        table_by_entity = {
            "person": "persons",
            "vehicle": "vehicles",
        }
        table = table_by_entity.get(entity_type)
        if table is None:
            raise ValueError(f"Unsupported representative bbox entity: {entity_type}")
        self.conn.execute(
            f"""UPDATE {table}
                SET bbox_json = ?, bbox_time_s = ?, bbox_frame = ?
                WHERE video_id = ? AND track_id = ?""",
            (bbox_json, timestamp_s, frame_index, video_id, track_id),
        )

    def get_crop_path(self, entity_type: str, entity_id: int) -> str | None:
        """Retrieve the crop image path for an entity."""
        row = self.conn.execute(
            "SELECT crop_path FROM crops WHERE entity_type = ? AND entity_id = ?",
            (entity_type, entity_id),
        ).fetchone()
        return row["crop_path"] if row else None

    def get_crops_for_video(self, video_id: str) -> list[dict[str, Any]]:
        """Get all crop records for a video."""
        rows = self.conn.execute(
            "SELECT * FROM crops WHERE video_id = ? ORDER BY entity_type, track_id",
            (video_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Embedding operations ───────────────────────────────────────────

    def store_embedding(
        self,
        video_id: str,
        entity_type: str,
        entity_id: int,
        track_id: int | None,
        timestamp_s: float | None,
        embedding: bytes,
        crop_path: str | None = None,
        model: str | None = None,
    ) -> None:
        """Store a SigLIP 2 embedding vector (as BLOB) for an entity.

        model tags the embedding model; only embeddings sharing the same tag
        are ever compared with each other during vector search.
        """
        if model is None:
            from common.embed import get_model_tag
            model = get_model_tag()
        self.conn.execute(
            """INSERT INTO embeddings
               (video_id, entity_type, entity_id, track_id, timestamp_s, ts,
                model, embedding, crop_path)
               VALUES (?, ?, ?, ?, ?, (SELECT start_epoch FROM videos WHERE video_id = ?) + ?,
                       ?, ?, ?)""",
            (video_id, entity_type, entity_id, track_id, timestamp_s,
             video_id, timestamp_s, model, embedding, crop_path),
        )

    def get_all_embeddings(
        self,
        video_id: str | None = None,
        entity_type: str | None = None,
        model: str | None = None,
        ts_from: float | None = None,
        ts_to: float | None = None,
        min_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve stored embeddings with metadata, optionally filtered.

        model filters on embeddings.model so vectors from different encoders
        are never mixed; when omitted, embeddings from ALL models are returned.
        min_id enables incremental refresh (only rows with id >= min_id).
        ts_from/ts_to filter on the absolute ts column.
        """
        conditions: list[str] = []
        params: list[Any] = []
        if video_id:
            conditions.append("e.video_id = ?")
            params.append(video_id)
        if entity_type:
            conditions.append("e.entity_type = ?")
            params.append(entity_type)
        if model:
            conditions.append("e.model = ?")
            params.append(model)
        if ts_from is not None:
            conditions.append("e.ts >= ?")
            params.append(ts_from)
        if ts_to is not None:
            conditions.append("e.ts <= ?")
            params.append(ts_to)
        if min_id is not None:
            conditions.append("e.id >= ?")
            params.append(min_id)
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        rows = self.conn.execute(
            f"""SELECT e.id, e.video_id, e.entity_type, e.entity_id,
                       e.track_id, e.timestamp_s, e.ts, e.model,
                       e.embedding, e.crop_path, v.camera_id
                FROM embeddings e
                JOIN videos v ON e.video_id = v.video_id
                {where}
                ORDER BY e.id ASC""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_embedding_max_id(
        self, entity_type: str | None = None, model: str | None = None
    ) -> int:
        """Highest embeddings.id currently stored (0 when table is empty)."""
        conditions: list[str] = []
        params: list[Any] = []
        if entity_type:
            conditions.append("entity_type = ?")
            params.append(entity_type)
        if model:
            conditions.append("model = ?")
            params.append(model)
        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        row = self.conn.execute(
            f"SELECT COALESCE(MAX(id), 0) FROM embeddings {where}", params
        ).fetchone()
        return int(row[0] or 0)

    def get_track_embedding(
        self, entity_type: str, video_id: str, track_id: int, model: str | None = None
    ) -> list[dict[str, Any]]:
        """All stored embeddings for one track, newest first (for find-similar)."""
        params: list[Any] = [entity_type, video_id, track_id]
        model_clause = ""
        if model:
            model_clause = " AND e.model = ?"
            params.append(model)
        rows = self.conn.execute(
            f"""SELECT e.id, e.video_id, e.entity_type, e.entity_id, e.track_id,
                       e.timestamp_s, e.ts, e.model, e.embedding, e.crop_path, v.camera_id
                FROM embeddings e JOIN videos v ON e.video_id = v.video_id
                WHERE e.entity_type = ? AND e.video_id = ? AND e.track_id = ? {model_clause}
                ORDER BY e.id DESC""",
            params,
        ).fetchall()
        return [dict(r) for r in rows]

    def get_indexed_time_bounds(self, camera_id: str | None = None) -> tuple[float | None, float | None]:
        """Absolute UTC epoch second range of all indexed footage and detections.

        Considers videos.start_epoch..start_epoch+duration as footage bounds and
        the *_ts columns as detection bounds. Returns (min_ts, max_ts), either
        None when nothing is indexed (or nothing has a start_epoch).
        """
        min_ts: float | None = None
        max_ts: float | None = None

        def absorb(a: float | None, b: float | None) -> None:
            nonlocal min_ts, max_ts
            if a is not None:
                min_ts = a if min_ts is None else min(min_ts, a)
            if b is not None:
                max_ts = b if max_ts is None else max(max_ts, b)

        where, qparams = "", []
        if camera_id:
            where, qparams = " AND camera_id = ?", [camera_id]
        row = self.conn.execute(
            f"SELECT MIN(start_epoch), MAX(start_epoch + duration_s) FROM videos WHERE 1=1 {where}",
            qparams,
        ).fetchone()
        absorb(row[0], row[1])

        for table, col in (
            ("persons", "first_seen_ts"), ("persons", "last_seen_ts"),
            ("vehicles", "first_seen_ts"), ("vehicles", "last_seen_ts"),
            ("objects", "first_seen_ts"), ("objects", "last_seen_ts"),
            ("events", "ts"), ("embeddings", "ts"), ("crops", "ts"),
        ):
            dwhere, dparams = f"WHERE {col} IS NOT NULL", []
            if camera_id:
                dwhere += " AND video_id IN (SELECT video_id FROM videos WHERE camera_id = ?)"
                dparams.append(camera_id)
            row = self.conn.execute(f"SELECT MIN({col}), MAX({col}) FROM {table} {dwhere}", dparams).fetchone()
            absorb(row[0], row[1])

        return (min_ts, max_ts)

    def count_embeddings(self) -> int:
        """Count total embeddings in the database."""
        return self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]

    def commit(self) -> None:
        """General commit for any pending writes."""
        self.conn.commit()

    # ── Unified search for Streamlit app ───────────────────────────────

    def get_all_persons(self, video_id: str | None = None) -> list[dict[str, Any]]:
        """Get all tracked persons, optionally filtered by video."""
        if video_id:
            rows = self.conn.execute(
                "SELECT p.*, v.file_path, v.camera_id FROM persons p JOIN videos v ON p.video_id = v.video_id WHERE p.video_id = ? ORDER BY p.first_seen_s",
                (video_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT p.*, v.file_path, v.camera_id FROM persons p JOIN videos v ON p.video_id = v.video_id ORDER BY p.first_seen_s"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_vehicles_list(self, video_id: str | None = None) -> list[dict[str, Any]]:
        """Get all tracked vehicles, optionally filtered by video."""
        if video_id:
            rows = self.conn.execute(
                "SELECT v2.*, v.file_path, v.camera_id FROM vehicles v2 JOIN videos v ON v2.video_id = v.video_id WHERE v2.video_id = ? ORDER BY v2.first_seen_s",
                (video_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT v2.*, v.file_path, v.camera_id FROM vehicles v2 JOIN videos v ON v2.video_id = v.video_id ORDER BY v2.first_seen_s"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_all_objects_list(self, video_id: str | None = None) -> list[dict[str, Any]]:
        """Get all tracked objects, optionally filtered by video."""
        if video_id:
            rows = self.conn.execute(
                "SELECT o.*, v.file_path, v.camera_id FROM objects o JOIN videos v ON o.video_id = v.video_id WHERE o.video_id = ? ORDER BY o.first_seen_s",
                (video_id,),
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT o.*, v.file_path, v.camera_id FROM objects o JOIN videos v ON o.video_id = v.video_id ORDER BY o.first_seen_s"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_total_counts(self) -> dict[str, int]:
        """Get total counts across all videos."""
        p = self.conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]
        v = self.conn.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0]
        o = self.conn.execute("SELECT COUNT(*) FROM objects").fetchone()[0]
        e = self.conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        return {"persons": p, "vehicles": v, "objects": o, "events": e}

    # ── Event operations (Gate Line Crossing) ──────────────────────────

    def insert_event(
        self,
        video_id: str,
        event_type: str,
        entity_type: str,
        track_id: int,
        timestamp_s: float,
        line_name: str = "main_gate",
        confidence: float = 0.0,
        bbox_json: str | None = None,
        metadata_json: str | None = None,
    ) -> int:
        """Insert a directional line crossing event (e.g. entry / exit)."""
        cursor = self.conn.execute(
            """INSERT INTO events
               (video_id, event_type, entity_type, track_id, line_name, timestamp_s,
                ts, confidence, bbox_json, metadata_json)
               VALUES (?, ?, ?, ?, ?, ?, (SELECT start_epoch FROM videos WHERE video_id = ?) + ?, ?, ?, ?)""",
            (video_id, event_type, entity_type, track_id, line_name, timestamp_s,
             video_id, timestamp_s, confidence, bbox_json, metadata_json),
        )
        return int(cursor.lastrowid or 0)

    def commit_events(self) -> None:
        """Commit pending event inserts."""
        self.conn.commit()

    def search_events(
        self,
        video_id: str | None = None,
        event_type: str | None = None,
        entity_type: str | None = None,
        time_start: float | None = None,
        time_end: float | None = None,
        ts_from: float | None = None,
        ts_to: float | None = None,
    ) -> list[dict[str, Any]]:
        """Search line-crossing events with filters (ts_from/ts_to: absolute point range)."""
        conditions: list[str] = []
        params: list[Any] = []
        if video_id:
            conditions.append("e.video_id = ?")
            params.append(video_id)
        if event_type:
            conditions.append("LOWER(e.event_type) = LOWER(?)")
            params.append(event_type)
        if entity_type:
            conditions.append("LOWER(e.entity_type) = LOWER(?)")
            params.append(entity_type)
        if ts_from is not None:
            conditions.append("e.ts >= ?")
            params.append(ts_from)
        if ts_to is not None:
            conditions.append("e.ts <= ?")
            params.append(ts_to)
        if time_start is not None:
            conditions.append("e.timestamp_s >= ?")
            params.append(time_start)
        if time_end is not None:
            conditions.append("e.timestamp_s <= ?")
            params.append(time_end)

        where = (" WHERE " + " AND ".join(conditions)) if conditions else ""
        query = f"""
            SELECT e.*, v.file_path, v.camera_id, v.fps
            FROM events e
            JOIN videos v ON e.video_id = v.video_id
            {where}
            ORDER BY e.timestamp_s ASC
        """
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def get_all_events(self, video_id: str | None = None) -> list[dict[str, Any]]:
        """Retrieve all recorded gate crossing events."""
        return self.search_events(video_id=video_id)
