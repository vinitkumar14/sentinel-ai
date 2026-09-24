"""
SENTINEL-AI Async SQLite Database
====================================
Async-first SQLite persistence using aiosqlite.

Tables:
  - incidents:     Detected threat events
  - blocked_ips:   Current and historical IP blocks
  - system_stats:  Periodic system statistics snapshots

Usage:
    db = Database(config)
    await db.init()
    await db.insert_incident(prediction, threat_score)
    incidents = await db.get_recent_incidents(limit=50)
    await db.close()
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import aiosqlite

from sentinel_ai.core.logger import get_logger

if TYPE_CHECKING:
    from sentinel_ai.ai.ensemble_predictor import PredictionResult
    from sentinel_ai.core.config import SentinelConfig

log = get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Schema DDL
# ─────────────────────────────────────────────────────────────────────────────

_CREATE_INCIDENTS_TABLE = """
CREATE TABLE IF NOT EXISTS incidents (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    flow_id         TEXT,
    src_ip          TEXT NOT NULL,
    dst_ip          TEXT,
    attack_type     TEXT NOT NULL,
    confidence      REAL NOT NULL,
    threat_score    REAL NOT NULL,
    severity        TEXT NOT NULL,
    is_blocked      INTEGER DEFAULT 0,
    dry_run         INTEGER DEFAULT 1,
    cnn_class       TEXT,
    xgb_class       TEXT,
    anomaly_score   REAL,
    inference_ms    REAL,
    extra_json      TEXT,
    created_at      REAL NOT NULL DEFAULT (unixepoch('now', 'subsecond'))
);
"""

_CREATE_BLOCKED_IPS_TABLE = """
CREATE TABLE IF NOT EXISTS blocked_ips (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ip          TEXT NOT NULL,
    reason      TEXT NOT NULL,
    blocked_at  REAL NOT NULL,
    expires_at  REAL,
    unblocked_at REAL,
    dry_run     INTEGER DEFAULT 1,
    created_at  REAL NOT NULL DEFAULT (unixepoch('now', 'subsecond'))
);
"""

_CREATE_SYSTEM_STATS_TABLE = """
CREATE TABLE IF NOT EXISTS system_stats (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    packets_total   INTEGER DEFAULT 0,
    flows_total     INTEGER DEFAULT 0,
    threats_total   INTEGER DEFAULT 0,
    benign_total    INTEGER DEFAULT 0,
    cpu_percent     REAL,
    ram_percent     REAL,
    created_at      REAL NOT NULL DEFAULT (unixepoch('now', 'subsecond'))
);
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_incidents_src_ip ON incidents(src_ip);",
    "CREATE INDEX IF NOT EXISTS idx_incidents_attack ON incidents(attack_type);",
    "CREATE INDEX IF NOT EXISTS idx_incidents_created ON incidents(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_blocked_ip ON blocked_ips(ip);",
]


class Database:
    """
    Async SQLite database manager for SENTINEL-AI.

    Args:
        config: Loaded SentinelConfig.
    """

    def __init__(self, config: "SentinelConfig") -> None:
        self._config = config
        self._db_path = config.project_root / config.storage.database_path
        self._db: Optional[aiosqlite.Connection] = None
        self._initialized: bool = False

    async def init(self) -> None:
        """Initialize the database: create tables and indexes."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._db = await aiosqlite.connect(str(self._db_path))
        self._db.row_factory = aiosqlite.Row  # type: ignore[assignment]

        # WAL mode for better concurrent read performance
        await self._db.execute("PRAGMA journal_mode=WAL;")
        await self._db.execute("PRAGMA synchronous=NORMAL;")
        await self._db.execute("PRAGMA foreign_keys=ON;")

        # Create tables
        await self._db.execute(_CREATE_INCIDENTS_TABLE)
        await self._db.execute(_CREATE_BLOCKED_IPS_TABLE)
        await self._db.execute(_CREATE_SYSTEM_STATS_TABLE)
        for idx_sql in _CREATE_INDEXES:
            await self._db.execute(idx_sql)

        await self._db.commit()
        self._initialized = True

        log.info("Database initialized", path=str(self._db_path))

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None
        log.info("Database closed")

    # ── Incidents ─────────────────────────────────────────────────────────────

    async def insert_incident(
        self,
        result: "PredictionResult",
        threat_score: float,
        severity: str,
        is_blocked: bool = False,
        dry_run: bool = True,
    ) -> int:
        """
        Insert a threat incident record.

        Returns:
            Row ID of the inserted record.
        """
        if not self._db:
            return -1

        cursor = await self._db.execute(
            """
            INSERT INTO incidents
                (flow_id, src_ip, dst_ip, attack_type, confidence, threat_score,
                 severity, is_blocked, dry_run, cnn_class, xgb_class, anomaly_score,
                 inference_ms, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.flow_id,
                result.src_ip,
                result.dst_ip,
                result.predicted_class,
                round(result.confidence, 6),
                round(threat_score, 4),
                severity,
                int(is_blocked),
                int(dry_run),
                result.cnn_class,
                result.xgb_class,
                round(result.anomaly_score, 6),
                round(result.inference_time_ms, 2),
                result.timestamp,
            ),
        )
        await self._db.commit()
        return cursor.lastrowid or -1

    async def get_recent_incidents(
        self,
        limit: int = 50,
        attack_type: Optional[str] = None,
        since: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """
        Fetch recent incidents from the database.

        Args:
            limit:       Maximum number of records.
            attack_type: Filter by attack type.
            since:       Unix timestamp lower bound.

        Returns:
            List of incident dicts (newest first).
        """
        if not self._db:
            return []

        conditions = []
        params: list[Any] = []

        if attack_type:
            conditions.append("attack_type = ?")
            params.append(attack_type)
        if since:
            conditions.append("created_at >= ?")
            params.append(since)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        params.append(limit)

        async with self._db.execute(
            f"""
            SELECT * FROM incidents {where}
            ORDER BY created_at DESC LIMIT ?
            """,
            params,
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_threat_stats(
        self, window_seconds: int = 3600
    ) -> dict[str, int]:
        """Get per-attack-type counts for the last N seconds."""
        if not self._db:
            return {}

        since = time.time() - window_seconds
        async with self._db.execute(
            """
            SELECT attack_type, COUNT(*) as count
            FROM incidents
            WHERE created_at >= ?
            GROUP BY attack_type
            ORDER BY count DESC
            """,
            (since,),
        ) as cursor:
            rows = await cursor.fetchall()
            return {row["attack_type"]: row["count"] for row in rows}

    # ── Blocked IPs ───────────────────────────────────────────────────────────

    async def insert_blocked_ip(
        self,
        ip: str,
        reason: str,
        expires_at: Optional[float],
        dry_run: bool,
    ) -> None:
        """Record a block event."""
        if not self._db:
            return
        await self._db.execute(
            """
            INSERT INTO blocked_ips (ip, reason, blocked_at, expires_at, dry_run)
            VALUES (?, ?, ?, ?, ?)
            """,
            (ip, reason, time.time(), expires_at, int(dry_run)),
        )
        await self._db.commit()

    async def get_active_blocks(self) -> list[dict[str, Any]]:
        """Get currently active blocks (not yet expired or unblocked)."""
        if not self._db:
            return []
        now = time.time()
        async with self._db.execute(
            """
            SELECT * FROM blocked_ips
            WHERE unblocked_at IS NULL
              AND (expires_at IS NULL OR expires_at > ?)
            ORDER BY blocked_at DESC
            """,
            (now,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    # ── System stats ──────────────────────────────────────────────────────────

    async def insert_system_stats(
        self,
        packets: int,
        flows: int,
        threats: int,
        benign: int,
        cpu: float,
        ram: float,
    ) -> None:
        """Insert a periodic system snapshot."""
        if not self._db:
            return
        await self._db.execute(
            """
            INSERT INTO system_stats
                (packets_total, flows_total, threats_total, benign_total, cpu_percent, ram_percent)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (packets, flows, threats, benign, cpu, ram),
        )
        await self._db.commit()
