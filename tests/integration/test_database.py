"""
SENTINEL-AI Integration Tests — Database
==========================================
Tests async SQLite operations: insert, query, pagination.
Uses an in-memory database (not the production file).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


@pytest.fixture
async def db():
    """Create an in-memory SQLite database for each test."""
    from sentinel_ai.storage.database import Database

    config = MagicMock()
    config.project_root = Path(".")
    config.storage.database_path = ":memory:"

    database = Database(config)
    # Patch the path to use in-memory SQLite
    database._db_path = Path(":memory:")
    await database.init()
    yield database
    await database.close()


def _make_result(attack_type: str = "ddos", src_ip: str = "1.2.3.4"):
    """Create a mock PredictionResult."""
    r = MagicMock()
    r.flow_id = f"test-{time.time()}"
    r.src_ip = src_ip
    r.dst_ip = "10.0.0.1"
    r.predicted_class = attack_type
    r.confidence = 0.92
    r.cnn_class = attack_type
    r.xgb_class = attack_type
    r.anomaly_score = 0.3
    r.inference_time_ms = 2.5
    r.timestamp = time.time()
    return r


class TestDatabase:
    """Integration tests for the async SQLite database."""

    @pytest.mark.asyncio
    async def test_insert_incident(self, db):
        result = _make_result("ddos")
        row_id = await db.insert_incident(result, 9.0, "CRITICAL", is_blocked=False)
        assert row_id > 0

    @pytest.mark.asyncio
    async def test_get_recent_incidents(self, db):
        result = _make_result("portscan", "5.6.7.8")
        await db.insert_incident(result, 5.0, "MEDIUM")
        incidents = await db.get_recent_incidents(limit=10)
        assert len(incidents) >= 1
        assert incidents[0]["attack_type"] == "portscan"

    @pytest.mark.asyncio
    async def test_filter_by_attack_type(self, db):
        await db.insert_incident(_make_result("ddos", "1.1.1.1"), 9.5, "CRITICAL")
        await db.insert_incident(_make_result("bot", "2.2.2.2"), 7.0, "HIGH")
        await db.insert_incident(_make_result("ddos", "3.3.3.3"), 8.5, "CRITICAL")

        ddos_only = await db.get_recent_incidents(attack_type="ddos")
        for inc in ddos_only:
            assert inc["attack_type"] == "ddos"

    @pytest.mark.asyncio
    async def test_threat_stats(self, db):
        for _ in range(3):
            await db.insert_incident(_make_result("bruteforce"), 6.0, "HIGH")
        for _ in range(2):
            await db.insert_incident(_make_result("portscan"), 4.0, "MEDIUM")

        stats = await db.get_threat_stats(window_seconds=3600)
        assert stats.get("bruteforce", 0) == 3
        assert stats.get("portscan", 0) == 2

    @pytest.mark.asyncio
    async def test_insert_blocked_ip(self, db):
        await db.insert_blocked_ip("10.20.30.40", "ddos", expires_at=None, dry_run=True)
        blocks = await db.get_active_blocks()
        ips = [b["ip"] for b in blocks]
        assert "10.20.30.40" in ips

    @pytest.mark.asyncio
    async def test_system_stats_insert(self, db):
        await db.insert_system_stats(
            packets=1000, flows=50, threats=5, benign=45, cpu=23.5, ram=61.2
        )
        # No exception = success

    @pytest.mark.asyncio
    async def test_pagination_limit(self, db):
        for i in range(10):
            await db.insert_incident(_make_result("bot", f"{i}.{i}.{i}.{i}"), 7.0, "HIGH")

        top5 = await db.get_recent_incidents(limit=5)
        assert len(top5) == 5

    @pytest.mark.asyncio
    async def test_since_filter(self, db):
        # Insert old record (fake old timestamp by using a past 'since')
        result = _make_result("heartbleed")
        await db.insert_incident(result, 10.0, "CRITICAL")

        future_cutoff = time.time() + 60  # Nothing is after this
        incidents = await db.get_recent_incidents(since=future_cutoff)
        assert len(incidents) == 0

    @pytest.mark.asyncio
    async def test_multiple_inserts_ordered_desc(self, db):
        for attack in ["bot", "ddos", "portscan"]:
            await db.insert_incident(_make_result(attack), 5.0, "MEDIUM")

        incidents = await db.get_recent_incidents(limit=100)
        # Most recent first
        for i in range(len(incidents) - 1):
            assert incidents[i]["created_at"] >= incidents[i + 1]["created_at"]
