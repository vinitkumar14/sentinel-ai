"""
Unit tests for SENTINEL-AI IP Blocker
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _make_config(live_defense: bool = False):
    cfg = MagicMock()
    cfg.live_defense = live_defense
    cfg.defense.whitelist = ["127.0.0.1", "::1", "10.0.0.0/8"]
    cfg.defense.block_duration = 3600
    cfg.defense.auto_block_enabled = False
    cfg.defense.thresholds.critical = 9.0
    return cfg


def _make_event_bus():
    bus = MagicMock()
    bus.publish = AsyncMock()
    return bus


class TestIPBlocker:
    """Tests for IPBlocker in observe mode."""

    @pytest.fixture
    def blocker(self):
        from sentinel_ai.defense.ip_blocker import IPBlocker
        return IPBlocker(_make_config(live_defense=False), _make_event_bus())

    @pytest.mark.asyncio
    async def test_block_ip_observe_mode(self, blocker):
        """Observe mode: block should be recorded but no firewall rule."""
        result = await blocker.block_ip("1.2.3.4", reason="portscan")
        assert result is True
        assert blocker.is_blocked("1.2.3.4")

    @pytest.mark.asyncio
    async def test_block_publishes_event(self, blocker):
        from sentinel_ai.core.events import EventType
        await blocker.block_ip("5.6.7.8", reason="ddos")
        # Event should have been published
        blocker._event_bus.publish.assert_called()

    @pytest.mark.asyncio
    async def test_whitelist_blocks_localhost(self, blocker):
        from sentinel_ai.core.exceptions import WhitelistViolationError
        with pytest.raises(WhitelistViolationError):
            await blocker.block_ip("127.0.0.1", reason="test")

    @pytest.mark.asyncio
    async def test_whitelist_rfc1918_range(self, blocker):
        """10.x.x.x is in whitelist CIDR — should raise."""
        from sentinel_ai.core.exceptions import WhitelistViolationError
        with pytest.raises(WhitelistViolationError):
            await blocker.block_ip("10.0.0.5", reason="test")

    @pytest.mark.asyncio
    async def test_invalid_ip_rejected(self, blocker):
        result = await blocker.block_ip("not-an-ip")
        assert result is False

    @pytest.mark.asyncio
    async def test_double_block_returns_true(self, blocker):
        await blocker.block_ip("1.2.3.4")
        result = await blocker.block_ip("1.2.3.4")
        assert result is True
        assert blocker.blocked_count == 1   # Still only one entry

    @pytest.mark.asyncio
    async def test_unblock_removes_ip(self, blocker):
        await blocker.block_ip("9.9.9.9")
        assert blocker.is_blocked("9.9.9.9")
        await blocker.unblock_ip("9.9.9.9")
        assert not blocker.is_blocked("9.9.9.9")

    @pytest.mark.asyncio
    async def test_unblock_non_existent_returns_false(self, blocker):
        result = await blocker.unblock_ip("1.1.1.1")
        assert result is False

    @pytest.mark.asyncio
    async def test_blocked_count_increments(self, blocker):
        await blocker.block_ip("11.0.0.1")
        await blocker.block_ip("11.0.0.2")
        await blocker.block_ip("11.0.0.3")
        assert blocker.blocked_count == 3

    @pytest.mark.asyncio
    async def test_blocked_ips_dict(self, blocker):
        await blocker.block_ip("20.0.0.1", reason="botnet")
        blocked = blocker.blocked_ips
        assert "20.0.0.1" in blocked
        assert blocked["20.0.0.1"].reason == "botnet"

    @pytest.mark.asyncio
    async def test_block_with_duration(self, blocker):
        await blocker.block_ip("30.0.0.1", duration=60)
        record = blocker.blocked_ips["30.0.0.1"]
        assert record.remaining_seconds <= 60
        assert record.remaining_seconds > 0

    @pytest.mark.asyncio
    async def test_permanent_block_duration_zero(self, blocker):
        await blocker.block_ip("40.0.0.1", duration=0)
        import math
        record = blocker.blocked_ips["40.0.0.1"]
        assert math.isinf(record.expires_at)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
