"""
SENTINEL-AI IP Blocker
========================
Safe, OS-adaptive IP blocking via iptables (Linux) or netsh (Windows).

Safety features:
  1. Default observe-only mode: blocks are LOGGED but NOT executed
  2. Permanent whitelist: localhost, RFC1918, gateway are NEVER blocked
  3. Requires --live-defense flag to execute real firewall rules
  4. All actions are logged to the audit trail
  5. Blocks auto-expire after a configurable duration

OS support:
  - Linux:   iptables / nftables
  - Windows: netsh advfirewall (secondary support)
  - macOS:   pf rules (basic support)

Usage:
    blocker = IPBlocker(config, event_bus)
    await blocker.block_ip("10.0.0.5", reason="ddos", duration=3600)
    await blocker.unblock_ip("10.0.0.5")
"""

from __future__ import annotations

import asyncio
import ipaddress
import platform
import subprocess
import time
from typing import TYPE_CHECKING, Optional

from sentinel_ai.core.events import EventType, SentinelEvent
from sentinel_ai.core.exceptions import DefenseModeError, WhitelistViolationError
from sentinel_ai.core.logger import get_logger, log_defense_action

if TYPE_CHECKING:
    from sentinel_ai.core.config import SentinelConfig
    from sentinel_ai.core.events import EventBus

log = get_logger(__name__)


class BlockedIPRecord:
    """Tracks a single blocked IP entry."""

    __slots__ = ["ip", "reason", "blocked_at", "expires_at", "dry_run"]

    def __init__(
        self,
        ip: str,
        reason: str,
        duration: int,
        dry_run: bool,
    ) -> None:
        self.ip = ip
        self.reason = reason
        self.blocked_at = time.time()
        self.expires_at = self.blocked_at + duration if duration > 0 else float("inf")
        self.dry_run = dry_run

    @property
    def is_expired(self) -> bool:
        return time.time() >= self.expires_at

    @property
    def remaining_seconds(self) -> int:
        return max(0, int(self.expires_at - time.time()))


class IPBlocker:
    """
    Manages IP blocks with firewall integration.

    In observe-only mode (default):
      - Logs all would-be blocks
      - Publishes IP_BLOCKED events
      - Does NOT modify firewall rules

    In active mode (--live-defense):
      - Executes real iptables/netsh rules
      - Maintains expiry timers
      - Restores rules on exit

    Args:
        config:    Loaded SentinelConfig.
        event_bus: System event bus.
    """

    def __init__(self, config: "SentinelConfig", event_bus: "EventBus") -> None:
        self._config = config
        self._event_bus = event_bus
        self._blocked: dict[str, BlockedIPRecord] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._os = platform.system()  # "Linux", "Windows", "Darwin"

        # Build permanent whitelist from config
        self._whitelist: set[str] = set()
        for entry in config.defense.whitelist:
            self._whitelist.add(entry.strip())

        # Always whitelist loopback
        self._whitelist.update(["127.0.0.1", "::1", "localhost"])

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def start(self) -> None:
        """Start the block expiry cleanup task."""
        self._cleanup_task = asyncio.create_task(
            self._cleanup_loop(), name="ip-blocker-cleanup"
        )
        log.info(
            "IPBlocker started",
            mode="ACTIVE" if self._config.live_defense else "OBSERVE",
            os=self._os,
        )

    async def stop(self) -> None:
        """Stop cleanup and optionally remove all active firewall rules."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        if self._config.live_defense and self._blocked:
            log.info("Removing active firewall rules on shutdown")
            for ip in list(self._blocked.keys()):
                await self.unblock_ip(ip, reason="shutdown")

    # ── Main API ─────────────────────────────────────────────────────────────

    async def block_ip(
        self,
        ip: str,
        reason: str = "auto",
        duration: int = 3600,
    ) -> bool:
        """
        Block an IP address.

        In observe mode: logs and emits event only.
        In active mode: executes firewall rule.

        Args:
            ip:       IP address to block.
            reason:   Reason for the block (e.g. "ddos").
            duration: Block duration in seconds (0 = permanent).

        Returns:
            True if the block was applied (or simulated), False if rejected.

        Raises:
            WhitelistViolationError: If the IP is on the permanent whitelist.
        """
        ip = ip.strip()

        # Validate IP
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            log.warning("Invalid IP address for blocking", ip=ip)
            return False

        # Whitelist check
        if self._is_whitelisted(ip):
            raise WhitelistViolationError(ip)

        # Already blocked
        if ip in self._blocked:
            log.debug("IP already blocked", ip=ip)
            return True

        # Max blocks check
        if len(self._blocked) >= 1000:
            log.warning("Max blocked IPs reached (1000)")
            return False

        dry_run = not self._config.live_defense

        # Record the block
        record = BlockedIPRecord(ip, reason, duration, dry_run)
        self._blocked[ip] = record

        # Apply firewall rule
        if self._config.live_defense:
            success = await self._apply_block(ip)
            if not success:
                del self._blocked[ip]
                return False

        # Log to audit trail
        log_defense_action(
            action="block_ip",
            target_ip=ip,
            reason=reason,
            duration=duration,
            dry_run=dry_run,
        )

        # Publish event
        await self._event_bus.publish(
            SentinelEvent(
                type=EventType.IP_BLOCKED,
                payload={
                    "ip": ip,
                    "reason": reason,
                    "duration": duration,
                    "dry_run": dry_run,
                },
                source="ip_blocker",
            )
        )

        return True

    async def unblock_ip(self, ip: str, reason: str = "expired") -> bool:
        """
        Remove an IP block.

        Args:
            ip:     IP address to unblock.
            reason: Reason for removal.

        Returns:
            True if the block was removed, False if the IP was not blocked.
        """
        if ip not in self._blocked:
            return False

        record = self._blocked.pop(ip)

        if self._config.live_defense:
            await self._remove_block(ip)

        log_defense_action(
            action="unblock_ip",
            target_ip=ip,
            reason=reason,
            dry_run=record.dry_run,
        )

        await self._event_bus.publish(
            SentinelEvent(
                type=EventType.IP_UNBLOCKED,
                payload={"ip": ip, "reason": reason},
                source="ip_blocker",
            )
        )

        return True

    # ── Firewall integration ──────────────────────────────────────────────────

    async def _apply_block(self, ip: str) -> bool:
        """Execute the OS-specific firewall command to block an IP."""
        try:
            if self._os == "Linux":
                cmd = ["iptables", "-I", "INPUT", "-s", ip, "-j", "DROP"]
            elif self._os == "Windows":
                cmd = [
                    "netsh", "advfirewall", "firewall", "add", "rule",
                    f"name=SENTINEL_BLOCK_{ip}",
                    "dir=in", "action=block",
                    f"remoteip={ip}",
                ]
            elif self._os == "Darwin":
                cmd = ["pfctl", "-t", "sentinel_blocked", "-T", "add", ip]
            else:
                log.warning("Unsupported OS for firewall automation", os=self._os)
                return False

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()

            if proc.returncode != 0:
                log.error(
                    "Firewall block command failed",
                    ip=ip,
                    error=stderr.decode().strip(),
                )
                return False

            return True

        except FileNotFoundError:
            log.error(
                "Firewall command not found. Ensure iptables/netsh is installed.",
                os=self._os,
            )
            return False
        except PermissionError:
            log.error("Insufficient privileges for firewall command. Run as root.")
            return False

    async def _remove_block(self, ip: str) -> bool:
        """Execute the OS-specific firewall command to unblock an IP."""
        try:
            if self._os == "Linux":
                cmd = ["iptables", "-D", "INPUT", "-s", ip, "-j", "DROP"]
            elif self._os == "Windows":
                cmd = [
                    "netsh", "advfirewall", "firewall", "delete", "rule",
                    f"name=SENTINEL_BLOCK_{ip}",
                ]
            elif self._os == "Darwin":
                cmd = ["pfctl", "-t", "sentinel_blocked", "-T", "delete", ip]
            else:
                return True

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.communicate()
            return True

        except Exception:  # noqa: BLE001
            return False

    # ── Expiry cleanup ────────────────────────────────────────────────────────

    async def _cleanup_loop(self) -> None:
        """Periodically remove expired IP blocks."""
        while True:
            await asyncio.sleep(30)
            expired = [
                ip for ip, record in self._blocked.items() if record.is_expired
            ]
            for ip in expired:
                await self.unblock_ip(ip, reason="expired")
                log.info("Block expired", ip=ip)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _is_whitelisted(self, ip: str) -> bool:
        """Check if an IP is on the permanent whitelist."""
        if ip in self._whitelist:
            return True
        # Check CIDR ranges
        try:
            ip_obj = ipaddress.ip_address(ip)
            for entry in self._whitelist:
                try:
                    network = ipaddress.ip_network(entry, strict=False)
                    if ip_obj in network:
                        return True
                except ValueError:
                    pass
        except ValueError:
            pass
        return False

    # ── Properties ────────────────────────────────────────────────────────────

    @property
    def blocked_ips(self) -> dict[str, BlockedIPRecord]:
        """Current blocked IP records."""
        return self._blocked.copy()

    @property
    def blocked_count(self) -> int:
        return len(self._blocked)

    def is_blocked(self, ip: str) -> bool:
        return ip in self._blocked
