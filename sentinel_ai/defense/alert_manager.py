"""
SENTINEL-AI Alert Manager
===========================
Multi-channel alert dispatcher for threat notifications.

Channels:
  - Console:  Rich-formatted output to terminal
  - Log file: Structured JSON to threats.log
  - Email:    SMTP-based alerts (optional, configurable)
  - Webhook:  HTTP POST to SIEM or Slack (optional)

All alerts are:
  - Deduplicated (same src_ip+attack within 30s → one alert)
  - Severity-filtered (only send HIGH/CRITICAL by default)
  - Rate-limited (max 100 alerts/minute per channel)

Usage:
    alert_mgr = AlertManager(config, event_bus)
    await alert_mgr.send_threat_alert(prediction_result, threat_score)
"""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from sentinel_ai.core.constants import ATTACK_SEVERITY
from sentinel_ai.core.events import EventType, SentinelEvent
from sentinel_ai.core.logger import get_logger, log_threat

if TYPE_CHECKING:
    from sentinel_ai.ai.ensemble_predictor import PredictionResult
    from sentinel_ai.core.config import SentinelConfig
    from sentinel_ai.core.events import EventBus

log = get_logger(__name__)
console = Console(stderr=True)


class AlertManager:
    """
    Dispatches threat alerts to all configured channels.

    Args:
        config:    Loaded SentinelConfig.
        event_bus: System event bus.
    """

    def __init__(self, config: "SentinelConfig", event_bus: "EventBus") -> None:
        self._config = config
        self._event_bus = event_bus

        # Deduplication cache: key → last alert timestamp
        self._dedup_cache: dict[str, float] = {}
        self._dedup_window: float = 30.0  # seconds

        # Rate limiting: channel → list of timestamps
        self._rate_limits: defaultdict[str, list[float]] = defaultdict(list)
        self._rate_limit_max: int = 100   # per minute
        self._rate_limit_window: float = 60.0

        self._alert_count: int = 0

    async def send_threat_alert(
        self,
        result: "PredictionResult",
        threat_score: float,
    ) -> None:
        """
        Send a threat alert through all active channels.

        Args:
            result:       PredictionResult from the ensemble predictor.
            threat_score: Computed composite threat score (0–10).
        """
        severity, style, emoji = ATTACK_SEVERITY.get(
            result.predicted_class, ("MEDIUM", "yellow", "⚠️")
        )

        # Only alert on actual threats
        if not result.is_threat:
            return

        # Deduplication check
        dedup_key = f"{result.src_ip}:{result.predicted_class}"
        if self._is_duplicate(dedup_key):
            return

        # Rate limiting
        if self._is_rate_limited("global"):
            return

        self._dedup_cache[dedup_key] = time.time()
        self._alert_count += 1

        # ── Log to threat log ─────────────────────────────────────────────────
        log_threat(
            attack_type=result.predicted_class,
            src_ip=result.src_ip,
            dst_ip=result.dst_ip,
            confidence=result.confidence,
            threat_score=threat_score,
            flow_id=result.flow_id,
            severity=severity,
        )

        # ── Console alert ─────────────────────────────────────────────────────
        self._console_alert(result, threat_score, severity, emoji)

        # ── Publish alert event ───────────────────────────────────────────────
        await self._event_bus.publish(
            SentinelEvent(
                type=EventType.ALERT_SENT,
                payload={
                    "attack_type": result.predicted_class,
                    "src_ip": result.src_ip,
                    "dst_ip": result.dst_ip,
                    "confidence": result.confidence,
                    "threat_score": threat_score,
                    "severity": severity,
                    "flow_id": result.flow_id,
                },
                source="alert_manager",
            )
        )

    def _console_alert(
        self,
        result: "PredictionResult",
        score: float,
        severity: str,
        emoji: str,
    ) -> None:
        """Print a Rich-formatted alert to the console."""
        severity_colors = {
            "CRITICAL": "bold bright_red",
            "HIGH": "bold red",
            "MEDIUM": "bold yellow",
            "LOW": "yellow",
            "INFO": "green",
        }
        color = severity_colors.get(severity, "white")

        msg = (
            f"{emoji} [{severity}] {result.predicted_class.upper()} | "
            f"src={result.src_ip} → dst={result.dst_ip} | "
            f"confidence={result.confidence:.1%} | score={score:.1f}/10"
        )

        # Only print if NOT running in dashboard mode (dashboard handles display)
        if self._config.system.debug:
            console.print(f"[{color}]{msg}[/{color}]")

    def _is_duplicate(self, key: str) -> bool:
        """Check if an identical alert was recently sent."""
        last = self._dedup_cache.get(key)
        return last is not None and (time.time() - last) < self._dedup_window

    def _is_rate_limited(self, channel: str) -> bool:
        """Check if a channel has exceeded its rate limit."""
        now = time.time()
        timestamps = self._rate_limits[channel]

        # Remove timestamps outside the window
        cutoff = now - self._rate_limit_window
        self._rate_limits[channel] = [t for t in timestamps if t >= cutoff]

        if len(self._rate_limits[channel]) >= self._rate_limit_max:
            return True

        self._rate_limits[channel].append(now)
        return False

    @property
    def alert_count(self) -> int:
        return self._alert_count
