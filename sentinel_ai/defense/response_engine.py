"""
SENTINEL-AI Automated Response Engine
=======================================
Orchestrates the full threat response pipeline:

  1. Receive a PredictionResult + threat score
  2. Evaluate severity against configured thresholds
  3. In observe mode:   log + alert + emit events
  4. In active mode:    block IP + alert + emit events
  5. Update River online learner with confirmed labels
  6. Record incident in database

Usage:
    engine = ResponseEngine(config, ip_blocker, alert_manager, river_learner, event_bus)
    await engine.respond(prediction_result, threat_score, flow)
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Optional

from sentinel_ai.core.constants import ATTACK_SEVERITY
from sentinel_ai.core.events import EventType, SentinelEvent
from sentinel_ai.core.exceptions import WhitelistViolationError
from sentinel_ai.core.logger import get_logger

if TYPE_CHECKING:
    from sentinel_ai.ai.ensemble_predictor import PredictionResult
    from sentinel_ai.ai.river_learner import RiverLearner
    from sentinel_ai.capture.flow_manager import NetworkFlow
    from sentinel_ai.core.config import SentinelConfig
    from sentinel_ai.core.events import EventBus
    from sentinel_ai.defense.alert_manager import AlertManager
    from sentinel_ai.defense.ip_blocker import IPBlocker

import numpy as np

log = get_logger(__name__)


class ResponseEngine:
    """
    Central coordinator for the defense response pipeline.

    Args:
        config:        Loaded SentinelConfig.
        ip_blocker:    IPBlocker instance.
        alert_manager: AlertManager instance.
        river_learner: RiverLearner for online model updates.
        event_bus:     System event bus.
    """

    def __init__(
        self,
        config: "SentinelConfig",
        ip_blocker: "IPBlocker",
        alert_manager: "AlertManager",
        river_learner: "RiverLearner",
        event_bus: "EventBus",
    ) -> None:
        self._config = config
        self._ip_blocker = ip_blocker
        self._alert_manager = alert_manager
        self._river_learner = river_learner
        self._event_bus = event_bus
        self._incidents_handled: int = 0

    async def respond(
        self,
        result: "PredictionResult",
        threat_score: float,
        features: Optional[np.ndarray] = None,
    ) -> None:
        """
        Execute the full response pipeline for a detected threat.

        Args:
            result:       PredictionResult from the ensemble.
            threat_score: Composite threat score (0–10).
            features:     Feature vector for online learning (optional).
        """
        if not result.is_threat:
            # Benign flow — still update online learner
            if features is not None:
                await self._river_learner.learn(features, "benign")
            return

        self._incidents_handled += 1
        attack_type = result.predicted_class
        src_ip = result.src_ip
        severity = self._threat_scorer_severity(threat_score)

        log.info(
            "Threat response initiated",
            attack=attack_type,
            src_ip=src_ip,
            score=round(threat_score, 2),
            severity=severity,
            mode="ACTIVE" if self._config.live_defense else "OBSERVE",
        )

        # ── 1. Send alert ─────────────────────────────────────────────────────
        await self._alert_manager.send_threat_alert(result, threat_score)

        # ── 2. Auto-block if threshold exceeded ───────────────────────────────
        if self._should_block(threat_score):
            block_duration = self._get_block_duration(attack_type)
            try:
                blocked = await self._ip_blocker.block_ip(
                    src_ip,
                    reason=attack_type,
                    duration=block_duration,
                )
                if blocked and not self._config.live_defense:
                    log.info(
                        "⚠️ Observe mode: Would have blocked IP",
                        ip=src_ip,
                        duration=block_duration,
                    )
            except WhitelistViolationError:
                log.debug("Skipping block — IP is whitelisted", ip=src_ip)

        # ── 3. Update online learner ──────────────────────────────────────────
        if features is not None:
            await self._river_learner.learn(features, attack_type)

        # ── 4. Publish THREAT_DETECTED event ────────────────────────────────
        await self._event_bus.publish(
            SentinelEvent(
                type=EventType.THREAT_DETECTED,
                payload={
                    "attack_type": attack_type,
                    "src_ip": src_ip,
                    "dst_ip": result.dst_ip,
                    "confidence": result.confidence,
                    "threat_score": threat_score,
                    "severity": severity,
                    "flow_id": result.flow_id,
                    "timestamp": result.timestamp,
                    "is_blocked": self._ip_blocker.is_blocked(src_ip),
                    "dry_run": not self._config.live_defense,
                },
                source="response_engine",
            )
        )

    def _should_block(self, threat_score: float) -> bool:
        """Determine if a block should be attempted based on score."""
        return (
            self._config.defense.auto_block_enabled
            or self._config.live_defense
        ) and threat_score >= self._config.defense.thresholds.critical

    def _get_block_duration(self, attack_type: str) -> int:
        """Get the configured block duration for an attack type."""
        durations = {
            "ddos":       7200,
            "bruteforce": 3600,
            "portscan":   1800,
            "bot":        86400,
            "webattack":  3600,
            "dos":        7200,
            "heartbleed": 86400,
        }
        return durations.get(attack_type, self._config.defense.block_duration)

    def _threat_scorer_severity(self, score: float) -> str:
        """Map score to severity string."""
        thresholds = self._config.defense.thresholds
        if score >= thresholds.critical:
            return "CRITICAL"
        if score >= thresholds.high:
            return "HIGH"
        if score >= thresholds.medium:
            return "MEDIUM"
        return "LOW"

    @property
    def incidents_handled(self) -> int:
        return self._incidents_handled
