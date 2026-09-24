"""
SENTINEL-AI Storage Data Models
==================================
Simple dataclass models for in-memory representation of database records.
(Not SQLAlchemy ORM — using aiosqlite directly for simplicity and performance)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Incident:
    """Represents a single threat detection incident."""

    flow_id: str = ""
    src_ip: str = ""
    dst_ip: str = ""
    attack_type: str = ""
    confidence: float = 0.0
    threat_score: float = 0.0
    severity: str = "INFO"
    is_blocked: bool = False
    dry_run: bool = True
    cnn_class: str = ""
    xgb_class: str = ""
    anomaly_score: float = 0.0
    inference_ms: float = 0.0
    created_at: float = field(default_factory=time.time)
    id: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "flow_id": self.flow_id,
            "src_ip": self.src_ip,
            "dst_ip": self.dst_ip,
            "attack_type": self.attack_type,
            "confidence": round(self.confidence, 4),
            "threat_score": round(self.threat_score, 2),
            "severity": self.severity,
            "is_blocked": self.is_blocked,
            "dry_run": self.dry_run,
            "created_at": self.created_at,
        }


@dataclass
class BlockedIP:
    """Represents a blocked IP record."""

    ip: str
    reason: str
    blocked_at: float = field(default_factory=time.time)
    expires_at: Optional[float] = None
    unblocked_at: Optional[float] = None
    dry_run: bool = True
    id: Optional[int] = None

    @property
    def is_active(self) -> bool:
        """True if the block is still active."""
        if self.unblocked_at is not None:
            return False
        if self.expires_at is None:
            return True
        return time.time() < self.expires_at

    @property
    def remaining_seconds(self) -> int:
        if self.expires_at is None:
            return -1  # Permanent
        return max(0, int(self.expires_at - time.time()))
