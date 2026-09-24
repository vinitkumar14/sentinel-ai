"""
SENTINEL-AI Custom Exception Hierarchy
=======================================
All system-specific exceptions with rich context for debugging and alerting.

Design principles:
  - Every exception carries a structured payload (error_code, context dict)
  - Exceptions are namespaced by module for easy log filtering
  - Base class provides consistent __str__ and JSON serialization
"""

from __future__ import annotations

import json
from typing import Any, Optional


class SentinelBaseError(Exception):
    """
    Root exception for all SENTINEL-AI errors.

    Attributes:
        message:    Human-readable error description
        error_code: Machine-parseable identifier (e.g. "CAPTURE_001")
        context:    Arbitrary dict of extra debugging information
    """

    def __init__(
        self,
        message: str,
        error_code: str = "SENTINEL_ERROR",
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context: dict[str, Any] = context or {}

    def __str__(self) -> str:
        ctx = f" | context={json.dumps(self.context)}" if self.context else ""
        return f"[{self.error_code}] {self.message}{ctx}"

    def to_dict(self) -> dict[str, Any]:
        """Serialize exception to a structured dictionary (for logging/alerting)."""
        return {
            "error_code": self.error_code,
            "message": self.message,
            "exception_type": type(self).__name__,
            "context": self.context,
        }


# ── Configuration Errors ──────────────────────────────────────────────────────


class ConfigError(SentinelBaseError):
    """Raised when configuration is invalid or missing."""

    def __init__(self, message: str, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message, error_code="CONFIG_001", context=context)


class ConfigFileNotFoundError(ConfigError):
    """Raised when a required configuration file does not exist."""

    def __init__(self, path: str) -> None:
        super().__init__(
            f"Configuration file not found: {path}",
            context={"path": path},
        )
        self.error_code = "CONFIG_002"


class ConfigValidationError(ConfigError):
    """Raised when configuration values fail Pydantic validation."""

    def __init__(self, field: str, value: Any, reason: str) -> None:
        super().__init__(
            f"Invalid config value for '{field}': {reason}",
            context={"field": field, "value": str(value), "reason": reason},
        )
        self.error_code = "CONFIG_003"


# ── Capture / Network Errors ─────────────────────────────────────────────────


class CaptureError(SentinelBaseError):
    """Base error for packet capture subsystem."""

    def __init__(self, message: str, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message, error_code="CAPTURE_001", context=context)


class InterfaceNotFoundError(CaptureError):
    """Raised when the requested network interface does not exist."""

    def __init__(self, interface: str) -> None:
        super().__init__(
            f"Network interface not found: '{interface}'",
            context={"interface": interface},
        )
        self.error_code = "CAPTURE_002"


class PermissionError(CaptureError):
    """Raised when packet capture requires elevated privileges."""

    def __init__(self) -> None:
        super().__init__(
            "Packet capture requires root/administrator privileges. "
            "Run with sudo or as Administrator.",
        )
        self.error_code = "CAPTURE_003"


class PacketQueueFullError(CaptureError):
    """Raised when the async packet queue is saturated."""

    def __init__(self, queue_size: int) -> None:
        super().__init__(
            f"Packet queue is full (capacity={queue_size}). Packets are being dropped.",
            context={"queue_size": queue_size},
        )
        self.error_code = "CAPTURE_004"


class PcapFileError(CaptureError):
    """Raised when a PCAP file cannot be read or written."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(
            f"PCAP file error '{path}': {reason}",
            context={"path": path, "reason": reason},
        )
        self.error_code = "CAPTURE_005"


# ── AI / Model Errors ─────────────────────────────────────────────────────────


class ModelError(SentinelBaseError):
    """Base error for AI/ML model subsystem."""

    def __init__(self, message: str, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message, error_code="MODEL_001", context=context)


class ModelNotFoundError(ModelError):
    """Raised when a model file does not exist at the expected path."""

    def __init__(self, model_name: str, path: str) -> None:
        super().__init__(
            f"Model '{model_name}' not found at path: {path}",
            context={"model_name": model_name, "path": path},
        )
        self.error_code = "MODEL_002"


class ModelLoadError(ModelError):
    """Raised when a model cannot be deserialized or loaded."""

    def __init__(self, model_name: str, reason: str) -> None:
        super().__init__(
            f"Failed to load model '{model_name}': {reason}",
            context={"model_name": model_name, "reason": reason},
        )
        self.error_code = "MODEL_003"


class InferenceError(ModelError):
    """Raised when model inference fails for a given input."""

    def __init__(self, model_name: str, reason: str) -> None:
        super().__init__(
            f"Inference failed for model '{model_name}': {reason}",
            context={"model_name": model_name, "reason": reason},
        )
        self.error_code = "MODEL_004"


class ArtifactNotFoundError(ModelError):
    """Raised when a preprocessing artifact (scaler, encoder) is missing."""

    def __init__(self, artifact_name: str, path: str) -> None:
        super().__init__(
            f"Preprocessing artifact '{artifact_name}' not found at: {path}",
            context={"artifact_name": artifact_name, "path": path},
        )
        self.error_code = "MODEL_005"


class DriftDetectedError(ModelError):
    """Raised (or emitted as event) when concept drift is detected."""

    def __init__(self, drift_score: float, threshold: float) -> None:
        super().__init__(
            f"Concept drift detected: score={drift_score:.4f} > threshold={threshold:.4f}",
            context={"drift_score": drift_score, "threshold": threshold},
        )
        self.error_code = "MODEL_006"


# ── Defense Errors ────────────────────────────────────────────────────────────


class DefenseError(SentinelBaseError):
    """Base error for the defense engine subsystem."""

    def __init__(self, message: str, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message, error_code="DEFENSE_001", context=context)


class FirewallError(DefenseError):
    """Raised when a firewall rule cannot be applied."""

    def __init__(self, command: str, reason: str) -> None:
        super().__init__(
            f"Firewall command failed — {reason}: {command}",
            context={"command": command, "reason": reason},
        )
        self.error_code = "DEFENSE_002"


class WhitelistViolationError(DefenseError):
    """Raised when an attempt is made to block a whitelisted IP."""

    def __init__(self, ip: str) -> None:
        super().__init__(
            f"Cannot block whitelisted IP: {ip}",
            context={"ip": ip},
        )
        self.error_code = "DEFENSE_003"


class DefenseModeError(DefenseError):
    """Raised when an active defense action is attempted in observe-only mode."""

    def __init__(self, action: str) -> None:
        super().__init__(
            f"Cannot execute '{action}' in observe-only mode. "
            "Start with --live-defense flag to enable active defense.",
            context={"action": action},
        )
        self.error_code = "DEFENSE_004"


# ── Storage Errors ────────────────────────────────────────────────────────────


class StorageError(SentinelBaseError):
    """Base error for the storage/database subsystem."""

    def __init__(self, message: str, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message, error_code="STORAGE_001", context=context)


class DatabaseConnectionError(StorageError):
    """Raised when the database cannot be opened or connected."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(
            f"Cannot connect to database '{path}': {reason}",
            context={"path": path, "reason": reason},
        )
        self.error_code = "STORAGE_002"


# ── Dashboard Errors ──────────────────────────────────────────────────────────


class DashboardError(SentinelBaseError):
    """Base error for the TUI dashboard subsystem."""

    def __init__(self, message: str, context: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message, error_code="DASHBOARD_001", context=context)
