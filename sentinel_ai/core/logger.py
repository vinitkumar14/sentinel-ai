"""
SENTINEL-AI Structured Logging
================================
Production-grade, structured logging using loguru with:
  - Rotating file handlers (per log category)
  - JSON output for machine parsing
  - Colored console output with threat-level emoji
  - Context binding for per-request/per-flow correlation IDs

Usage:
    from sentinel_ai.core.logger import get_logger

    log = get_logger(__name__)
    log.info("Packet captured", src_ip="192.168.1.1", dst_port=80)
    log.bind(flow_id="flow-abc123").warning("Suspicious flow", score=8.7)
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from sentinel_ai.core.config import SentinelConfig

# ─────────────────────────────────────────────────────────────────────────────
# Module-level logger registry
# ─────────────────────────────────────────────────────────────────────────────

_initialized: bool = False


def get_logger(name: str) -> "logger":  # type: ignore[type-arg]
    """
    Return a loguru logger bound with a module name context.

    Args:
        name: Typically __name__ of the calling module.

    Returns:
        A loguru logger instance pre-bound with ``module=name``.
    """
    return logger.bind(module=name)


def setup_logging(config: "SentinelConfig | None" = None) -> None:
    """
    Configure all logging sinks for SENTINEL-AI.

    Must be called once at application startup before any logging occurs.
    If called multiple times, additional handlers are NOT added.

    Args:
        config: Loaded SentinelConfig. If None, uses sensible defaults.
    """
    global _initialized
    if _initialized:
        return

    # Remove default loguru sink
    logger.remove()

    # ── Determine paths ──────────────────────────────────────────────────────
    if config is not None:
        logs_dir = config.project_root / config.paths.logs_dir
        debug = config.system.debug
        env = config.system.environment
    else:
        logs_dir = Path("logs")
        debug = False
        env = "production"

    logs_dir.mkdir(parents=True, exist_ok=True)

    log_level = "DEBUG" if debug else "INFO"

    # ── Console sink (colored, human-friendly) ───────────────────────────────
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{extra[module]:<25}</cyan> | "
        "<white>{message}</white>"
    )

    logger.add(
        sys.stderr,
        format=console_format,
        level=log_level,
        colorize=True,
        backtrace=debug,
        diagnose=debug,
        filter=lambda record: record["extra"].get("module") is not None or True,
    )

    # ── Main log file (all levels, JSON for production) ──────────────────────
    main_log_path = logs_dir / "sentinel.log"
    logger.add(
        str(main_log_path),
        format="{time:ISO8601} | {level} | {extra[module]} | {message} | {extra}",
        level=log_level,
        rotation="50 MB",
        retention=10,
        compression="gz",
        encoding="utf-8",
        serialize=(env == "production"),     # JSON in production
        backtrace=debug,
        diagnose=debug,
        catch=True,
    )

    # ── Threat log (WARNING and above) ────────────────────────────────────────
    threat_log_path = logs_dir / "threats.log"
    logger.add(
        str(threat_log_path),
        format="{time:ISO8601} | {level} | {extra[module]} | {message} | {extra}",
        level="WARNING",
        rotation="100 MB",
        retention=20,
        compression="gz",
        encoding="utf-8",
        serialize=True,     # Always JSON for threat logs
        filter=_is_threat_record,
        catch=True,
    )

    # ── Defense action log ────────────────────────────────────────────────────
    defense_log_path = logs_dir / "defense.log"
    logger.add(
        str(defense_log_path),
        format="{time:ISO8601} | {level} | {extra[module]} | {message} | {extra}",
        level="INFO",
        rotation="50 MB",
        retention=10,
        compression="gz",
        encoding="utf-8",
        serialize=True,
        filter=_is_defense_record,
        catch=True,
    )

    # ── Audit log (security-critical, never compressed) ───────────────────────
    audit_log_path = logs_dir / "audit.log"
    logger.add(
        str(audit_log_path),
        format="{time:ISO8601} | {level} | AUDIT | {message} | {extra}",
        level="INFO",
        rotation="100 MB",
        retention=50,
        encoding="utf-8",
        serialize=True,
        filter=_is_audit_record,
        catch=True,
    )

    _initialized = True
    logger.bind(module="core.logger").info(
        "Logging initialized",
        logs_dir=str(logs_dir),
        level=log_level,
        environment=env,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Log record filters
# ─────────────────────────────────────────────────────────────────────────────


def _is_threat_record(record: dict) -> bool:
    """Filter: only threat-related log records (tagged with threat=True)."""
    return record["extra"].get("threat", False) is True


def _is_defense_record(record: dict) -> bool:
    """Filter: only defense-action log records (tagged with defense_action=True)."""
    return record["extra"].get("defense_action", False) is True


def _is_audit_record(record: dict) -> bool:
    """Filter: only security audit records (tagged with audit=True)."""
    return record["extra"].get("audit", False) is True


# ─────────────────────────────────────────────────────────────────────────────
# Helper: structured threat alert log
# ─────────────────────────────────────────────────────────────────────────────


def log_threat(
    attack_type: str,
    src_ip: str,
    dst_ip: str,
    confidence: float,
    threat_score: float,
    flow_id: str = "",
    **extra: object,
) -> None:
    """
    Emit a structured threat detection log entry.

    This record is routed to both the main log and the threats.log sink.

    Args:
        attack_type:  Detected attack class (e.g. "ddos", "portscan").
        src_ip:       Source IP address.
        dst_ip:       Destination IP address.
        confidence:   Model confidence (0–1).
        threat_score: Composite threat score (0–10).
        flow_id:      Optional flow correlation ID.
        **extra:      Any additional context.
    """
    logger.bind(
        module="threat.detector",
        threat=True,
        attack_type=attack_type,
        src_ip=src_ip,
        dst_ip=dst_ip,
        confidence=round(confidence, 4),
        threat_score=round(threat_score, 2),
        flow_id=flow_id,
        **extra,
    ).warning(
        f"🚨 {attack_type.upper()} detected | "
        f"src={src_ip} → dst={dst_ip} | "
        f"confidence={confidence:.1%} | score={threat_score:.1f}"
    )


def log_defense_action(
    action: str,
    target_ip: str,
    reason: str,
    duration: int = 0,
    dry_run: bool = True,
    **extra: object,
) -> None:
    """
    Emit a structured defense action log entry.

    Args:
        action:     The action taken (e.g. "block_ip", "rate_limit").
        target_ip:  The IP address affected.
        reason:     Human-readable reason for the action.
        duration:   Block duration in seconds (0 = permanent).
        dry_run:    True if the action was simulated (observe mode).
        **extra:    Any additional context.
    """
    mode = "DRY-RUN" if dry_run else "LIVE"
    logger.bind(
        module="defense.engine",
        defense_action=True,
        audit=True,
        action=action,
        target_ip=target_ip,
        reason=reason,
        duration=duration,
        dry_run=dry_run,
        **extra,
    ).info(
        f"🛡️ [{mode}] {action.upper()} | "
        f"target={target_ip} | "
        f"reason={reason} | "
        f"duration={duration}s"
    )
