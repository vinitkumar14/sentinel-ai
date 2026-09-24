"""
SENTINEL-AI Defense Module
============================
IP blocking, rate limiting, alert management, and automated incident response.
All active defense actions require --live-defense flag to execute.
"""

from sentinel_ai.defense.ip_blocker import IPBlocker
from sentinel_ai.defense.alert_manager import AlertManager
from sentinel_ai.defense.response_engine import ResponseEngine

__all__ = ["IPBlocker", "AlertManager", "ResponseEngine"]
