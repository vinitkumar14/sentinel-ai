"""
SENTINEL-AI Dashboard Panels Package
=======================================
Modular Textual widget components for the main dashboard.
Each panel is a self-contained, testable widget.
"""

from sentinel_ai.dashboard.panels.stats_bar import StatsBarPanel
from sentinel_ai.dashboard.panels.traffic import LiveTrafficPanel
from sentinel_ai.dashboard.panels.threats import ThreatDetectionPanel
from sentinel_ai.dashboard.panels.ai_status import AIStatusPanel
from sentinel_ai.dashboard.panels.system import SystemMetricsPanel
from sentinel_ai.dashboard.panels.blocked_ips import BlockedIPsPanel

__all__ = [
    "StatsBarPanel",
    "LiveTrafficPanel",
    "ThreatDetectionPanel",
    "AIStatusPanel",
    "SystemMetricsPanel",
    "BlockedIPsPanel",
]
