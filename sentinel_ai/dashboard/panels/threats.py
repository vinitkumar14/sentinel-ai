"""
SENTINEL-AI — Threat Detection Panel
=======================================
Shows recent threat detections with animated threat score bars.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from rich.table import Table
from rich.progress import BarColumn, Progress, TaskID
from rich import box
from textual.widgets import Static


@dataclass
class ThreatEvent:
    """A single detected threat event."""
    timestamp: float
    src_ip: str
    attack_type: str
    score: float
    severity: str
    confidence: float
    blocked: bool
    dry_run: bool


_SEVERITY_COLORS = {
    "CRITICAL": "#ff0033",
    "HIGH":     "#ff4500",
    "MEDIUM":   "#ffa500",
    "LOW":      "#ffff00",
    "INFO":     "#8b949e",
}

_ATTACK_EMOJIS = {
    "ddos":       "💥",
    "dos":        "🔴",
    "bruteforce": "🔨",
    "portscan":   "🔍",
    "bot":        "🤖",
    "heartbleed": "❤️",
    "webattack":  "🌐",
    "benign":     "✅",
}


class ThreatDetectionPanel(Static):
    """
    Recent threat detections with severity indicators and score bars.
    """

    DEFAULT_CSS = """
    ThreatDetectionPanel {
        height: 1fr;
        border: solid #30363d;
        background: #0d1117;
        padding: 0 1;
    }
    """

    def __init__(self, max_events: int = 15, **kwargs) -> None:
        super().__init__(**kwargs)
        self._events: deque[ThreatEvent] = deque(maxlen=max_events)

    def on_mount(self) -> None:
        self.set_interval(1.0, self.refresh)

    def add_threat(self, event: ThreatEvent) -> None:
        """Add a new threat event."""
        self._events.appendleft(event)

    def _score_bar(self, score: float) -> str:
        """Build a colored ASCII score bar."""
        filled = int((score / 10.0) * 12)
        empty = 12 - filled
        if score >= 9:
            color = "#ff0033"
        elif score >= 7:
            color = "#ff4500"
        elif score >= 5:
            color = "#ffa500"
        else:
            color = "#39ff14"
        return f"[{color}]{'█' * filled}{'░' * empty}[/{color}] {score:.1f}"

    def render(self) -> Table:
        table = Table(
            title="[bold #ff4500]🚨 THREAT DETECTIONS[/bold #ff4500]",
            box=box.SIMPLE_HEAD,
            border_style="#30363d",
            header_style="bold #8b949e",
            show_edge=False,
            expand=True,
        )
        table.add_column("Time", style="#8b949e", width=8)
        table.add_column("Src IP", style="#e6edf3", width=16)
        table.add_column("Attack", width=14)
        table.add_column("Severity", width=10)
        table.add_column("Score ────────────", width=22)
        table.add_column("Conf", justify="right", width=6)
        table.add_column("Status", width=10)

        for evt in self._events:
            ts = time.strftime("%H:%M:%S", time.localtime(evt.timestamp))
            emoji = _ATTACK_EMOJIS.get(evt.attack_type, "⚠️")
            sev_color = _SEVERITY_COLORS.get(evt.severity, "#8b949e")
            status = (
                "[bold #ffa500]DRY-RUN[/bold #ffa500]" if evt.dry_run and evt.blocked
                else "[bold #ff0033]BLOCKED[/bold #ff0033]" if evt.blocked
                else "[#8b949e]logged[/#8b949e]"
            )
            conf_color = "#39ff14" if evt.confidence >= 0.8 else "#ffa500"

            table.add_row(
                ts,
                evt.src_ip,
                f"{emoji} {evt.attack_type.upper():10}",
                f"[bold {sev_color}]{evt.severity}[/bold {sev_color}]",
                self._score_bar(evt.score),
                f"[{conf_color}]{evt.confidence:.2f}[/{conf_color}]",
                status,
            )

        if not self._events:
            table.add_row("—", "—", "—", "—",
                          "[#8b949e]No threats detected yet[/#8b949e]",
                          "—", "—")
        return table
