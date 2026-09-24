"""
SENTINEL-AI — Stats Bar Panel
================================
Top header strip showing live counters:
  Packets | Flows | Threats | Blocked | Mode | Uptime
"""

from __future__ import annotations

import time
from textual.reactive import reactive
from textual.widgets import Static


class StatsBarPanel(Static):
    """
    Single-line status bar showing system-wide live counters.
    Auto-refreshes every second.
    """

    packets: reactive[int] = reactive(0)
    flows: reactive[int] = reactive(0)
    threats: reactive[int] = reactive(0)
    blocked: reactive[int] = reactive(0)
    mode: reactive[str] = reactive("OBSERVE")
    _start_time: float = 0.0

    DEFAULT_CSS = """
    StatsBarPanel {
        height: 1;
        background: #161b22;
        color: #e6edf3;
        padding: 0 2;
    }
    """

    def on_mount(self) -> None:
        self._start_time = time.time()
        self.set_interval(1.0, self.refresh)

    def render(self) -> str:
        uptime = int(time.time() - self._start_time)
        h, m, s = uptime // 3600, (uptime % 3600) // 60, uptime % 60
        mode_color = "#ff4500" if self.mode == "ACTIVE" else "#00ffff"
        return (
            f"[bold #00ffff]📡 PKT[/bold #00ffff] [white]{self.packets:,}[/white]"
            f"  [bold #00ffff]🌊 FLOWS[/bold #00ffff] [white]{self.flows:,}[/white]"
            f"  [bold #ff4500]🚨 THREATS[/bold #ff4500] [white]{self.threats:,}[/white]"
            f"  [bold #ffa500]🚫 BLOCKED[/bold #ffa500] [white]{self.blocked:,}[/white]"
            f"  [bold {mode_color}]⚙ {self.mode}[/bold {mode_color}]"
            f"  [#8b949e]⏱ {h:02d}:{m:02d}:{s:02d}[/#8b949e]"
        )

    def update_all(
        self, packets: int, flows: int, threats: int, blocked: int, mode: str
    ) -> None:
        """Batch update all counters at once."""
        self.packets = packets
        self.flows = flows
        self.threats = threats
        self.blocked = blocked
        self.mode = mode
