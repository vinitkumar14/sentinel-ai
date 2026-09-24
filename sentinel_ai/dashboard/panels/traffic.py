"""
SENTINEL-AI — Live Traffic Panel
===================================
Scrollable table of real-time network flows.
Color-coded by classification label.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rich.table import Table
from rich import box
from textual.widgets import Static

if TYPE_CHECKING:
    pass

_ATTACK_COLORS = {
    "benign":     "#39ff14",
    "ddos":       "#ff0033",
    "dos":        "#ff4500",
    "bruteforce": "#ffa500",
    "portscan":   "#00ffff",
    "bot":        "#ff1493",
    "heartbleed": "#8b00ff",
    "webattack":  "#ffff00",
    "unknown":    "#8b949e",
}


@dataclass
class FlowRow:
    """Single row in the traffic table."""
    timestamp: float
    src_ip: str
    dst_port: int
    protocol: str
    packets: int
    bytes_kb: float
    label: str
    confidence: float
    score: float


class LiveTrafficPanel(Static):
    """
    Scrollable live traffic table showing the last N flows.

    Displays:
      Time | Src IP | Port | Proto | Pkts | KB | Class | Conf | Score
    """

    DEFAULT_CSS = """
    LiveTrafficPanel {
        height: 1fr;
        border: solid #30363d;
        background: #0d1117;
        padding: 0 1;
    }
    """

    def __init__(self, max_rows: int = 20, **kwargs) -> None:
        super().__init__(**kwargs)
        self._max_rows = max_rows
        self._rows: deque[FlowRow] = deque(maxlen=max_rows)

    def on_mount(self) -> None:
        self.set_interval(1.0, self.refresh)

    def add_flow(self, row: FlowRow) -> None:
        """Add a new flow row to the live table."""
        self._rows.appendleft(row)

    def render(self) -> Table:
        table = Table(
            title="[bold #00ffff]⚡ LIVE TRAFFIC MONITOR[/bold #00ffff]",
            box=box.SIMPLE_HEAD,
            border_style="#30363d",
            header_style="bold #8b949e",
            show_edge=False,
            expand=True,
        )
        table.add_column("Time", style="#8b949e", width=8)
        table.add_column("Source IP", style="#e6edf3", width=15)
        table.add_column("Port", justify="right", width=6)
        table.add_column("Proto", width=6)
        table.add_column("Pkts", justify="right", width=6)
        table.add_column("KB", justify="right", width=8)
        table.add_column("Classification", width=14)
        table.add_column("Conf", justify="right", width=6)
        table.add_column("Score", justify="right", width=6)

        for row in self._rows:
            ts = time.strftime("%H:%M:%S", time.localtime(row.timestamp))
            color = _ATTACK_COLORS.get(row.label, "#8b949e")
            label_fmt = f"[bold {color}]{row.label.upper():12}[/bold {color}]"
            conf_color = "#39ff14" if row.confidence >= 0.8 else "#ffa500" if row.confidence >= 0.6 else "#ff0033"
            score_color = "#ff0033" if row.score >= 7 else "#ffa500" if row.score >= 4 else "#39ff14"

            table.add_row(
                ts,
                row.src_ip,
                str(row.dst_port),
                row.protocol,
                str(row.packets),
                f"{row.bytes_kb:.1f}",
                label_fmt,
                f"[{conf_color}]{row.confidence:.2f}[/{conf_color}]",
                f"[{score_color}]{row.score:.1f}[/{score_color}]",
            )

        if not self._rows:
            table.add_row("—", "—", "—", "—", "—", "—",
                          "[#8b949e]Waiting for traffic...[/#8b949e]",
                          "—", "—")
        return table
