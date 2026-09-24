"""
SENTINEL-AI — System Metrics Panel
=====================================
CPU, RAM, disk, network I/O, and packets/sec sparkline.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass

from rich.table import Table
from rich import box
from textual.widgets import Static


@dataclass
class SystemSnapshot:
    """Point-in-time system metrics."""
    cpu_percent: float = 0.0
    ram_percent: float = 0.0
    ram_used_gb: float = 0.0
    ram_total_gb: float = 0.0
    disk_percent: float = 0.0
    net_recv_mb: float = 0.0
    net_sent_mb: float = 0.0
    packets_per_sec: float = 0.0
    flows_per_sec: float = 0.0
    uptime_seconds: int = 0


def _bar(percent: float, width: int = 20) -> str:
    """Colored progress bar."""
    filled = int((percent / 100.0) * width)
    empty = width - filled
    if percent >= 90:
        color = "#ff0033"
    elif percent >= 70:
        color = "#ffa500"
    else:
        color = "#39ff14"
    return f"[{color}]{'█' * filled}{'░' * empty}[/{color}] {percent:.1f}%"


class SystemMetricsPanel(Static):
    """
    Live system resource usage with colored bars.
    """

    DEFAULT_CSS = """
    SystemMetricsPanel {
        height: 1fr;
        border: solid #30363d;
        background: #0d1117;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._snapshot = SystemSnapshot()
        self._pps_history: deque[float] = deque(maxlen=30)

    def on_mount(self) -> None:
        self.set_interval(2.0, self._collect_system_metrics)
        self.set_interval(1.0, self.refresh)

    def _collect_system_metrics(self) -> None:
        """Collect system metrics from psutil."""
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage("/")
            net = psutil.net_io_counters()

            self._snapshot.cpu_percent = cpu
            self._snapshot.ram_percent = mem.percent
            self._snapshot.ram_used_gb = mem.used / (1024 ** 3)
            self._snapshot.ram_total_gb = mem.total / (1024 ** 3)
            self._snapshot.disk_percent = disk.percent
            self._snapshot.net_recv_mb = net.bytes_recv / (1024 ** 2)
            self._snapshot.net_sent_mb = net.bytes_sent / (1024 ** 2)
        except ImportError:
            pass

    def update_traffic(self, pps: float, fps: float) -> None:
        """Update traffic metrics from the capture layer."""
        self._snapshot.packets_per_sec = pps
        self._snapshot.flows_per_sec = fps
        self._pps_history.append(pps)

    def _sparkline(self, data: deque, width: int = 20) -> str:
        """ASCII sparkline from a history deque."""
        if not data:
            return "[#8b949e]─" * width + "[/#8b949e]"
        blocks = " ▁▂▃▄▅▆▇█"
        max_val = max(data) or 1.0
        chars = [blocks[int((v / max_val) * 8)] for v in list(data)[-width:]]
        return "[#00ffff]" + "".join(chars).ljust(width) + "[/#00ffff]"

    def render(self) -> Table:
        s = self._snapshot
        table = Table(
            title="[bold #00ff88]💻 SYSTEM METRICS[/bold #00ff88]",
            box=box.SIMPLE_HEAD,
            border_style="#30363d",
            header_style="bold #8b949e",
            show_edge=False,
            expand=True,
        )
        table.add_column("Metric", width=18)
        table.add_column("Value", style="#e6edf3")

        table.add_row("CPU Usage",    _bar(s.cpu_percent))
        table.add_row("RAM Usage",    _bar(s.ram_percent))
        table.add_row("RAM Details",  f"{s.ram_used_gb:.1f}GB / {s.ram_total_gb:.1f}GB")
        table.add_row("Disk Usage",   _bar(s.disk_percent))
        table.add_row("", "")
        table.add_row("Net Recv",     f"{s.net_recv_mb:.1f} MB total")
        table.add_row("Net Sent",     f"{s.net_sent_mb:.1f} MB total")
        table.add_row("", "")
        table.add_row("Packets/sec",  f"[#00ffff]{s.packets_per_sec:.1f}[/#00ffff]")
        table.add_row("Flows/sec",    f"[#00ffff]{s.flows_per_sec:.2f}[/#00ffff]")
        table.add_row("PPS History",  self._sparkline(self._pps_history))

        return table


class BlockedIPsPanel(Static):
    """
    Compact list of currently blocked IPs.
    """

    DEFAULT_CSS = """
    BlockedIPsPanel {
        height: 1fr;
        border: solid #30363d;
        background: #0d1117;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._blocked: list[dict] = []

    def on_mount(self) -> None:
        self.set_interval(5.0, self.refresh)

    def update_blocked(self, blocked: list[dict]) -> None:
        """Update the list of blocked IPs."""
        self._blocked = blocked

    def render(self) -> Table:
        table = Table(
            title="[bold #ffa500]🚫 BLOCKED IPs[/bold #ffa500]",
            box=box.SIMPLE_HEAD,
            border_style="#30363d",
            header_style="bold #8b949e",
            show_edge=False,
            expand=True,
        )
        table.add_column("IP Address", width=16)
        table.add_column("Reason", width=14)
        table.add_column("Remaining", justify="right", width=10)
        table.add_column("Mode", width=8)

        for record in self._blocked[:15]:
            rem = record.get("remaining_seconds", 0)
            h, m, s = rem // 3600, (rem % 3600) // 60, rem % 60
            mode = "[#ffa500]DRY[/#ffa500]" if record.get("dry_run") else "[#ff0033]LIVE[/#ff0033]"
            table.add_row(
                record.get("ip", "—"),
                record.get("reason", "—")[:12],
                f"{h:02d}:{m:02d}:{s:02d}",
                mode,
            )

        if not self._blocked:
            table.add_row("[#8b949e]None[/#8b949e]", "—", "—", "—")

        return table
