"""
SENTINEL-AI Main Textual Dashboard Application
================================================
Real-time TUI dashboard with cybersecurity aesthetics.

Layout:
  ┌─────────────────────────────────────────────────────┐
  │          SENTINEL-AI HEADER BAR                     │
  ├──────────────┬──────────────┬───────────────────────┤
  │ LIVE TRAFFIC │ THREAT INTEL │ AI MODEL STATUS       │
  │   (packets)  │ (detections) │ (confidence, drift)   │
  ├──────────────┴──────────────┴───────────────────────┤
  │ SYSTEM METRICS   (CPU/RAM/Network/Blocked IPs)       │
  ├──────────────────────────────────────────────────────┤
  │ LOG STREAM      (live rolling event log)             │
  └──────────────────────────────────────────────────────┘

Keyboard shortcuts:
  q / Ctrl+C  → Quit
  p           → Pause/Resume capture
  d           → Toggle demo mode
  b           → View blocked IPs
  h           → Help
  r           → Reset statistics
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from typing import TYPE_CHECKING, Any, Optional

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical
from textual.reactive import reactive
from textual.widgets import (
    DataTable,
    Footer,
    Header,
    Label,
    Log,
    ProgressBar,
    RichLog,
    Static,
)
from rich.text import Text
from rich.table import Table
from rich.panel import Panel
from rich.console import Console
from rich.align import Align
from rich import box

from sentinel_ai.dashboard.theme import ATTACK_EMOJIS, COLORS, SENTINEL_THEME, SEVERITY_STYLES

if TYPE_CHECKING:
    from sentinel_ai.ai.ensemble_predictor import PredictionResult
    from sentinel_ai.core.config import SentinelConfig
    from sentinel_ai.core.events import EventBus


BANNER = r"""
  ███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗██╗      █████╗ ██╗
  ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝██║     ██╔══██╗██║
  ███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗  ██║     ███████║██║
  ╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝  ██║     ██╔══██║╚═╝
  ███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗███████╗██║  ██║██╗
  ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝
                        AI CYBER DEFENSE SYSTEM v1.0
"""


# ─────────────────────────────────────────────────────────────────────────────
# Live stats panel
# ─────────────────────────────────────────────────────────────────────────────


class StatsBar(Static):
    """Top header bar with live packet and threat counters."""

    packets = reactive(0)
    flows = reactive(0)
    threats = reactive(0)
    blocked = reactive(0)
    mode = reactive("OBSERVE")
    uptime_start: float = 0.0

    def on_mount(self) -> None:
        self.uptime_start = time.time()
        self.set_interval(1.0, self.refresh_stats)

    def refresh_stats(self) -> None:
        self.refresh()

    def render(self) -> str:
        uptime = int(time.time() - self.uptime_start)
        h, m, s = uptime // 3600, (uptime % 3600) // 60, uptime % 60
        mode_color = "#ff4500" if self.mode == "ACTIVE" else "#00ffff"
        return (
            f"[bold #00ffff]📡 PACKETS[/bold #00ffff] [white]{self.packets:,}[/white]   "
            f"[bold #00ffff]🌊 FLOWS[/bold #00ffff] [white]{self.flows:,}[/white]   "
            f"[bold #ff4500]🚨 THREATS[/bold #ff4500] [white]{self.threats:,}[/white]   "
            f"[bold #ffa500]🚫 BLOCKED[/bold #ffa500] [white]{self.blocked:,}[/white]   "
            f"[bold {mode_color}]⚙ MODE[/bold {mode_color}] [{mode_color}]{self.mode}[/{mode_color}]   "
            f"[bold #8b949e]⏱ UPTIME[/bold #8b949e] [white]{h:02d}:{m:02d}:{s:02d}[/white]"
        )


class TrafficPanel(Static):
    """Live packet stream panel showing recent flows."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._recent_flows: deque[dict] = deque(maxlen=20)

    def add_flow(self, src_ip: str, dst_ip: str, proto: str, size: int, label: str) -> None:
        """Add a new flow to the display."""
        self._recent_flows.appendleft({
            "ts": time.strftime("%H:%M:%S"),
            "src": src_ip,
            "dst": dst_ip,
            "proto": proto,
            "size": size,
            "label": label,
        })
        self.refresh()

    def render(self) -> Table:
        table = Table(
            title="[bold #00ffff]📡 LIVE TRAFFIC[/bold #00ffff]",
            box=box.SIMPLE_HEAVY,
            border_style="#30363d",
            header_style="bold #8b949e",
            show_header=True,
            padding=(0, 1),
        )
        table.add_column("Time", style="#8b949e", width=9)
        table.add_column("Source IP", style="#00d4ff", width=16)
        table.add_column("→ Destination", style="#e6edf3", width=16)
        table.add_column("Proto", width=6, justify="center")
        table.add_column("Bytes", width=7, justify="right")
        table.add_column("Classification", width=14)

        for flow in list(self._recent_flows)[:18]:
            label = flow["label"]
            severity_style = SEVERITY_STYLES.get(
                "BENIGN" if label == "benign" else "HIGH", "#e6edf3"
            )
            emoji = ATTACK_EMOJIS.get(label, "❓")
            table.add_row(
                flow["ts"],
                flow["src"],
                flow["dst"],
                flow["proto"],
                str(flow["size"]),
                f"[{severity_style}]{emoji} {label}[/{severity_style}]",
            )

        if not self._recent_flows:
            table.add_row(
                "...", "Waiting for traffic...", "", "", "", ""
            )

        return table


class ThreatPanel(Static):
    """Recent threat detections with severity indicators."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._threats: deque[dict] = deque(maxlen=15)

    def add_threat(self, result: dict) -> None:
        self._threats.appendleft({**result, "ts": time.strftime("%H:%M:%S")})
        self.refresh()

    def render(self) -> Table:
        table = Table(
            title="[bold #ff4500]🚨 THREAT DETECTIONS[/bold #ff4500]",
            box=box.SIMPLE_HEAVY,
            border_style="#30363d",
            header_style="bold #8b949e",
            show_header=True,
            padding=(0, 1),
        )
        table.add_column("Time", style="#8b949e", width=9)
        table.add_column("Attack", width=12)
        table.add_column("Source IP", style="#ff6b6b", width=16)
        table.add_column("Score", width=6, justify="right")
        table.add_column("Conf", width=7, justify="right")
        table.add_column("Sev", width=9, justify="center")

        for t in list(self._threats)[:13]:
            sev = t.get("severity", "MEDIUM")
            sev_style = SEVERITY_STYLES.get(sev, "white")
            emoji = ATTACK_EMOJIS.get(t.get("attack_type", ""), "⚠️")
            score = t.get("threat_score", 0.0)
            score_bar = "█" * int(score) + "░" * (10 - int(score))
            table.add_row(
                t["ts"],
                f"[{sev_style}]{emoji} {t.get('attack_type', '?')}[/{sev_style}]",
                t.get("src_ip", "?"),
                f"[bold]{score:.1f}[/bold]",
                f"{t.get('confidence', 0):.1%}",
                f"[{sev_style}]{sev}[/{sev_style}]",
            )

        if not self._threats:
            table.add_row("...", "[#39ff14]No threats detected[/#39ff14]", "", "", "", "")

        return table


class AIStatusPanel(Static):
    """AI model status, confidence, and drift indicators."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._models: dict[str, bool] = {
            "CNN+BiLSTM": False,
            "XGBoost": False,
            "IsoForest": False,
            "River": False,
        }
        self._last_prediction: dict = {}
        self._drift_detected: bool = False
        self._river_updates: int = 0
        self._inference_ms: float = 0.0

    def update_models(self, models_ready: dict) -> None:
        self._models = {
            "CNN+BiLSTM": models_ready.get("cnn_bilstm", False),
            "XGBoost":    models_ready.get("xgboost", False),
            "IsoForest":  models_ready.get("isolation_forest", False),
            "River":      models_ready.get("river", False),
        }
        self.refresh()

    def update_prediction(self, result: dict) -> None:
        self._last_prediction = result
        self._inference_ms = result.get("inference_time_ms", 0.0)
        self.refresh()

    def set_drift(self, detected: bool) -> None:
        self._drift_detected = detected
        self.refresh()

    def render(self) -> str:
        lines = ["[bold #00ffff]🤖 AI ENGINE STATUS[/bold #00ffff]\n"]

        # Model readiness
        for name, ready in self._models.items():
            icon = "✅" if ready else "❌"
            color = "#39ff14" if ready else "#ff0033"
            lines.append(f"  {icon} [{color}]{name}[/{color}]")

        lines.append("")

        # Last prediction
        if self._last_prediction:
            cls = self._last_prediction.get("predicted_class", "?")
            conf = self._last_prediction.get("confidence", 0.0)
            emoji = ATTACK_EMOJIS.get(cls, "❓")
            cls_color = "#39ff14" if cls == "benign" else "#ff4500"
            lines.append(f"  [#8b949e]Last:[/#8b949e] [{cls_color}]{emoji} {cls}[/{cls_color}]")
            lines.append(f"  [#8b949e]Conf:[/#8b949e] [white]{conf:.1%}[/white]")
            lines.append(f"  [#8b949e]Time:[/#8b949e] [white]{self._inference_ms:.1f}ms[/white]")
        else:
            lines.append("  [#8b949e]Waiting for data...[/#8b949e]")

        lines.append("")

        # Drift status
        drift_text = "[bold #ff4500]⚠️ DRIFT DETECTED[/bold #ff4500]" if self._drift_detected else "[#39ff14]✓ Stable[/#39ff14]"
        lines.append(f"  [#8b949e]Drift:[/#8b949e] {drift_text}")
        lines.append(f"  [#8b949e]River Updates:[/#8b949e] [white]{self._river_updates:,}[/white]")

        return "\n".join(lines)


class SystemPanel(Static):
    """System resource metrics (CPU, RAM, network)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cpu: float = 0.0
        self._ram: float = 0.0
        self._blocked_count: int = 0
        self._packets_per_sec: float = 0.0

    def update_metrics(self, cpu: float, ram: float, blocked: int, pps: float) -> None:
        self._cpu = cpu
        self._ram = ram
        self._blocked_count = blocked
        self._packets_per_sec = pps
        self.refresh()

    def _bar(self, pct: float, width: int = 20) -> str:
        """Render a colored progress bar."""
        filled = int(pct / 100 * width)
        empty = width - filled
        if pct >= 90:
            color = "#ff0033"
        elif pct >= 70:
            color = "#ffa500"
        else:
            color = "#39ff14"
        bar = "█" * filled + "░" * empty
        return f"[{color}]{bar}[/{color}] [white]{pct:.0f}%[/white]"

    def render(self) -> str:
        lines = [
            "[bold #00ffff]⚙ SYSTEM METRICS[/bold #00ffff]\n",
            f"  [#8b949e]CPU Usage:   [/#8b949e]{self._bar(self._cpu)}",
            f"  [#8b949e]RAM Usage:   [/#8b949e]{self._bar(self._ram)}",
            f"  [#8b949e]Packets/s:   [/#8b949e][white]{self._packets_per_sec:.0f}[/white]",
            f"  [#8b949e]Blocked IPs: [/#8b949e][bold #ffa500]{self._blocked_count}[/bold #ffa500]",
        ]
        return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main Textual Application
# ─────────────────────────────────────────────────────────────────────────────


class SentinelDashboard(App):
    """
    SENTINEL-AI Real-Time Security Dashboard.

    Full-screen TUI with live packet visualization, threat detection
    status, AI model health, system metrics, and event log.
    """

    CSS = """
    Screen {
        background: #0d1117;
    }

    StatsBar {
        height: 1;
        background: #161b22;
        color: #e6edf3;
        padding: 0 2;
        border-bottom: solid #30363d;
    }

    #header-banner {
        height: 8;
        background: #0d1117;
        color: #00ffff;
        content-align: center middle;
    }

    #main-grid {
        layout: grid;
        grid-size: 3 1;
        grid-columns: 1fr 1fr 1fr;
        height: auto;
    }

    #top-panels {
        layout: horizontal;
        height: 22;
    }

    TrafficPanel {
        width: 1fr;
        height: 100%;
        border: solid #30363d;
        background: #161b22;
        padding: 0 1;
    }

    ThreatPanel {
        width: 1fr;
        height: 100%;
        border: solid #30363d;
        background: #161b22;
        padding: 0 1;
    }

    #right-panels {
        width: 24;
        layout: vertical;
        height: 100%;
    }

    AIStatusPanel {
        height: 14;
        border: solid #30363d;
        background: #161b22;
        padding: 1 2;
    }

    SystemPanel {
        height: 8;
        border: solid #30363d;
        background: #161b22;
        padding: 1 2;
    }

    #log-panel {
        height: 1fr;
        border: solid #30363d;
        background: #161b22;
        border-title-color: #00ffff;
    }

    Footer {
        background: #161b22;
        color: #8b949e;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit", priority=True),
        Binding("ctrl+c", "quit", "Quit", show=False, priority=True),
        Binding("p", "toggle_pause", "Pause"),
        Binding("d", "toggle_demo", "Demo Mode"),
        Binding("r", "reset_stats", "Reset Stats"),
        Binding("b", "show_blocks", "Blocked IPs"),
        Binding("h", "show_help", "Help"),
    ]

    TITLE = "SENTINEL-AI ─ Enterprise Cyber Defense System"

    def __init__(self, config: "SentinelConfig", event_bus: "EventBus") -> None:
        super().__init__()
        self._config = config
        self._event_bus = event_bus
        self._paused: bool = False
        self._stats = {
            "packets": 0,
            "flows": 0,
            "threats": 0,
            "blocked": 0,
        }
        self._pps_history: deque[tuple[float, int]] = deque(maxlen=10)

    def compose(self) -> ComposeResult:
        """Build the UI layout."""
        yield Header(show_clock=True)

        # Stats bar
        yield StatsBar(id="stats-bar")

        # Main layout
        with Horizontal(id="top-panels"):
            yield TrafficPanel(id="traffic-panel")
            yield ThreatPanel(id="threat-panel")
            with Vertical(id="right-panels"):
                yield AIStatusPanel(id="ai-panel")
                yield SystemPanel(id="system-panel")

        # Event log
        yield RichLog(
            id="log-panel",
            auto_scroll=True,
            highlight=True,
            markup=True,
            max_lines=self._config.dashboard.max_log_lines,
        )

        yield Footer()

    def on_mount(self) -> None:
        """Set up periodic refresh timers and subscribe to events."""
        # Register and apply custom theme safely
        try:
            self.register_theme(SENTINEL_THEME)
            self.theme = "sentinel"
        except Exception:
            pass  # Fall back to default theme if registration fails

        # System metrics refresh
        self.set_interval(2.0, self._refresh_system_metrics)

        # Register event bus subscriptions (from the async side)
        self._log("[#00ffff]SENTINEL-AI dashboard started[/#00ffff]")

        mode = "🛡️ ACTIVE DEFENSE" if self._config.live_defense else "👁️ OBSERVE MODE"
        self._log(f"[#8b949e]Defense mode:[/#8b949e] [bold]{mode}[/bold]")

        # Update mode in stats bar
        stats_bar = self.query_one("#stats-bar", StatsBar)
        stats_bar.mode = "ACTIVE" if self._config.live_defense else "OBSERVE"

    def update_packet_stats(self, packets: int, flows: int) -> None:
        """Update packet and flow counters from the capture pipeline."""
        self._stats["packets"] = packets
        self._stats["flows"] = flows

        stats_bar = self.query_one("#stats-bar", StatsBar)
        stats_bar.packets = packets
        stats_bar.flows = flows

    def on_threat_detected(
        self,
        attack_type: str,
        src_ip: str,
        dst_ip: str,
        confidence: float,
        threat_score: float,
        severity: str,
        flow_id: str = "",
    ) -> None:
        """Handle an incoming threat detection event."""
        self._stats["threats"] += 1

        # Update stats bar
        stats_bar = self.query_one("#stats-bar", StatsBar)
        stats_bar.threats = self._stats["threats"]

        # Add to threat panel
        threat_panel = self.query_one("#threat-panel", ThreatPanel)
        threat_panel.add_threat({
            "attack_type": attack_type,
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "confidence": confidence,
            "threat_score": threat_score,
            "severity": severity,
            "flow_id": flow_id,
        })

        # Log event
        emoji = ATTACK_EMOJIS.get(attack_type, "⚠️")
        sev_style = SEVERITY_STYLES.get(severity, "white")
        self._log(
            f"[{sev_style}]{emoji} THREAT[/{sev_style}] "
            f"[bold]{attack_type.upper()}[/bold] | "
            f"src=[#ff6b6b]{src_ip}[/#ff6b6b] | "
            f"conf=[white]{confidence:.1%}[/white] | "
            f"score=[bold]{threat_score:.1f}[/bold]"
        )

    def on_flow_classified(
        self,
        src_ip: str,
        dst_ip: str,
        proto: str,
        size: int,
        label: str,
        inference_ms: float = 0.0,
    ) -> None:
        """Handle a classified flow event (benign or threat)."""
        # Add to traffic panel
        traffic_panel = self.query_one("#traffic-panel", TrafficPanel)
        traffic_panel.add_flow(src_ip, dst_ip, proto, size, label)

        # Update AI panel
        ai_panel = self.query_one("#ai-panel", AIStatusPanel)
        ai_panel.update_prediction({
            "predicted_class": label,
            "confidence": 0.9,   # Simplified for display
            "inference_time_ms": inference_ms,
        })

    def on_ip_blocked(self, ip: str, reason: str, dry_run: bool) -> None:
        """Handle an IP block event."""
        self._stats["blocked"] += 1
        stats_bar = self.query_one("#stats-bar", StatsBar)
        stats_bar.blocked = self._stats["blocked"]

        mode_tag = "[#8b949e](simulate)[/#8b949e]" if dry_run else "[bold #ff4500](LIVE)[/bold #ff4500]"
        self._log(
            f"[bold #ffa500]🚫 IP BLOCKED[/bold #ffa500] {mode_tag} "
            f"[#ff6b6b]{ip}[/#ff6b6b] ─ reason: [bold]{reason}[/bold]"
        )

    def on_drift_detected(self) -> None:
        """Handle concept drift detection."""
        ai_panel = self.query_one("#ai-panel", AIStatusPanel)
        ai_panel.set_drift(True)
        self._log("[bold #ffa500]⚠️ CONCEPT DRIFT DETECTED — Feature distribution has shifted[/bold #ffa500]")

    def on_model_loaded(self, models_ready: dict) -> None:
        """Update AI panel when models are loaded."""
        ai_panel = self.query_one("#ai-panel", AIStatusPanel)
        ai_panel.update_models(models_ready)
        self._log("[#39ff14]✅ AI models loaded successfully[/#39ff14]")

    def _log(self, message: str) -> None:
        """Append a timestamped message to the event log."""
        ts = time.strftime("%H:%M:%S")
        log_panel = self.query_one("#log-panel", RichLog)
        log_panel.write(f"[#484f58]{ts}[/#484f58] {message}")

    async def _refresh_system_metrics(self) -> None:
        """Fetch and update system resource metrics."""
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().percent
        except ImportError:
            cpu, ram = 0.0, 0.0

        system_panel = self.query_one("#system-panel", SystemPanel)
        system_panel.update_metrics(
            cpu=cpu,
            ram=ram,
            blocked=self._stats["blocked"],
            pps=0.0,  # Updated by capture pipeline
        )

    # ── Key bindings ──────────────────────────────────────────────────────────

    def action_toggle_pause(self) -> None:
        """Toggle packet capture pause state."""
        self._paused = not self._paused
        status = "PAUSED ⏸" if self._paused else "RUNNING ▶"
        self._log(f"[#ffa500]Capture {status}[/#ffa500]")

    def action_toggle_demo(self) -> None:
        """Toggle demo mode (simulated traffic)."""
        self._config.demo_mode = not self._config.demo_mode
        status = "ON" if self._config.demo_mode else "OFF"
        self._log(f"[#00ffff]Demo mode: {status}[/#00ffff]")

    def action_reset_stats(self) -> None:
        """Reset all displayed statistics."""
        self._stats = {"packets": 0, "flows": 0, "threats": 0, "blocked": 0}
        stats_bar = self.query_one("#stats-bar", StatsBar)
        stats_bar.packets = 0
        stats_bar.flows = 0
        stats_bar.threats = 0
        stats_bar.blocked = 0
        self._log("[#8b949e]Statistics reset[/#8b949e]")

    def action_show_blocks(self) -> None:
        """Show blocked IPs summary in the log."""
        self._log(f"[#ffa500]Blocked IPs: {self._stats['blocked']} total[/#ffa500]")

    def action_show_help(self) -> None:
        """Show keyboard shortcut help."""
        self._log(
            "[#00ffff]Shortcuts:[/#00ffff] "
            "[q] Quit  [p] Pause  [d] Demo  [r] Reset  [b] Blocks  [h] Help"
        )
