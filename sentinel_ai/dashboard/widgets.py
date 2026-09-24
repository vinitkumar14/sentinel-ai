"""
SENTINEL-AI Dashboard — Custom Textual Widgets
=================================================
Reusable Textual widgets beyond the standard library:
  - GaugeWidget: animated percentage gauge
  - SparklineWidget: ASCII sparkline history chart
  - AlertBannerWidget: auto-dismissing alert popup
  - ThreatMeterWidget: 0–10 animated threat level indicator
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

from textual.reactive import reactive
from textual.widgets import Static


# ── Gauge Widget ──────────────────────────────────────────────────────────────

class GaugeWidget(Static):
    """
    Animated percentage gauge (0–100%).
    Color transitions: green → orange → red at thresholds.

    Usage:
        gauge = GaugeWidget(label="CPU", warn_at=70, crit_at=90)
        gauge.value = 45.3
    """

    value: reactive[float] = reactive(0.0)

    DEFAULT_CSS = """
    GaugeWidget {
        height: 1;
        padding: 0 1;
    }
    """

    def __init__(
        self,
        label: str = "METRIC",
        warn_at: float = 70.0,
        crit_at: float = 90.0,
        width: int = 20,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._warn_at = warn_at
        self._crit_at = crit_at
        self._width = width

    def _color(self) -> str:
        if self.value >= self._crit_at:
            return "#ff0033"
        if self.value >= self._warn_at:
            return "#ffa500"
        return "#39ff14"

    def render(self) -> str:
        color = self._color()
        filled = int((self.value / 100.0) * self._width)
        empty = self._width - filled
        bar = f"[{color}]{'█' * filled}{'░' * empty}[/{color}]"
        val_str = f"[{color}]{self.value:5.1f}%[/{color}]"
        return f"[bold #8b949e]{self._label:8}[/bold #8b949e] {bar} {val_str}"


# ── Sparkline Widget ──────────────────────────────────────────────────────────

class SparklineWidget(Static):
    """
    Horizontally scrolling ASCII sparkline history chart.

    Usage:
        spark = SparklineWidget(label="PPS", max_points=40)
        spark.push(pps_value)
    """

    DEFAULT_CSS = """
    SparklineWidget {
        height: 2;
        padding: 0 1;
    }
    """

    _BLOCKS = " ▁▂▃▄▅▆▇█"

    def __init__(
        self,
        label: str = "VALUE",
        max_points: int = 40,
        color: str = "#00ffff",
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._label = label
        self._history: deque[float] = deque(maxlen=max_points)
        self._color = color

    def push(self, value: float) -> None:
        """Add a new data point and refresh."""
        self._history.append(value)
        self.refresh()

    def render(self) -> str:
        if not self._history:
            return f"[#8b949e]{self._label}: no data[/#8b949e]"

        max_val = max(self._history) or 1.0
        chars = [
            self._BLOCKS[int((v / max_val) * 8)]
            for v in self._history
        ]
        sparkline = "".join(chars)
        current = list(self._history)[-1]
        return (
            f"[bold #8b949e]{self._label:8}[/bold #8b949e] "
            f"[{self._color}]{sparkline}[/{self._color}]  "
            f"[{self._color}]{current:8.1f}[/{self._color}]"
        )


# ── Alert Banner Widget ───────────────────────────────────────────────────────

class AlertBannerWidget(Static):
    """
    Flashing auto-dismissing alert banner.
    Shows critical threat alerts prominently.

    Usage:
        banner = AlertBannerWidget()
        banner.show_alert("DDoS detected from 1.2.3.4", severity="CRITICAL")
    """

    DEFAULT_CSS = """
    AlertBannerWidget {
        height: 1;
        padding: 0 2;
        background: #161b22;
    }
    """

    def __init__(self, dismiss_after: float = 10.0, **kwargs) -> None:
        super().__init__(**kwargs)
        self._message: str = ""
        self._severity: str = "INFO"
        self._shown_at: float = 0.0
        self._dismiss_after = dismiss_after
        self._blink_state: bool = True

    def on_mount(self) -> None:
        self.set_interval(0.5, self._blink)

    def _blink(self) -> None:
        if self._message:
            self._blink_state = not self._blink_state
            self.refresh()
            if time.time() - self._shown_at > self._dismiss_after:
                self._message = ""
                self.refresh()

    def show_alert(self, message: str, severity: str = "HIGH") -> None:
        """Display an alert message."""
        self._message = message
        self._severity = severity
        self._shown_at = time.time()
        self._blink_state = True
        self.refresh()

    def render(self) -> str:
        if not self._message:
            return "[#8b949e]No active alerts[/#8b949e]"

        color_map = {
            "CRITICAL": "#ff0033",
            "HIGH":     "#ff4500",
            "MEDIUM":   "#ffa500",
            "LOW":      "#ffff00",
        }
        color = color_map.get(self._severity, "#8b949e")
        icon = "⚡" if self._blink_state else "⚠ "
        return f"[bold {color}]{icon} {self._severity}: {self._message}[/bold {color}]"


# ── Threat Meter Widget ───────────────────────────────────────────────────────

class ThreatMeterWidget(Static):
    """
    Visual 0–10 threat level meter with animated indicator.
    Updates in real-time as new threats are detected.

    Usage:
        meter = ThreatMeterWidget()
        meter.score = 8.5
    """

    score: reactive[float] = reactive(0.0)

    DEFAULT_CSS = """
    ThreatMeterWidget {
        height: 3;
        padding: 0 1;
    }
    """

    def _threat_color(self) -> str:
        if self.score >= 9.0:
            return "#ff0033"
        if self.score >= 7.0:
            return "#ff4500"
        if self.score >= 5.0:
            return "#ffa500"
        if self.score >= 3.0:
            return "#ffff00"
        return "#39ff14"

    def _severity_label(self) -> str:
        if self.score >= 9.0:
            return "CRITICAL"
        if self.score >= 7.0:
            return "HIGH"
        if self.score >= 5.0:
            return "MEDIUM"
        if self.score >= 3.0:
            return "LOW"
        return "INFO"

    def render(self) -> str:
        color = self._threat_color()
        label = self._severity_label()
        filled = int((self.score / 10.0) * 30)
        empty = 30 - filled
        bar = f"[{color}]{'█' * filled}{'░' * empty}[/{color}]"
        return (
            f"[bold #8b949e]THREAT LEVEL[/bold #8b949e]\n"
            f"{bar}\n"
            f"[bold {color}]{self.score:4.1f}/10  {label}[/bold {color}]"
        )
