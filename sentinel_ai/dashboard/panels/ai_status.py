"""
SENTINEL-AI — AI Status Panel
================================
Model readiness, last prediction details, drift status,
River online learning progress, and SHAP top features.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from rich import box
from rich.text import Text
from textual.widgets import Static


@dataclass
class AIStatusData:
    """Snapshot of AI engine state for display."""
    cnn_ready: bool = False
    xgb_ready: bool = False
    iso_ready: bool = False
    river_ready: bool = False
    last_class: str = "—"
    last_confidence: float = 0.0
    last_inference_ms: float = 0.0
    avg_inference_ms: float = 0.0
    total_predictions: int = 0
    drift_detected: bool = False
    drift_count: int = 0
    river_updates: int = 0
    top_features: list[tuple[str, float]] = field(default_factory=list)
    last_explanation: str = ""


class AIStatusPanel(Static):
    """
    AI engine status panel with model readiness indicators,
    prediction metrics, drift alerts, and SHAP top features.
    """

    DEFAULT_CSS = """
    AIStatusPanel {
        height: 1fr;
        border: solid #30363d;
        background: #0d1117;
        padding: 0 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._data = AIStatusData()

    def on_mount(self) -> None:
        self.set_interval(1.0, self.refresh)

    def update(self, data: AIStatusData) -> None:
        """Update displayed AI status data."""
        self._data = data

    def _model_indicator(self, ready: bool, name: str) -> str:
        icon = "●" if ready else "○"
        color = "#39ff14" if ready else "#ff0033"
        return f"[{color}]{icon}[/{color}] {name}"

    def _shap_bar(self, value: float) -> str:
        """Simple ASCII bar for SHAP feature importance."""
        filled = min(int(abs(value) * 20), 20)
        sign_color = "#ff4500" if value > 0 else "#00ffff"
        sign = "▲" if value > 0 else "▼"
        return f"[{sign_color}]{sign}{'█' * filled}[/{sign_color}]"

    def render(self) -> Table:
        d = self._data
        table = Table(
            title="[bold #8b00ff]🤖 AI ENGINE STATUS[/bold #8b00ff]",
            box=box.SIMPLE_HEAD,
            border_style="#30363d",
            header_style="bold #8b949e",
            show_edge=False,
            expand=True,
        )
        table.add_column("Component", width=28)
        table.add_column("Value", style="#e6edf3")

        # Model readiness
        table.add_row(
            "[bold #8b949e]── MODELS ──────────────[/bold #8b949e]", ""
        )
        table.add_row("CNN+BiLSTM+Attention", self._model_indicator(d.cnn_ready, "PRIMARY"))
        table.add_row("XGBoost",             self._model_indicator(d.xgb_ready, "SECONDARY"))
        table.add_row("Isolation Forest",    self._model_indicator(d.iso_ready, "ANOMALY"))
        table.add_row("River (online)",      self._model_indicator(d.river_ready, "ONLINE"))

        # Last prediction
        table.add_row("", "")
        table.add_row(
            "[bold #8b949e]── LAST PREDICTION ─────[/bold #8b949e]", ""
        )
        class_color = "#39ff14" if d.last_class == "benign" else "#ff4500"
        table.add_row("Classification",  f"[bold {class_color}]{d.last_class.upper()}[/bold {class_color}]")
        conf_color = "#39ff14" if d.last_confidence >= 0.8 else "#ffa500"
        table.add_row("Confidence",      f"[{conf_color}]{d.last_confidence:.2%}[/{conf_color}]")
        table.add_row("Inference Time",  f"{d.last_inference_ms:.2f} ms")
        table.add_row("Avg Latency",     f"{d.avg_inference_ms:.2f} ms")
        table.add_row("Total Preds",     f"{d.total_predictions:,}")

        # Drift
        table.add_row("", "")
        table.add_row(
            "[bold #8b949e]── DRIFT & LEARNING ────[/bold #8b949e]", ""
        )
        drift_indicator = (
            "[bold #ff0033]⚠ DRIFT DETECTED[/bold #ff0033]" if d.drift_detected
            else "[#39ff14]● Stable[/#39ff14]"
        )
        table.add_row("Concept Drift",  drift_indicator)
        table.add_row("Drift Events",   str(d.drift_count))
        table.add_row("River Updates",  f"{d.river_updates:,}")

        # SHAP top features (if available)
        if d.top_features:
            table.add_row("", "")
            table.add_row(
                "[bold #8b949e]── TOP FEATURES (SHAP) ─[/bold #8b949e]", ""
            )
            for name, val in d.top_features[:5]:
                short = name[:20]
                table.add_row(f"  {short}", f"{self._shap_bar(val)} {val:+.3f}")

        return table
