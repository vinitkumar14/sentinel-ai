"""
SENTINEL-AI Dashboard Theme
==============================
Custom Rich/Textual color theme with cybersecurity aesthetics.
"""

from textual.theme import Theme

# ─────────────────────────────────────────────────────────────────────────────
# Color palette
# ─────────────────────────────────────────────────────────────────────────────

COLORS = {
    # Backgrounds
    "bg_dark":     "#0d1117",
    "bg_panel":    "#161b22",
    "bg_header":   "#21262d",
    "bg_accent":   "#1f2937",

    # Primary accent
    "cyan":        "#00ffff",
    "cyan_dim":    "#00b4b4",
    "green":       "#39ff14",    # Neon green
    "green_dim":   "#22c55e",

    # Threat levels
    "critical":    "#ff0033",
    "high":        "#ff4500",
    "medium":      "#ffa500",
    "low":         "#ffff00",
    "info":        "#00d4ff",
    "benign":      "#39ff14",

    # Text
    "text_primary":   "#e6edf3",
    "text_secondary": "#8b949e",
    "text_dim":       "#484f58",

    # Borders
    "border":      "#30363d",
    "border_active": "#00ffff",
}

# Rich markup aliases for common elements
SEVERITY_STYLES = {
    "CRITICAL": "bold #ff0033",
    "HIGH":     "bold #ff4500",
    "MEDIUM":   "bold #ffa500",
    "LOW":      "#ffff00",
    "INFO":     "#00d4ff",
    "BENIGN":   "#39ff14",
}

ATTACK_EMOJIS = {
    "benign":     "✅",
    "bot":        "🤖",
    "bruteforce": "🔨",
    "ddos":       "💥",
    "dos":        "🔴",
    "heartbleed": "❤️",
    "portscan":   "🔍",
    "webattack":  "🌐",
    "anomaly":    "⚠️",
}

SENTINEL_THEME = Theme(
    name="sentinel",
    primary="#00ffff",
    secondary="#39ff14",
    accent="#ff4500",
    foreground="#e6edf3",
    background="#0d1117",
    surface="#161b22",
    panel="#161b22",
    boost="#21262d",
    error="#ff0033",
    warning="#ffa500",
    success="#39ff14",
    dark=True,
    variables={
        "cyan":           "#00ffff",
        "cyan-dim":       "#00b4b4",
        "green":          "#39ff14",
        "green-dim":      "#22c55e",
        "critical":       "#ff0033",
        "high":           "#ff4500",
        "medium":         "#ffa500",
        "low":            "#ffff00",
        "border":         "#30363d",
        "border-active":  "#00ffff",
        "text-primary":   "#e6edf3",
        "text-secondary": "#8b949e",
        "text-dim":       "#484f58",
    },
)
