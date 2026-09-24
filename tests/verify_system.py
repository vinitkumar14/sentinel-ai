"""
SENTINEL-AI System Verification Script
=========================================
Checks that all dependencies, artifacts, and modules are in order
before launching the system.

Run:
    python scripts/verify_system.py

Returns exit code 0 if ready, 1 if something is missing.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    console = Console()
except ImportError:
    print("ERROR: 'rich' is not installed. Run: pip install rich")
    sys.exit(1)


def check(label: str, ok: bool, detail: str = "") -> bool:
    status = "[#39ff14]✓ OK[/#39ff14]" if ok else "[#ff0033]✗ MISSING[/#ff0033]"
    console.print(f"  {status}  {label}" + (f"  [#8b949e]{detail}[/#8b949e]" if detail else ""))
    return ok


def main() -> int:
    console.print("\n[bold #00ffff]SENTINEL-AI System Verification[/bold #00ffff]\n")

    all_ok = True

    # ── Python version ────────────────────────────────────────────────────────
    console.print("[bold]Python Environment[/bold]")
    py_ok = sys.version_info >= (3, 11)
    all_ok &= check(
        f"Python >= 3.11",
        py_ok,
        f"(found {sys.version.split()[0]})",
    )

    # ── Critical imports ──────────────────────────────────────────────────────
    console.print("\n[bold]Core Dependencies[/bold]")
    core_deps = [
        ("tensorflow", "2.15+"),
        ("sklearn", "scikit-learn"),
        ("xgboost", "2.0+"),
        ("numpy", "1.26+"),
        ("pandas", "2.1+"),
        ("rich", "13.7+"),
        ("textual", "0.50+"),
        ("loguru", "logging"),
        ("structlog", "structured logging"),
        ("pydantic", "v2"),
        ("aiosqlite", "async SQLite"),
        ("yaml", "PyYAML"),
    ]

    for mod_name, desc in core_deps:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "?")
            all_ok &= check(f"{desc}", True, f"v{ver}")
        except ImportError:
            all_ok &= check(f"{desc}", False, f"(pip install {mod_name})")

    # ── Optional dependencies ─────────────────────────────────────────────────
    console.print("\n[bold]Optional Dependencies[/bold]")
    optional_deps = [
        ("scapy", "Packet capture (required for live mode)"),
        ("river", "Online learning"),
        ("shap", "Model explanations"),
        ("psutil", "System metrics"),
        ("redis", "Redis pub/sub caching"),
    ]

    for mod_name, desc in optional_deps:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", "?")
            check(f"{desc}", True, f"v{ver}")
        except ImportError:
            check(f"{desc}", False, "(optional)")

    # ── Sentinel-AI modules ───────────────────────────────────────────────────
    console.print("\n[bold]SENTINEL-AI Modules[/bold]")
    sentinel_modules = [
        ("sentinel_ai.core.config", "Config system"),
        ("sentinel_ai.core.events", "Event bus"),
        ("sentinel_ai.core.logger", "Logger"),
        ("sentinel_ai.core.exceptions", "Exceptions"),
        ("sentinel_ai.core.constants", "Constants"),
        ("sentinel_ai.capture.packet_sniffer", "Packet sniffer"),
        ("sentinel_ai.capture.flow_manager", "Flow manager"),
        ("sentinel_ai.capture.feature_extractor", "Feature extractor"),
        ("sentinel_ai.ai.model_manager", "Model manager"),
        ("sentinel_ai.ai.ensemble_predictor", "Ensemble predictor"),
        ("sentinel_ai.ai.threat_scorer", "Threat scorer"),
        ("sentinel_ai.ai.drift_detector", "Drift detector"),
        ("sentinel_ai.defense.ip_blocker", "IP blocker"),
        ("sentinel_ai.defense.response_engine", "Response engine"),
        ("sentinel_ai.storage.database", "Database"),
        ("sentinel_ai.dashboard.app", "Dashboard"),
    ]

    for mod_path, desc in sentinel_modules:
        try:
            importlib.import_module(mod_path)
            all_ok &= check(desc, True)
        except ImportError as e:
            all_ok &= check(desc, False, str(e)[:60])

    # ── Artifacts ─────────────────────────────────────────────────────────────
    console.print("\n[bold]Preprocessing Artifacts[/bold]")
    artifacts = [
        ("scaler.pkl", "RobustScaler"),
        ("label_encoder.pkl", "LabelEncoder"),
        ("selector.pkl", "VarianceThreshold"),
    ]

    for fname, desc in artifacts:
        path = _PROJECT_ROOT / "artifacts" / fname
        all_ok &= check(f"{desc} ({fname})", path.exists())

    # ── Model files ───────────────────────────────────────────────────────────
    console.print("\n[bold]AI Model Files[/bold]")
    models = [
        ("models/cnn_bilstm_attention/model.h5", "CNN+BiLSTM+Attention (REQUIRED)"),
        ("models/xgboost/model.pkl", "XGBoost (optional)"),
        ("models/isolation_forest/model.pkl", "IsolationForest (optional)"),
        ("models/river/online_model.pkl", "River online model (optional)"),
    ]

    for rel_path, desc in models:
        path = _PROJECT_ROOT / rel_path
        required = "REQUIRED" in desc
        exists = path.exists()
        if required:
            all_ok &= check(desc, exists)
        else:
            check(desc, exists)  # Don't fail for optional models

    # ── Config files ──────────────────────────────────────────────────────────
    console.print("\n[bold]Configuration Files[/bold]")
    configs = [
        "config/default.yaml",
        "config/models.yaml",
        "config/defense.yaml",
        "config/logging.yaml",
    ]
    for cfg in configs:
        path = _PROJECT_ROOT / cfg
        all_ok &= check(cfg, path.exists())

    # ── Final result ──────────────────────────────────────────────────────────
    console.print()
    if all_ok:
        console.print("[bold #39ff14]✅ SYSTEM READY — All checks passed![/bold #39ff14]")
        console.print("[#8b949e]Start with: python main.py --demo[/#8b949e]\n")
        return 0
    else:
        console.print(
            "[bold #ff4500]⚠️  SYSTEM NOT READY — Fix errors above before starting.[/bold #ff4500]"
        )
        console.print(
            "[#8b949e]Run: python scripts/import_artifacts.py[/#8b949e]\n"
            "[#8b949e]Then: python scripts/export_model.py[/#8b949e]\n"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
