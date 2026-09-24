#!/usr/bin/env python3
"""
SENTINEL-AI: Enterprise AI Cyber Defense System
=================================================
Main CLI entry point.

Usage:
    # Start with dashboard (default — observe-only mode)
    python main.py

    # Enable active IP blocking
    python main.py --live-defense

    # Demo mode (no root required, simulates traffic)
    python main.py --demo

    # Headless mode (no TUI dashboard)
    python main.py --headless

    # Replay a PCAP file
    python main.py --replay path/to/capture.pcap

    # Custom config
    python main.py --config config/custom.yaml

    # Specific network interface
    python main.py --interface eth0

Options:
    --live-defense    Enable active IP blocking (requires root/admin)
    --demo            Run in demo mode (simulated traffic, no root needed)
    --headless        No TUI dashboard; log to file only
    --interface IF    Network interface to capture on (default: auto)
    --replay FILE     Replay a PCAP file instead of live capture
    --config FILE     Path to custom YAML config file
    --debug           Enable debug logging
    --version         Show version and exit
"""

from __future__ import annotations

import asyncio
import os
import platform
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Ensure the sentinel-ai directory is on the Python path
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from sentinel_ai.core.config import load_config
from sentinel_ai.core.constants import APP_DESCRIPTION, APP_NAME, VERSION
from sentinel_ai.core.logger import get_logger, setup_logging

# Set event loop policy for Windows
if platform.system() == "Windows":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
else:
    try:
        import uvloop
        asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    except ImportError:
        pass  # Fall back to default asyncio

log = get_logger(__name__)
app = typer.Typer(
    name="sentinel-ai",
    help=APP_DESCRIPTION,
    no_args_is_help=False,
    add_completion=False,
)

console = Console()

STARTUP_BANNER = r"""
[bold #00ffff]
  ███████╗███████╗███╗   ██╗████████╗██╗███╗   ██╗███████╗██╗      █████╗ ██╗
  ██╔════╝██╔════╝████╗  ██║╚══██╔══╝██║████╗  ██║██╔════╝██║     ██╔══██╗██║
  ███████╗█████╗  ██╔██╗ ██║   ██║   ██║██╔██╗ ██║█████╗  ██║     ███████║██║
  ╚════██║██╔══╝  ██║╚██╗██║   ██║   ██║██║╚██╗██║██╔══╝  ██║     ██╔══██║╚═╝
  ███████║███████╗██║ ╚████║   ██║   ██║██║ ╚████║███████╗███████╗██║  ██║██╗
  ╚══════╝╚══════╝╚═╝  ╚═══╝   ╚═╝   ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝
[/bold #00ffff]
[#8b949e]               Enterprise AI Self-Learning Cyber Defense System[/#8b949e]
[bold #39ff14]                          Version {version}[/bold #39ff14]
"""


def print_banner() -> None:
    """Print the SENTINEL-AI ASCII banner."""
    console.print(STARTUP_BANNER.format(version=VERSION))


def check_privileges(live_defense: bool) -> None:
    """Warn if running without root when live defense is enabled."""
    if not live_defense:
        return

    is_root = (os.geteuid() == 0) if hasattr(os, "geteuid") else (
        os.getenv("USERNAME") == "Administrator"
    )

    if not is_root:
        console.print(
            "\n[bold #ff0033]⚠️  WARNING: --live-defense requires root/Administrator privileges![/bold #ff0033]\n"
            "[#8b949e]  IP blocking commands (iptables/netsh) will fail without elevated permissions.[/#8b949e]\n"
            "[#8b949e]  Run with: sudo python main.py --live-defense[/#8b949e]\n"
        )


@app.command()
def main(
    live_defense: bool = typer.Option(
        False, "--live-defense",
        help="Enable active IP blocking (requires root/Administrator)",
    ),
    demo: bool = typer.Option(
        False, "--demo",
        help="Run in demo mode with simulated traffic (no root required)",
    ),
    headless: bool = typer.Option(
        False, "--headless",
        help="Run without TUI dashboard (log to file only)",
    ),
    interface: str = typer.Option(
        "auto", "--interface", "-i",
        help="Network interface to capture on (default: auto-detect)",
    ),
    replay: Optional[str] = typer.Option(
        None, "--replay",
        help="Replay a PCAP file instead of live capture",
    ),
    config_file: Optional[str] = typer.Option(
        None, "--config", "-c",
        help="Path to custom YAML configuration file",
    ),
    debug: bool = typer.Option(
        False, "--debug",
        help="Enable debug logging",
    ),
    version: bool = typer.Option(
        False, "--version",
        help="Show version and exit",
    ),
) -> None:
    """
    Start the SENTINEL-AI Cyber Defense System.
    """
    if version:
        console.print(f"[bold]{APP_NAME}[/bold] v{VERSION}")
        raise typer.Exit()

    print_banner()
    check_privileges(live_defense)

    # ── Load configuration ────────────────────────────────────────────────────
    config = load_config(
        config_path=config_file,
        project_root=str(_HERE),
    )

    # Apply CLI overrides
    if live_defense:
        config.enable_live_defense()
        console.print(
            "[bold #ff4500]🛡️  ACTIVE DEFENSE MODE ENABLED — IP blocking is LIVE[/bold #ff4500]\n"
        )
    else:
        console.print(
            "[bold #00ffff]👁️  OBSERVE MODE — Threats will be logged but NOT blocked[/bold #00ffff]\n"
        )

    if demo:
        config.demo_mode = True
    if debug:
        config.system.debug = True
    if interface != "auto":
        config.capture.interface = interface

    # ── Setup logging ─────────────────────────────────────────────────────────
    setup_logging(config)

    log.info(
        "SENTINEL-AI starting",
        version=VERSION,
        mode="ACTIVE" if live_defense else "OBSERVE",
        demo=demo,
        interface=config.capture.interface,
    )

    # ── Run the system ────────────────────────────────────────────────────────
    try:
        if headless:
            _run_headless(config, replay)
        else:
            _run_with_dashboard(config, replay)
    except KeyboardInterrupt:
        console.print("\n[#8b949e]Shutdown requested by user[/#8b949e]")
    except PermissionError as exc:
        console.print(
            f"\n[bold #ff0033]PERMISSION ERROR:[/bold #ff0033] {exc}\n"
            "[#8b949e]Try running with sudo or as Administrator[/#8b949e]"
        )
        raise typer.Exit(code=1)
    except Exception as exc:
        log.error("Fatal error", error=str(exc))
        console.print(f"\n[bold #ff0033]FATAL ERROR:[/bold #ff0033] {exc}")
        raise typer.Exit(code=1)


def _run_with_dashboard(config: object, replay: Optional[str]) -> None:
    """Run SENTINEL-AI with the full Textual TUI dashboard."""
    from sentinel_ai.dashboard.app import SentinelDashboard
    from sentinel_ai.sentinel import SentinelOrchestrator

    dashboard = SentinelDashboard(config, None)  # type: ignore[arg-type]

    orchestrator = SentinelOrchestrator(config, dashboard)  # type: ignore[arg-type]

    async def _async_run() -> None:
        await orchestrator.initialize()

        # Run orchestrator in background
        asyncio.create_task(orchestrator.run())

        # Handle PCAP replay mode
        if replay:
            from sentinel_ai.capture.pcap_replayer import PcapReplayer
            replayer = PcapReplayer(config, None)  # type: ignore[arg-type]
            asyncio.create_task(
                replayer.replay(replay, speed=10.0, loop=True)
            )

        # Run Textual app (blocks until user quits)
        await dashboard.run_async()

        # Cleanup
        await orchestrator.stop()

    asyncio.run(_async_run())


def _run_headless(config: object, replay: Optional[str]) -> None:
    """Run SENTINEL-AI without a dashboard (log to file only)."""
    from sentinel_ai.sentinel import SentinelOrchestrator

    orchestrator = SentinelOrchestrator(config)  # type: ignore[arg-type]

    async def _async_run() -> None:
        await orchestrator.initialize()
        log.info("Running in headless mode — Press Ctrl+C to stop")
        await orchestrator.run()

        # Block until Ctrl+C
        stop_event = asyncio.Event()
        try:
            await stop_event.wait()
        except asyncio.CancelledError:
            pass
        finally:
            await orchestrator.stop()

    try:
        asyncio.run(_async_run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    app()
