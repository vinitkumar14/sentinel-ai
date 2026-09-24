"""
SENTINEL-AI Main Orchestrator
================================
The central async pipeline that wires all modules together:

  PacketSniffer → FlowManager → FeatureExtractor → EnsemblePredictor
               → ThreatScorer → ResponseEngine → SentinelDashboard

This module is the runtime backbone — it starts all async tasks,
manages their lifecycle, and handles graceful shutdown.

Usage:
    from sentinel_ai.sentinel import SentinelOrchestrator

    orchestrator = SentinelOrchestrator(config)
    await orchestrator.run()
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Optional

import numpy as np

from sentinel_ai.ai.drift_detector import DriftDetector
from sentinel_ai.ai.ensemble_predictor import EnsemblePredictor
from sentinel_ai.ai.model_manager import ModelManager
from sentinel_ai.ai.river_learner import RiverLearner
from sentinel_ai.ai.threat_scorer import ThreatScorer
from sentinel_ai.capture.feature_extractor import FeatureExtractor
from sentinel_ai.capture.flow_manager import FlowManager
from sentinel_ai.capture.packet_sniffer import PacketSniffer
from sentinel_ai.capture.pcap_writer import PcapWriter
from sentinel_ai.core.constants import PROTOCOL_MAP
from sentinel_ai.core.events import EventBus, EventType, SentinelEvent
from sentinel_ai.core.logger import get_logger
from sentinel_ai.defense.alert_manager import AlertManager
from sentinel_ai.defense.ip_blocker import IPBlocker
from sentinel_ai.defense.response_engine import ResponseEngine
from sentinel_ai.storage.database import Database

if TYPE_CHECKING:
    from sentinel_ai.capture.flow_manager import NetworkFlow
    from sentinel_ai.core.config import SentinelConfig
    from sentinel_ai.dashboard.app import SentinelDashboard

log = get_logger(__name__)


class SentinelOrchestrator:
    """
    Central coordinator for all SENTINEL-AI subsystems.

    Manages:
      - Component initialization and health
      - The async inference pipeline (sniffer → flow → features → AI → defense)
      - Demo mode traffic simulation
      - Periodic stats publishing
      - Graceful shutdown

    Args:
        config:    Loaded SentinelConfig.
        dashboard: SentinelDashboard instance (optional; None = headless mode).
    """

    def __init__(
        self,
        config: "SentinelConfig",
        dashboard: Optional["SentinelDashboard"] = None,
    ) -> None:
        self._config = config
        self._dashboard = dashboard

        # ── Event bus ─────────────────────────────────────────────────────────
        self._event_bus = EventBus(queue_size=config.capture.queue_size)

        # ── Capture layer ─────────────────────────────────────────────────────
        self._sniffer = PacketSniffer(config, self._event_bus)
        self._flow_manager = FlowManager(config, self._event_bus)
        self._feature_extractor = FeatureExtractor(config)
        self._pcap_writer = PcapWriter(config)

        # ── AI layer ──────────────────────────────────────────────────────────
        self._model_manager = ModelManager(config)
        self._predictor = EnsemblePredictor(self._model_manager, config)
        self._threat_scorer = ThreatScorer(config)
        self._drift_detector = DriftDetector(config, self._event_bus)
        self._river_learner: Optional[RiverLearner] = None  # Set after model load

        # ── Defense layer ─────────────────────────────────────────────────────
        self._ip_blocker = IPBlocker(config, self._event_bus)
        self._alert_manager = AlertManager(config, self._event_bus)

        # ── Storage ───────────────────────────────────────────────────────────
        self._db = Database(config)

        # ── State ─────────────────────────────────────────────────────────────
        self._running: bool = False
        self._stats = {
            "packets": 0,
            "flows": 0,
            "threats": 0,
            "benign": 0,
        }
        self._inference_task: Optional[asyncio.Task] = None
        self._stats_task: Optional[asyncio.Task] = None
        self._demo_task: Optional[asyncio.Task] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Initialize all components. Call before run()."""
        log.info("Initializing SENTINEL-AI...")

        # Database
        await self._db.init()

        # Load AI models
        log.info("Loading AI models...")
        self._model_manager.load_all()

        # Load preprocessing artifacts
        try:
            self._feature_extractor.load_artifacts()
        except Exception as exc:
            log.warning(
                "Could not load feature extraction artifacts",
                error=str(exc),
                hint="Run 'python scripts/import_artifacts.py' first.",
            )

        # Initialize River learner after models are loaded
        self._river_learner = RiverLearner(
            self._model_manager, self._config, self._event_bus
        )

        # Initialize response engine
        self._response_engine = ResponseEngine(
            self._config,
            self._ip_blocker,
            self._alert_manager,
            self._river_learner,
            self._event_bus,
        )

        # Register event bus handlers
        self._register_event_handlers()

        # Start subsystems
        await self._event_bus.start()
        await self._flow_manager.start()
        await self._ip_blocker.start()
        await self._pcap_writer.start()

        # Notify dashboard
        if self._dashboard:
            self._dashboard.on_model_loaded(self._model_manager.models_ready)

        log.info(
            "SENTINEL-AI initialized",
            models_ready=self._model_manager.models_ready,
            live_defense=self._config.live_defense,
            demo_mode=self._config.demo_mode,
        )

    async def run(self) -> None:
        """Start all async tasks and run until shutdown."""
        self._running = True

        # Periodic stats task
        self._stats_task = asyncio.create_task(
            self._stats_loop(), name="stats-loop"
        )

        if self._config.demo_mode:
            # Demo mode: simulate traffic without a live network
            self._demo_task = asyncio.create_task(
                self._demo_traffic_loop(), name="demo-traffic"
            )
            log.info("Demo mode active — simulating traffic")
        else:
            # Live mode: real packet capture
            self._inference_task = asyncio.create_task(
                self._live_capture_loop(), name="capture-pipeline"
            )

        # Publish system started event
        await self._event_bus.publish(
            SentinelEvent(
                type=EventType.SYSTEM_STARTED,
                payload={"timestamp": time.time()},
                source="orchestrator",
            )
        )

    async def stop(self) -> None:
        """Graceful shutdown of all subsystems."""
        log.info("Initiating graceful shutdown...")
        self._running = False

        # Cancel background tasks
        for task in [self._inference_task, self._stats_task, self._demo_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Stop subsystems
        await self._sniffer.stop()
        await self._flow_manager.stop()
        await self._ip_blocker.stop()
        await self._pcap_writer.stop()
        await self._event_bus.stop()

        # Save River checkpoint
        if self._river_learner:
            await self._river_learner.maybe_checkpoint()

        # Close DB
        await self._db.close()

        log.info(
            "SENTINEL-AI stopped",
            packets=self._stats["packets"],
            flows=self._stats["flows"],
            threats=self._stats["threats"],
        )

    # ── Live capture pipeline ─────────────────────────────────────────────────

    async def _live_capture_loop(self) -> None:
        """
        Main async pipeline: sniffer → flow manager → feature extractor → AI → defense.
        """
        try:
            async with self._sniffer:
                async for packet in self._sniffer.packet_generator():
                    if not self._running:
                        break

                    self._stats["packets"] += 1

                    # Feed packet to flow manager
                    completed_flows = await self._flow_manager.process_packet(packet)

                    # Process any completed flows
                    for flow in completed_flows:
                        await self._process_flow(flow)

        except asyncio.CancelledError:
            pass
        except Exception as exc:
            log.error("Live capture loop error", error=str(exc))

    async def _process_flow(self, flow: "NetworkFlow") -> None:
        """Extract features and run AI inference on a completed flow."""
        self._stats["flows"] += 1

        try:
            # Feature extraction
            if not self._feature_extractor._artifacts_loaded:
                return

            features = self._feature_extractor.extract(flow)
            if features is None:
                return

            # AI inference
            result = await self._predictor.predict(
                features,
                flow_id=flow.flow_id,
                src_ip=flow.src_ip,
                dst_ip=flow.dst_ip,
            )

            # Drift detection
            await self._drift_detector.update(features, result.predicted_class)

            # Threat scoring
            threat_score = self._threat_scorer.compute(result)
            severity = self._threat_scorer.get_severity(threat_score)

            # Protocol info
            proto = PROTOCOL_MAP.get(flow.protocol, "UNK")

            # Update dashboard
            if self._dashboard:
                self._dashboard.on_flow_classified(
                    src_ip=flow.src_ip,
                    dst_ip=flow.dst_ip,
                    proto=proto,
                    size=flow.total_bytes,
                    label=result.predicted_class,
                    inference_ms=result.inference_time_ms,
                )

                if result.is_threat:
                    self._dashboard.on_threat_detected(
                        attack_type=result.predicted_class,
                        src_ip=flow.src_ip,
                        dst_ip=flow.dst_ip,
                        confidence=result.confidence,
                        threat_score=threat_score,
                        severity=severity,
                        flow_id=flow.flow_id,
                    )

            # Automated response
            if result.is_threat:
                self._stats["threats"] += 1
                await self._response_engine.respond(result, threat_score, features)

                # Persist to DB
                await self._db.insert_incident(
                    result, threat_score, severity,
                    is_blocked=self._ip_blocker.is_blocked(flow.src_ip),
                    dry_run=not self._config.live_defense,
                )
            else:
                self._stats["benign"] += 1
                if self._river_learner:
                    await self._river_learner.learn(features, "benign")

        except Exception as exc:
            log.debug("Flow processing error", flow_id=flow.flow_id, error=str(exc))

    # ── Demo mode ─────────────────────────────────────────────────────────────

    async def _demo_traffic_loop(self) -> None:
        """
        Simulate realistic network traffic in demo mode.
        Generates a mix of benign flows and attacks at regular intervals.
        """
        import random
        import ipaddress

        attack_schedule = [
            ("benign",     0.60),
            ("portscan",   0.10),
            ("ddos",       0.08),
            ("bruteforce", 0.07),
            ("bot",        0.06),
            ("dos",        0.05),
            ("webattack",  0.03),
            ("heartbleed", 0.01),
        ]

        rng = random.Random(42)
        flow_counter = 0

        while self._running:
            try:
                await asyncio.sleep(0.2)   # 5 flows/second in demo

                flow_counter += 1
                self._stats["flows"] += 1

                # Pick attack type by probability
                attack_type = rng.choices(
                    [a[0] for a in attack_schedule],
                    weights=[a[1] for a in attack_schedule],
                )[0]

                # Generate synthetic IPs
                src_ip = str(ipaddress.IPv4Address(rng.randint(0x01000001, 0xFEFFFFFF)))
                dst_ip = "10.0.0.1" if attack_type != "benign" else str(
                    ipaddress.IPv4Address(rng.randint(0x01000001, 0xFEFFFFFF))
                )
                proto = rng.choice(["TCP", "UDP", "ICMP"])
                size = rng.randint(64, 65535)

                # Generate synthetic features
                features = self._feature_extractor.extract_demo(attack_type)

                # Run through AI
                if self._model_manager.is_primary_ready:
                    result = await self._predictor.predict(
                        features, flow_id=f"demo-{flow_counter:06d}",
                        src_ip=src_ip, dst_ip=dst_ip,
                    )
                    threat_score = self._threat_scorer.compute(result)
                    severity = self._threat_scorer.get_severity(threat_score)
                    is_threat = result.is_threat
                else:
                    # Use the simulated attack type directly
                    is_threat = attack_type != "benign"
                    threat_score = 7.5 if is_threat else 0.0
                    severity = "HIGH" if threat_score >= 7 else "LOW"
                    result = type("R", (), {
                        "predicted_class": attack_type,
                        "confidence": 0.85,
                        "src_ip": src_ip,
                        "dst_ip": dst_ip,
                        "flow_id": f"demo-{flow_counter:06d}",
                        "is_threat": is_threat,
                        "inference_time_ms": 2.5,
                        "cnn_class": attack_type,
                        "xgb_class": attack_type,
                        "anomaly_score": 0.3,
                        "timestamp": time.time(),
                    })()  # type: ignore

                self._stats["packets"] += rng.randint(5, 100)

                # Update dashboard
                if self._dashboard:
                    self._dashboard.update_packet_stats(
                        self._stats["packets"], self._stats["flows"]
                    )
                    self._dashboard.on_flow_classified(
                        src_ip=src_ip, dst_ip=dst_ip,
                        proto=proto, size=size,
                        label=attack_type,
                        inference_ms=2.5,
                    )

                    if is_threat:
                        self._stats["threats"] += 1
                        self._dashboard.on_threat_detected(
                            attack_type=attack_type,
                            src_ip=src_ip,
                            dst_ip=dst_ip,
                            confidence=result.confidence,
                            threat_score=threat_score,
                            severity=severity,
                        )

                        # Simulate IP block
                        if threat_score >= 9.0:
                            self._dashboard.on_ip_blocked(
                                src_ip, attack_type, dry_run=True
                            )
                            self._stats["blocked"] = self._stats.get("blocked", 0) + 1

                else:
                    self._stats["benign"] += 1

            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.debug("Demo loop error", error=str(exc))

    # ── Periodic stats ────────────────────────────────────────────────────────

    async def _stats_loop(self) -> None:
        """Publish system stats every 30 seconds for database snapshots."""
        while self._running:
            await asyncio.sleep(30)
            try:
                import psutil
                cpu = psutil.cpu_percent(interval=None)
                ram = psutil.virtual_memory().percent
            except ImportError:
                cpu, ram = 0.0, 0.0

            await self._db.insert_system_stats(
                packets=self._stats["packets"],
                flows=self._stats["flows"],
                threats=self._stats["threats"],
                benign=self._stats["benign"],
                cpu=cpu,
                ram=ram,
            )

    # ── Event handlers ────────────────────────────────────────────────────────

    def _register_event_handlers(self) -> None:
        """Register all event bus subscriptions."""

        @self._event_bus.subscribe(EventType.IP_BLOCKED)
        async def on_ip_blocked(event: SentinelEvent) -> None:
            ip = event.payload.get("ip", "")
            reason = event.payload.get("reason", "")
            dry_run = event.payload.get("dry_run", True)
            if self._dashboard:
                self._dashboard.on_ip_blocked(ip, reason, dry_run)

        @self._event_bus.subscribe(EventType.DRIFT_DETECTED)
        async def on_drift(event: SentinelEvent) -> None:
            if self._dashboard:
                self._dashboard.on_drift_detected()
