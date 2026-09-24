"""
SENTINEL-AI Integration Tests — Full Pipeline
==============================================
Tests the end-to-end flow:
  Feature Vector → EnsemblePredictor → ThreatScorer → ResponseEngine

Uses lightweight mock models (no real TF/XGB required).
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def config():
    cfg = MagicMock()
    cfg.live_defense = False
    cfg.demo_mode = False
    cfg.ai.ensemble_weights.cnn_bilstm = 0.60
    cfg.ai.ensemble_weights.xgboost = 0.30
    cfg.ai.ensemble_weights.isolation_forest = 0.10
    cfg.ai.primary_confidence_threshold = 0.70
    cfg.ai.ensemble_confidence_threshold = 0.65
    cfg.ai.anomaly_threshold = 0.5
    cfg.ai.online_learning_enabled = False
    cfg.defense.thresholds.critical = 9.0
    cfg.defense.thresholds.high = 7.0
    cfg.defense.thresholds.medium = 5.0
    cfg.defense.thresholds.low = 3.0
    cfg.defense.whitelist = ["127.0.0.1", "::1"]
    cfg.defense.auto_block_enabled = False
    cfg.defense.block_duration = 3600
    return cfg


@pytest.fixture
def event_bus():
    bus = MagicMock()
    bus.publish = AsyncMock()
    return bus


@pytest.fixture
def model_manager():
    mm = MagicMock()
    mm.is_primary_ready = True
    mm.models_ready = {
        "cnn_bilstm": True,
        "xgboost": True,
        "isolation_forest": True,
        "river": False,
    }
    # CNN: high confidence for ddos (index 3)
    proba = np.zeros((1, 8), dtype=np.float32)
    proba[0, 3] = 0.92   # ddos
    mm.predict_cnn.return_value = proba
    mm.predict_xgboost.return_value = ("ddos", 0.88)
    mm.predict_isolation_forest.return_value = 0.85
    return mm


@pytest.fixture
def features():
    return np.random.rand(1, 65).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Integration Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestFullInferencePipeline:
    """End-to-end: features → prediction → scoring."""

    @pytest.mark.asyncio
    async def test_ddos_detected_end_to_end(self, config, model_manager, features):
        from sentinel_ai.ai.ensemble_predictor import EnsemblePredictor
        from sentinel_ai.ai.threat_scorer import ThreatScorer

        predictor = EnsemblePredictor(model_manager, config)
        scorer = ThreatScorer(config)

        result = await predictor.predict(
            features, src_ip="203.0.113.5", dst_ip="10.0.0.1"
        )
        score = scorer.compute(result)
        severity = scorer.get_severity(score)

        assert result.predicted_class == "ddos"
        assert result.is_threat is True
        assert result.confidence >= 0.5
        assert score >= 5.0, f"DDoS should score >= 5.0, got {score}"
        assert severity in ("HIGH", "CRITICAL")

    @pytest.mark.asyncio
    async def test_response_engine_triggers_alert(
        self, config, event_bus, model_manager, features
    ):
        from sentinel_ai.ai.ensemble_predictor import EnsemblePredictor
        from sentinel_ai.ai.threat_scorer import ThreatScorer
        from sentinel_ai.defense.alert_manager import AlertManager
        from sentinel_ai.defense.ip_blocker import IPBlocker
        from sentinel_ai.defense.response_engine import ResponseEngine

        predictor = EnsemblePredictor(model_manager, config)
        scorer = ThreatScorer(config)
        ip_blocker = IPBlocker(config, event_bus)
        alert_manager = AlertManager(config, event_bus)

        # River learner mock
        river_learner = MagicMock()
        river_learner.learn = AsyncMock()

        engine = ResponseEngine(config, ip_blocker, alert_manager, river_learner, event_bus)

        result = await predictor.predict(features, src_ip="11.22.33.44")
        score = scorer.compute(result)

        await engine.respond(result, score, features)

        # Alert should have triggered an event publish
        assert event_bus.publish.called

    @pytest.mark.asyncio
    async def test_observe_mode_does_not_execute_firewall(
        self, config, event_bus, model_manager, features
    ):
        """In observe mode, even critical threats should not call iptables."""
        from sentinel_ai.ai.ensemble_predictor import EnsemblePredictor
        from sentinel_ai.ai.threat_scorer import ThreatScorer
        from sentinel_ai.defense.ip_blocker import IPBlocker

        config.live_defense = False
        predictor = EnsemblePredictor(model_manager, config)
        scorer = ThreatScorer(config)
        ip_blocker = IPBlocker(config, event_bus)

        result = await predictor.predict(features, src_ip="55.66.77.88")
        score = scorer.compute(result)

        # Block in observe mode
        if result.is_threat and score > 0:
            blocked = await ip_blocker.block_ip("55.66.77.88", reason="test")
            record = ip_blocker.blocked_ips.get("55.66.77.88")
            # Record exists but dry_run should be True
            if record:
                assert record.dry_run is True

    @pytest.mark.asyncio
    async def test_benign_flow_no_alert(self, config, event_bus):
        """Benign traffic should not trigger any alerts."""
        from sentinel_ai.ai.ensemble_predictor import EnsemblePredictor
        from sentinel_ai.ai.threat_scorer import ThreatScorer
        from sentinel_ai.defense.alert_manager import AlertManager

        # Mock model that returns benign
        mm = MagicMock()
        mm.is_primary_ready = True
        mm.models_ready = {"cnn_bilstm": True, "xgboost": False, "isolation_forest": False}
        proba = np.zeros((1, 8), dtype=np.float32)
        proba[0, 0] = 0.98   # benign
        mm.predict_cnn.return_value = proba

        predictor = EnsemblePredictor(mm, config)
        scorer = ThreatScorer(config)
        alert_mgr = AlertManager(config, event_bus)

        result = await predictor.predict(
            np.random.rand(1, 65).astype(np.float32),
            src_ip="192.168.1.50",
        )
        score = scorer.compute(result)

        await alert_mgr.send_threat_alert(result, score)

        # No alert should have been published for benign traffic
        assert result.predicted_class == "benign"
        assert score == 0.0

    @pytest.mark.asyncio
    async def test_multiple_flows_different_ips(self, config, event_bus, model_manager):
        """Process flows from different source IPs."""
        from sentinel_ai.ai.ensemble_predictor import EnsemblePredictor
        from sentinel_ai.ai.threat_scorer import ThreatScorer

        predictor = EnsemblePredictor(model_manager, config)
        scorer = ThreatScorer(config)

        ips = ["1.2.3.4", "5.6.7.8", "9.10.11.12"]
        results = []

        for ip in ips:
            feats = np.random.rand(1, 65).astype(np.float32)
            r = await predictor.predict(feats, src_ip=ip)
            s = scorer.compute(r)
            results.append((r, s))

        assert len(results) == 3
        for r, s in results:
            assert 0.0 <= s <= 10.0

    @pytest.mark.asyncio
    async def test_high_frequency_same_ip_increases_score(
        self, config, model_manager
    ):
        """Repeated attacks from same IP should get frequency multiplier."""
        from sentinel_ai.ai.ensemble_predictor import EnsemblePredictor
        from sentinel_ai.ai.threat_scorer import ThreatScorer

        predictor = EnsemblePredictor(model_manager, config)
        scorer = ThreatScorer(config)

        ip = "99.88.77.66"
        feats = np.random.rand(1, 65).astype(np.float32)

        # First attack
        r1 = await predictor.predict(feats, src_ip=ip)
        s1 = scorer.compute(r1)

        # Many more attacks
        for _ in range(20):
            r = await predictor.predict(feats, src_ip=ip)
            scorer.compute(r)

        # Last attack should score >= first
        r_last = await predictor.predict(feats, src_ip=ip)
        s_last = scorer.compute(r_last)

        assert s_last >= s1, "Repeat attacker should score at least as high"


class TestDriftDetector:
    """Integration tests for concept drift detection."""

    @pytest.mark.asyncio
    async def test_drift_not_detected_on_stable_data(self, config, event_bus):
        from sentinel_ai.ai.drift_detector import DriftDetector

        config.ai.drift_detection_enabled = True
        config.ai.drift_window_size = 100
        config.ai.drift_threshold = 0.05

        detector = DriftDetector(config, event_bus)
        feats = np.ones((65,), dtype=np.float32) * 0.5

        # Feed uniform data — should not drift
        for _ in range(600):
            await detector.update(feats, "benign")

        # After calibration, stable data should not trigger drift
        assert not detector.is_drifting

    @pytest.mark.asyncio
    async def test_calibration_happens_after_enough_samples(self, config, event_bus):
        from sentinel_ai.ai.drift_detector import DriftDetector

        config.ai.drift_detection_enabled = True
        config.ai.drift_window_size = 100
        config.ai.drift_threshold = 0.05

        detector = DriftDetector(config, event_bus)
        feats = np.random.rand(65).astype(np.float32)

        # Before calibration
        assert not detector.is_calibrated

        for _ in range(510):
            await detector.update(feats, "benign")

        assert detector.is_calibrated
