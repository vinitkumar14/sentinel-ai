"""
Unit tests for SENTINEL-AI Threat Scorer
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _make_config():
    """Create a mock config for testing."""
    cfg = MagicMock()
    cfg.defense.thresholds.critical = 9.0
    cfg.defense.thresholds.high = 7.0
    cfg.defense.thresholds.medium = 5.0
    cfg.defense.thresholds.low = 3.0
    cfg.ai.anomaly_threshold = 0.5
    return cfg


def _make_result(predicted_class="benign", confidence=0.9, anomaly_score=0.0, src_ip="192.168.1.1"):
    """Create a mock PredictionResult."""
    result = MagicMock()
    result.predicted_class = predicted_class
    result.confidence = confidence
    result.anomaly_score = anomaly_score
    result.src_ip = src_ip
    result.is_threat = predicted_class != "benign"
    return result


class TestThreatScorer:
    """Tests for ThreatScorer."""

    def setup_method(self):
        from sentinel_ai.ai.threat_scorer import ThreatScorer
        self.scorer = ThreatScorer(_make_config())

    def test_benign_returns_zero(self):
        result = _make_result("benign")
        score = self.scorer.compute(result)
        assert score == 0.0

    def test_ddos_returns_high_score(self):
        result = _make_result("ddos", confidence=0.95, anomaly_score=0.8)
        score = self.scorer.compute(result)
        assert score >= 7.0, f"Expected score >= 7.0, got {score}"
        assert score <= 10.0, f"Score must not exceed 10.0"

    def test_bruteforce_moderate_score(self):
        result = _make_result("bruteforce", confidence=0.75)
        score = self.scorer.compute(result)
        assert score >= 3.0, f"Expected score >= 3.0, got {score}"

    def test_score_clamped_to_10(self):
        result = _make_result("ddos", confidence=1.0, anomaly_score=1.0)
        score = self.scorer.compute(result)
        assert score <= 10.0

    def test_score_non_negative(self):
        for attack in ["bot", "dos", "portscan", "webattack", "heartbleed"]:
            result = _make_result(attack, confidence=0.5)
            score = self.scorer.compute(result)
            assert score >= 0.0, f"Negative score for {attack}"

    def test_severity_critical(self):
        assert self.scorer.get_severity(9.5) == "CRITICAL"

    def test_severity_high(self):
        assert self.scorer.get_severity(7.5) == "HIGH"

    def test_severity_medium(self):
        assert self.scorer.get_severity(5.5) == "MEDIUM"

    def test_severity_low(self):
        assert self.scorer.get_severity(3.5) == "LOW"

    def test_severity_info(self):
        assert self.scorer.get_severity(1.0) == "INFO"

    def test_frequency_multiplier_increases_with_repeat(self):
        """Repeat attacks from same IP should get higher scores."""
        ip = "10.0.0.1"
        result1 = _make_result("portscan", confidence=0.8, src_ip=ip)
        score1 = self.scorer.compute(result1)

        # Same IP, many more attacks
        for _ in range(15):
            r = _make_result("portscan", confidence=0.8, src_ip=ip)
            self.scorer.compute(r)

        score_later = self.scorer.compute(_make_result("portscan", confidence=0.8, src_ip=ip))
        assert score_later >= score1, "Repeated IP should get equal or higher score"

    def test_ip_threat_level(self):
        ip = "172.16.0.99"
        for _ in range(5):
            self.scorer.compute(_make_result("bot", confidence=0.85, src_ip=ip))
        level = self.scorer.get_ip_threat_level(ip)
        assert level > 0.0

    def test_unknown_ip_threat_level_zero(self):
        level = self.scorer.get_ip_threat_level("1.2.3.4")
        assert level == 0.0


class TestThreatScorerEdgeCases:
    """Edge case tests for ThreatScorer."""

    def setup_method(self):
        from sentinel_ai.ai.threat_scorer import ThreatScorer
        self.scorer = ThreatScorer(_make_config())

    def test_zero_confidence(self):
        result = _make_result("ddos", confidence=0.0)
        score = self.scorer.compute(result)
        assert score >= 0.0

    def test_max_anomaly_score(self):
        result = _make_result("bot", confidence=0.7, anomaly_score=1.0)
        score = self.scorer.compute(result)
        assert score <= 10.0

    def test_heartbleed_gets_max_base_score(self):
        result = _make_result("heartbleed", confidence=0.99)
        score = self.scorer.compute(result)
        assert score >= 8.0  # Heartbleed base is 10.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
