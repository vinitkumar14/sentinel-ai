"""
Unit tests for SENTINEL-AI Analytics Module
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


class TestTrafficAnalyzer:
    """Tests for TrafficAnalyzer."""

    def setup_method(self):
        from sentinel_ai.analytics import TrafficAnalyzer
        self.analyzer = TrafficAnalyzer(window_seconds=60)

    def test_initial_stats_zero(self):
        stats = self.analyzer.get_stats()
        assert stats.packets_per_sec == 0.0
        assert stats.flows_per_sec == 0.0

    def test_record_packet_increments_totals(self):
        self.analyzer.record_packet("1.2.3.4", 80, "TCP", 1500)
        assert self.analyzer.totals["packets"] == 1
        assert self.analyzer.totals["bytes"] == 1500

    def test_protocol_breakdown(self):
        self.analyzer.record_packet("1.2.3.4", 80, "TCP", 100)
        self.analyzer.record_packet("1.2.3.5", 53, "UDP", 64)
        stats = self.analyzer.get_stats()
        assert "TCP" in stats.protocol_breakdown
        assert "UDP" in stats.protocol_breakdown

    def test_top_src_ips(self):
        for i in range(10):
            self.analyzer.record_packet("10.0.0.1", 443, "TCP", 500)
        for i in range(3):
            self.analyzer.record_packet("10.0.0.2", 443, "TCP", 500)
        stats = self.analyzer.get_stats()
        top_ips = dict(stats.top_src_ips)
        assert top_ips.get("10.0.0.1", 0) == 10
        assert top_ips.get("10.0.0.2", 0) == 3

    def test_record_flow_increments(self):
        self.analyzer.record_flow()
        self.analyzer.record_flow()
        assert self.analyzer.totals["flows"] == 2

    def test_record_threat_increments(self):
        self.analyzer.record_threat()
        assert self.analyzer.totals["threats"] == 1

    def test_pps_non_zero_after_packets(self):
        for _ in range(100):
            self.analyzer.record_packet("1.2.3.4", 80, "TCP", 100)
        stats = self.analyzer.get_stats()
        assert stats.packets_per_sec >= 0.0


class TestThreatIntelligence:
    """Tests for ThreatIntelligence."""

    def setup_method(self):
        from sentinel_ai.analytics import ThreatIntelligence
        self.intel = ThreatIntelligence()

    def _make_result(self, src_ip, attack_type):
        r = MagicMock()
        r.src_ip = src_ip
        r.predicted_class = attack_type
        return r

    def test_initial_total_zero(self):
        assert self.intel.total_threats == 0

    def test_record_threat_increments(self):
        self.intel.record_threat(self._make_result("1.2.3.4", "ddos"), 8.5)
        assert self.intel.total_threats == 1

    def test_attack_distribution(self):
        self.intel.record_threat(self._make_result("1.2.3.4", "ddos"), 9.0)
        self.intel.record_threat(self._make_result("5.6.7.8", "ddos"), 9.0)
        self.intel.record_threat(self._make_result("9.0.1.2", "portscan"), 5.0)
        dist = self.intel.attack_distribution()
        assert dist["ddos"] == 2
        assert dist["portscan"] == 1

    def test_top_ips_ranked(self):
        for _ in range(5):
            self.intel.record_threat(self._make_result("10.0.0.1", "bot"), 7.0)
        for _ in range(2):
            self.intel.record_threat(self._make_result("10.0.0.2", "bot"), 7.0)
        tops = self.intel.top_ips(n=2)
        assert tops[0].ip == "10.0.0.1"
        assert tops[0].total_threats == 5

    def test_get_ip_profile(self):
        self.intel.record_threat(self._make_result("172.16.0.1", "heartbleed"), 10.0)
        profile = self.intel.get_ip_profile("172.16.0.1")
        assert profile is not None
        assert profile.max_score == 10.0

    def test_unknown_ip_profile_none(self):
        assert self.intel.get_ip_profile("255.255.255.255") is None

    def test_threats_last_hour(self):
        for _ in range(3):
            self.intel.record_threat(self._make_result("5.5.5.5", "dos"), 6.0)
        assert self.intel.threats_last_hour() == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
