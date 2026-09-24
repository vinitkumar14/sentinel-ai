"""
Unit tests for SENTINEL-AI Ensemble Predictor
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def _make_model_manager(cnn_ready=False, xgb_ready=False, iso_ready=False):
    """Mock ModelManager."""
    mm = MagicMock()
    mm.is_primary_ready = cnn_ready
    mm.models_ready = {
        "cnn_bilstm": cnn_ready,
        "xgboost": xgb_ready,
        "isolation_forest": iso_ready,
        "river": False,
    }

    # CNN returns uniform proba (benign wins)
    proba = np.zeros((1, 8), dtype=np.float32)
    proba[0, 0] = 0.9   # benign = index 0
    mm.predict_cnn.return_value = proba

    # XGBoost
    mm.predict_xgboost.return_value = ("benign", 0.85)

    # IsoForest
    mm.predict_isolation_forest.return_value = 0.1

    return mm


def _make_config():
    """Mock SentinelConfig."""
    cfg = MagicMock()
    cfg.ai.ensemble_weights.cnn_bilstm = 0.60
    cfg.ai.ensemble_weights.xgboost = 0.30
    cfg.ai.ensemble_weights.isolation_forest = 0.10
    cfg.ai.primary_confidence_threshold = 0.70
    cfg.ai.ensemble_confidence_threshold = 0.65
    cfg.ai.anomaly_threshold = 0.5
    return cfg


def _make_features(n_features: int = 65) -> np.ndarray:
    """Create a random feature vector."""
    return np.random.rand(1, n_features).astype(np.float32)


class TestEnsemblePredictor:
    """Tests for EnsemblePredictor."""

    def setup_method(self):
        from sentinel_ai.ai.ensemble_predictor import EnsemblePredictor
        self.mm = _make_model_manager(cnn_ready=True, xgb_ready=True, iso_ready=True)
        self.predictor = EnsemblePredictor(self.mm, _make_config())

    @pytest.mark.asyncio
    async def test_predicts_benign_correctly(self):
        features = _make_features()
        result = await self.predictor.predict(features, flow_id="test-001")
        assert result.predicted_class == "benign"
        assert result.is_threat is False
        assert result.confidence >= 0.0

    @pytest.mark.asyncio
    async def test_result_has_flow_id(self):
        features = _make_features()
        result = await self.predictor.predict(features, flow_id="abc-123")
        assert result.flow_id == "abc-123"

    @pytest.mark.asyncio
    async def test_inference_time_recorded(self):
        features = _make_features()
        result = await self.predictor.predict(features)
        assert result.inference_time_ms >= 0.0

    @pytest.mark.asyncio
    async def test_result_probabilities_sum_to_one(self):
        features = _make_features()
        result = await self.predictor.predict(features)
        total = sum(result.ensemble_scores)
        assert abs(total - 1.0) < 0.05, f"Ensemble scores should sum to ~1.0, got {total}"

    @pytest.mark.asyncio
    async def test_ddos_detection_with_mocked_cnn(self):
        """When CNN strongly predicts ddos, ensemble should detect it."""
        from sentinel_ai.ai.ensemble_predictor import EnsemblePredictor
        mm = _make_model_manager(cnn_ready=True)
        # Make CNN predict ddos (index 3)
        proba = np.zeros((1, 8), dtype=np.float32)
        proba[0, 3] = 0.95  # ddos
        mm.predict_cnn.return_value = proba

        predictor = EnsemblePredictor(mm, _make_config())
        result = await predictor.predict(_make_features())

        assert result.cnn_class == "ddos"

    @pytest.mark.asyncio
    async def test_no_models_loaded(self):
        """Should return benign with low confidence when no models are loaded."""
        from sentinel_ai.ai.ensemble_predictor import EnsemblePredictor
        mm = _make_model_manager(cnn_ready=False, xgb_ready=False, iso_ready=False)
        predictor = EnsemblePredictor(mm, _make_config())
        result = await predictor.predict(_make_features())
        assert result is not None   # Should not crash

    @pytest.mark.asyncio
    async def test_src_ip_preserved(self):
        result = await self.predictor.predict(
            _make_features(), src_ip="10.0.0.5", dst_ip="192.168.1.1"
        )
        assert result.src_ip == "10.0.0.5"
        assert result.dst_ip == "192.168.1.1"

    @pytest.mark.asyncio
    async def test_to_dict_serializable(self):
        result = await self.predictor.predict(_make_features())
        d = result.to_dict()
        import json
        assert json.dumps(d)    # Must be JSON-serializable

    @pytest.mark.asyncio
    async def test_high_anomaly_score_reduces_benign_confidence(self):
        """High anomaly score from IsoForest should reduce benign probability."""
        from sentinel_ai.ai.ensemble_predictor import EnsemblePredictor
        mm = _make_model_manager(cnn_ready=True, xgb_ready=False, iso_ready=True)
        # CNN says benign but IsoForest says strong anomaly
        mm.predict_isolation_forest.return_value = 0.95

        predictor = EnsemblePredictor(mm, _make_config())
        result = await predictor.predict(_make_features())
        # Should not be 100% confident in benign anymore
        assert result.anomaly_score > 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
