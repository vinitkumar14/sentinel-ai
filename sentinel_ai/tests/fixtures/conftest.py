"""
SENTINEL-AI Test Fixtures
===========================
Shared pytest fixtures for unit and integration tests.
Provides sample feature vectors, mock flows, and config factories.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from sentinel_ai.core.constants import ATTACK_CLASSES, NUM_CLASSES, NUM_SELECTED_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# Feature Generators
# ─────────────────────────────────────────────────────────────────────────────

# Known-good feature templates per attack type (normalized, scaled)
_ATTACK_FEATURE_TEMPLATES: dict[str, np.ndarray] = {}


def _build_templates():
    """Build synthetic but realistic feature vectors for each attack type."""
    rng = np.random.default_rng(seed=42)

    templates = {
        "benign":     rng.uniform(0.0, 0.3,  NUM_SELECTED_FEATURES),
        "ddos":       rng.uniform(0.8, 1.0,  NUM_SELECTED_FEATURES),  # High rate
        "dos":        rng.uniform(0.7, 0.95, NUM_SELECTED_FEATURES),
        "bruteforce": rng.uniform(0.5, 0.8,  NUM_SELECTED_FEATURES),
        "portscan":   rng.uniform(0.2, 0.6,  NUM_SELECTED_FEATURES),
        "bot":        rng.uniform(0.3, 0.7,  NUM_SELECTED_FEATURES),
        "heartbleed": rng.uniform(0.6, 0.9,  NUM_SELECTED_FEATURES),
        "webattack":  rng.uniform(0.4, 0.75, NUM_SELECTED_FEATURES),
    }

    # Characteristic overrides for key features
    # Feature 5: Flow_Packets/s — high for DDoS
    templates["ddos"][5] = 1.0
    templates["dos"][5] = 0.9
    templates["benign"][5] = 0.1

    # Feature 12: SYN_Flag_Count — high for port scan
    templates["portscan"][12] = 0.95
    templates["bruteforce"][12] = 0.7

    for k, v in templates.items():
        _ATTACK_FEATURE_TEMPLATES[k] = v.astype(np.float32)


_build_templates()


@pytest.fixture
def features_benign() -> np.ndarray:
    """Feature vector for normal benign traffic."""
    return _ATTACK_FEATURE_TEMPLATES["benign"].reshape(1, -1).copy()


@pytest.fixture
def features_ddos() -> np.ndarray:
    """Feature vector for DDoS traffic."""
    return _ATTACK_FEATURE_TEMPLATES["ddos"].reshape(1, -1).copy()


@pytest.fixture
def features_portscan() -> np.ndarray:
    return _ATTACK_FEATURE_TEMPLATES["portscan"].reshape(1, -1).copy()


@pytest.fixture
def features_bruteforce() -> np.ndarray:
    return _ATTACK_FEATURE_TEMPLATES["bruteforce"].reshape(1, -1).copy()


@pytest.fixture
def features_batch() -> np.ndarray:
    """Batch of 100 random feature vectors."""
    rng = np.random.default_rng(seed=99)
    return rng.uniform(0.0, 1.0, (100, NUM_SELECTED_FEATURES)).astype(np.float32)


@pytest.fixture
def attack_features_all() -> dict[str, np.ndarray]:
    """Dict of attack_type → feature vector for all 8 classes."""
    return {k: v.reshape(1, -1).copy() for k, v in _ATTACK_FEATURE_TEMPLATES.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Config Factories
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def base_config():
    """Minimal SentinelConfig mock for testing."""
    cfg = MagicMock()
    cfg.live_defense = False
    cfg.demo_mode = False
    cfg.system.debug = True

    cfg.ai.ensemble_weights.cnn_bilstm = 0.60
    cfg.ai.ensemble_weights.xgboost = 0.30
    cfg.ai.ensemble_weights.isolation_forest = 0.10
    cfg.ai.primary_confidence_threshold = 0.70
    cfg.ai.ensemble_confidence_threshold = 0.65
    cfg.ai.anomaly_threshold = 0.5
    cfg.ai.online_learning_enabled = False
    cfg.ai.drift_detection_enabled = True
    cfg.ai.drift_window_size = 100
    cfg.ai.drift_threshold = 0.05

    cfg.defense.thresholds.critical = 9.0
    cfg.defense.thresholds.high = 7.0
    cfg.defense.thresholds.medium = 5.0
    cfg.defense.thresholds.low = 3.0
    cfg.defense.whitelist = ["127.0.0.1", "::1", "10.0.0.0/8"]
    cfg.defense.auto_block_enabled = False
    cfg.defense.block_duration = 3600

    return cfg


@pytest.fixture
def event_bus():
    """Mock EventBus with async publish."""
    bus = MagicMock()
    bus.publish = AsyncMock()
    bus.subscribe = MagicMock()
    return bus


@pytest.fixture
def model_manager_benign():
    """ModelManager that always predicts benign."""
    mm = MagicMock()
    mm.is_primary_ready = True
    mm.models_ready = {
        "cnn_bilstm": True, "xgboost": True,
        "isolation_forest": True, "river": False,
    }
    proba = np.zeros((1, 8), dtype=np.float32)
    proba[0, 0] = 0.98   # benign
    mm.predict_cnn.return_value = proba
    mm.predict_xgboost.return_value = ("benign", 0.95)
    mm.predict_isolation_forest.return_value = 0.05
    return mm


@pytest.fixture
def model_manager_ddos():
    """ModelManager that always predicts DDoS."""
    mm = MagicMock()
    mm.is_primary_ready = True
    mm.models_ready = {
        "cnn_bilstm": True, "xgboost": True,
        "isolation_forest": True, "river": False,
    }
    proba = np.zeros((1, 8), dtype=np.float32)
    proba[0, 3] = 0.94   # ddos = index 3
    mm.predict_cnn.return_value = proba
    mm.predict_xgboost.return_value = ("ddos", 0.90)
    mm.predict_isolation_forest.return_value = 0.88
    return mm
