"""
Root conftest.py — shared fixtures available to ALL tests.
Imports from tests/fixtures/conftest.py.
"""

import sys
from pathlib import Path

# Ensure sentinel_ai is importable
sys.path.insert(0, str(Path(__file__).parent))

# Re-export all fixtures from fixtures/conftest.py
from tests.fixtures.conftest import (
    features_benign,
    features_ddos,
    features_portscan,
    features_bruteforce,
    features_batch,
    attack_features_all,
    base_config,
    event_bus,
    model_manager_benign,
    model_manager_ddos,
)

__all__ = [
    "features_benign", "features_ddos", "features_portscan",
    "features_bruteforce", "features_batch", "attack_features_all",
    "base_config", "event_bus", "model_manager_benign", "model_manager_ddos",
]
