#!/usr/bin/env python3
"""
SENTINEL-AI Deep Audit Test Suite
Comprehensive validation of all components
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, ".")

# Colors for terminal output
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

def test_section(title):
    """Print a test section header"""
    print(f"\n{BLUE}{'=' * 70}{RESET}")
    print(f"{BLUE}{title:^70}{RESET}")
    print(f"{BLUE}{'=' * 70}{RESET}\n")

def test_pass(msg):
    print(f"{GREEN}[PASS]{RESET} {msg}")

def test_fail(msg):
    print(f"{RED}[FAIL]{RESET} {msg}")

def test_warn(msg):
    print(f"{YELLOW}[WARN]{RESET} {msg}")

# ============================================================================
# TEST 1: FILE INTEGRITY
# ============================================================================

test_section("TEST 1: FILE INTEGRITY & ARTIFACT AVAILABILITY")

models_path = Path("models")
artifacts_path = Path("artifacts")

files_to_check = [
    ("models/cnn_bilstm_attention/model.h5", models_path / "cnn_bilstm_attention" / "model.h5"),
    ("models/xgboost/model.pkl", models_path / "xgboost" / "model.pkl"),
    ("models/isolation_forest/model.pkl", models_path / "isolation_forest" / "model.pkl"),
    ("models/river/online_model.pkl", models_path / "river" / "online_model.pkl"),
    ("artifacts/scaler.pkl", artifacts_path / "scaler.pkl"),
    ("artifacts/selector.pkl", artifacts_path / "selector.pkl"),
    ("artifacts/label_encoder.pkl", artifacts_path / "label_encoder.pkl"),
]

all_files_exist = True
for name, path in files_to_check:
    exists = path.exists()
    if exists:
        size_kb = path.stat().st_size / 1024
        test_pass(f"{name:<45} {size_kb:>10.1f} KB")
    else:
        test_fail(f"{name:<45} MISSING")
        all_files_exist = False

if all_files_exist:
    test_pass("All artifact files present")
else:
    test_fail("Some artifact files missing")

# ============================================================================
# TEST 2: CONFIG LOADING
# ============================================================================

test_section("TEST 2: CONFIGURATION SYSTEM")

try:
    from sentinel_ai.core.config import load_config
    test_pass("Config module imports successfully")
    
    cfg = load_config()
    test_pass(f"Config loads: {cfg.system.name} v{cfg.system.version}")
    test_pass(f"  - Capture interface: {cfg.capture.interface}")
    test_pass(f"  - Defense mode: {cfg.defense.mode}")
    test_pass(f"  - AI models: CNN={cfg.ai.primary_model}, XGB={cfg.ai.secondary_model}")
except Exception as e:
    test_fail(f"Config loading failed: {type(e).__name__}: {e}")
    traceback.print_exc()

# ============================================================================
# TEST 3: MODEL LOADING & COMPATIBILITY
# ============================================================================

test_section("TEST 3: MODEL LOADING & COMPATIBILITY")

# TensorFlow CNN Model
try:
    import tensorflow as tf
    model_path = models_path / "cnn_bilstm_attention" / "model.h5"
    if model_path.exists():
        print(f"Loading {model_path}...")
        model = tf.keras.models.load_model(str(model_path))
        test_pass(f"CNN+BiLSTM model loaded")
        test_pass(f"  - Input shape: {model.input_shape}")
        test_pass(f"  - Output shape: {model.output_shape}")
        test_pass(f"  - Parameters: {model.count_params():,}")
        
        # Test inference
        import numpy as np
        dummy = np.random.randn(1, 65, 1).astype(np.float32)
        output = model.predict(dummy, verbose=0)
        test_pass(f"  - Inference test: output shape {output.shape}")
    else:
        test_fail(f"CNN model file not found: {model_path}")
except Exception as e:
    test_fail(f"CNN model loading failed: {type(e).__name__}: {str(e)[:100]}")

# XGBoost Model
try:
    import joblib
    xgb_path = models_path / "xgboost" / "model.pkl"
    if xgb_path.exists():
        xgb_model = joblib.load(str(xgb_path))
        test_pass(f"XGBoost model loaded: {type(xgb_model).__name__}")
        
        # Test inference
        import numpy as np
        dummy = np.random.randn(1, 65).astype(np.float32)
        try:
            proba = xgb_model.predict_proba(dummy)
            test_pass(f"  - Inference test: proba shape {proba.shape}")
        except Exception as inf_err:
            test_warn(f"  - Inference test failed: {type(inf_err).__name__}")
    else:
        test_fail(f"XGBoost model file not found: {xgb_path}")
except Exception as e:
    test_fail(f"XGBoost model loading failed: {type(e).__name__}: {str(e)[:100]}")

# IsolationForest Model
try:
    import joblib
    iso_path = models_path / "isolation_forest" / "model.pkl"
    if iso_path.exists():
        iso_model = joblib.load(str(iso_path))
        test_pass(f"IsolationForest model loaded: {type(iso_model).__name__}")
        
        # Test inference
        import numpy as np
        dummy = np.random.randn(1, 65).astype(np.float32)
        try:
            score = iso_model.decision_function(dummy)
            test_pass(f"  - Inference test: anomaly score {score[0]:.4f}")
        except Exception as inf_err:
            test_warn(f"  - Inference test failed: {type(inf_err).__name__}")
    else:
        test_fail(f"IsolationForest model file not found: {iso_path}")
except Exception as e:
    test_fail(f"IsolationForest loading failed: {type(e).__name__}: {str(e)[:100]}")

# ============================================================================
# TEST 4: PREPROCESSING ARTIFACTS
# ============================================================================

test_section("TEST 4: PREPROCESSING ARTIFACTS")

try:
    import joblib
    
    scaler_path = artifacts_path / "scaler.pkl"
    scaler = joblib.load(str(scaler_path))
    test_pass(f"Scaler loaded: {type(scaler).__name__}")
    
    selector_path = artifacts_path / "selector.pkl"
    selector = joblib.load(str(selector_path))
    test_pass(f"Selector loaded: {type(selector).__name__}")
    
    label_enc_path = artifacts_path / "label_encoder.pkl"
    label_enc = joblib.load(str(label_enc_path))
    test_pass(f"LabelEncoder loaded: {type(label_enc).__name__}")
    
    # Test on dummy data
    import numpy as np
    raw_features = np.random.randn(1, 77).astype(np.float32)
    
    # Apply selector
    selected = selector.transform(raw_features)
    test_pass(f"  - VarianceThreshold: 77 → {selected.shape[1]} features")
    
    # Apply scaler
    scaled = scaler.transform(selected)
    test_pass(f"  - RobustScaler: scaled to mean={scaled.mean():.4f}, std={scaled.std():.4f}")
    
except Exception as e:
    test_fail(f"Artifact loading failed: {type(e).__name__}: {e}")

# ============================================================================
# TEST 5: ENSEMBLE PREDICTOR
# ============================================================================

test_section("TEST 5: ENSEMBLE PREDICTOR")

try:
    from sentinel_ai.core.config import load_config
    from sentinel_ai.ai.model_manager import ModelManager
    from sentinel_ai.ai.ensemble_predictor import EnsemblePredictor
    import numpy as np
    
    cfg = load_config()
    mgr = ModelManager(cfg)
    mgr.load_all()
    
    test_pass(f"ModelManager initialized")
    test_pass(f"  - CNN+BiLSTM ready: {mgr.is_primary_ready}")
    test_pass(f"  - Models status: {mgr.models_ready}")
    
    predictor = EnsemblePredictor(mgr, cfg)
    test_pass(f"EnsemblePredictor created")
    
    # Test async prediction
    import asyncio
    
    async def test_predict():
        dummy = np.random.randn(1, 65).astype(np.float32)
        result = await predictor.predict(
            dummy, 
            flow_id="test-001",
            src_ip="192.168.1.100",
            dst_ip="10.0.0.1"
        )
        return result
    
    result = asyncio.run(test_predict())
    test_pass(f"Ensemble prediction test completed")
    test_pass(f"  - Predicted class: {result.predicted_class}")
    test_pass(f"  - Confidence: {result.confidence:.4f}")
    test_pass(f"  - Is threat: {result.is_threat}")
    test_pass(f"  - Inference time: {result.inference_time_ms:.2f} ms")
    
except Exception as e:
    test_fail(f"Ensemble predictor failed: {type(e).__name__}: {e}")
    traceback.print_exc()

# ============================================================================
# TEST 6: FEATURE EXTRACTION PIPELINE
# ============================================================================

test_section("TEST 6: FEATURE EXTRACTION PIPELINE")

try:
    from sentinel_ai.core.config import load_config
    from sentinel_ai.capture.feature_extractor import FeatureExtractor
    from sentinel_ai.capture.flow_manager import NetworkFlow, DirectionalFlow
    import numpy as np
    
    cfg = load_config()
    extractor = FeatureExtractor(cfg)
    extractor.load_artifacts()
    
    test_pass("FeatureExtractor initialized and artifacts loaded")
    
    # Create a dummy flow
    flow = NetworkFlow(
        src_ip="192.168.1.100",
        dst_ip="10.0.0.1",
        src_port=55555,
        dst_port=443,
        protocol=6,  # TCP
    )
    
    # Add some dummy packets
    for i in range(10):
        flow.fwd.add_packet(length=100, flags=0x18)  # PSH|ACK
        flow.bwd.add_packet(length=50, flags=0x10)   # ACK
    
    # Extract features
    features = extractor.extract(flow)
    test_pass(f"Features extracted: shape={features.shape}")
    test_pass(f"  - Data type: {features.dtype}")
    test_pass(f"  - Min value: {features.min():.4f}")
    test_pass(f"  - Max value: {features.max():.4f}")
    test_pass(f"  - Has NaN: {np.isnan(features).any()}")
    test_pass(f"  - Has Inf: {np.isinf(features).any()}")
    
except Exception as e:
    test_fail(f"Feature extraction failed: {type(e).__name__}: {e}")
    traceback.print_exc()

# ============================================================================
# TEST 7: CAPTURE PIPELINE
# ============================================================================

test_section("TEST 7: PACKET CAPTURE & FLOW MANAGEMENT")

try:
    from sentinel_ai.core.config import load_config
    from sentinel_ai.capture.packet_sniffer import PacketSniffer, PacketInfo
    from sentinel_ai.capture.flow_manager import FlowManager
    from sentinel_ai.core.events import EventBus
    import asyncio
    
    cfg = load_config()
    event_bus = EventBus(queue_size=1000)
    
    test_pass("EventBus created")
    
    sniffer = PacketSniffer(cfg, event_bus)
    test_pass("PacketSniffer created")
    
    flow_manager = FlowManager(cfg, event_bus)
    test_pass("FlowManager created")
    
    # Test flow processing with synthetic packets
    async def test_flow_proc():
        pkt = PacketInfo(
            src_ip="192.168.1.100",
            dst_ip="10.0.0.1",
            protocol=6,  # TCP
            src_port=55555,
            dst_port=443,
            tcp_flags=0x18,  # SYN
            payload_length=100,
            header_length=40,
        )
        
        completed = await flow_manager.process_packet(pkt)
        return len(completed)
    
    completed_flows = asyncio.run(test_flow_proc())
    test_pass(f"Flow processing works (completed flows: {completed_flows})")
    
except Exception as e:
    test_fail(f"Capture pipeline test failed: {type(e).__name__}: {e}")
    traceback.print_exc()

# ============================================================================
# TEST 8: DASHBOARD
# ============================================================================

test_section("TEST 8: DASHBOARD SYSTEM")

try:
    from sentinel_ai.dashboard.app import SentinelDashboard
    test_pass("SentinelDashboard imports successfully")
    
    # Don't actually launch the dashboard in CLI mode
    test_warn("Dashboard not tested in audit mode (requires terminal)")
    
except Exception as e:
    test_fail(f"Dashboard import failed: {type(e).__name__}: {e}")

# ============================================================================
# TEST 9: DATABASE
# ============================================================================

test_section("TEST 9: DATABASE SYSTEM")

try:
    from sentinel_ai.core.config import load_config
    from sentinel_ai.storage.database import Database
    import asyncio
    
    cfg = load_config()
    db = Database(cfg)
    
    test_pass("Database module imports successfully")
    
    async def test_db():
        await db.init()
        test_pass("Database initialized")
        
        # Test query
        incidents = await db.get_recent_incidents(limit=1)
        test_pass(f"Database query works (recent incidents: {len(incidents)})")
        
        await db.close()
        test_pass("Database closed")
    
    asyncio.run(test_db())
    
except Exception as e:
    test_fail(f"Database test failed: {type(e).__name__}: {e}")
    traceback.print_exc()

# ============================================================================
# SUMMARY
# ============================================================================

test_section("AUDIT COMPLETE")
print(f"{GREEN}Deep audit test suite completed.{RESET}\n")
print("Next steps:")
print("  1. Review all failures above")
print("  2. Run main.py --demo for end-to-end testing")
print("  3. Check logs in logs/ directory")
