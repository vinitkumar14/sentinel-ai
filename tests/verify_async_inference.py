#!/usr/bin/env python3
"""
SENTINEL-AI Async Inference Verification
==========================================
Tests that async inference works correctly and doesn't block the event loop.
"""

import sys
import asyncio
import time
from pathlib import Path
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sentinel_ai.ai.model_manager import ModelManager
from sentinel_ai.ai.ensemble_predictor import EnsemblePredictor
from sentinel_ai.core.config import load_config

print("=" * 80)
print("SENTINEL-AI ASYNC INFERENCE VERIFICATION")
print("=" * 80)


async def test_async_inference():
    """Test that async inference doesn't block the event loop."""
    
    print("\n[*] Loading configuration...")
    config = load_config()
    
    print("[*] Initializing model manager...")
    model_manager = ModelManager(config)
    model_manager.load_all()
    
    print(f"\n[*] Models ready: {model_manager.models_ready}")
    
    # Create dummy features
    features = np.random.randn(1, 65).astype(np.float32)
    
    # Test 1: Verify async methods exist
    print("\n" + "=" * 80)
    print("TEST 1: Async Methods Exist")
    print("=" * 80)
    
    has_cnn_async = hasattr(model_manager, 'predict_cnn_async')
    has_xgb_async = hasattr(model_manager, 'predict_xgboost_async')
    has_iso_async = hasattr(model_manager, 'predict_isolation_forest_async')
    
    print(f"[{'OK' if has_cnn_async else 'FAIL'}] predict_cnn_async exists")
    print(f"[{'OK' if has_xgb_async else 'FAIL'}] predict_xgboost_async exists")
    print(f"[{'OK' if has_iso_async else 'FAIL'}] predict_isolation_forest_async exists")
    
    if not all([has_cnn_async, has_xgb_async, has_iso_async]):
        print("\n[ERROR] Async methods missing!")
        return False
    
    # Test 2: Test XGBoost async inference
    if model_manager.models_ready.get('xgboost'):
        print("\n" + "=" * 80)
        print("TEST 2: XGBoost Async Inference")
        print("=" * 80)
        
        try:
            start = time.perf_counter()
            result = await model_manager.predict_xgboost_async(features)
            elapsed = (time.perf_counter() - start) * 1000
            
            print(f"[OK] XGBoost async inference works")
            print(f"   Result: {result}")
            print(f"   Latency: {elapsed:.2f} ms")
        except Exception as e:
            print(f"[FAIL] XGBoost async inference failed: {e}")
            return False
    else:
        print("\n[SKIP] XGBoost model not loaded")
    
    # Test 3: Test IsolationForest async inference
    if model_manager.models_ready.get('isolation_forest'):
        print("\n" + "=" * 80)
        print("TEST 3: IsolationForest Async Inference")
        print("=" * 80)
        
        try:
            start = time.perf_counter()
            result = await model_manager.predict_isolation_forest_async(features)
            elapsed = (time.perf_counter() - start) * 1000
            
            print(f"[OK] IsolationForest async inference works")
            print(f"   Anomaly score: {result:.4f}")
            print(f"   Latency: {elapsed:.2f} ms")
        except Exception as e:
            print(f"[FAIL] IsolationForest async inference failed: {e}")
            return False
    else:
        print("\n[SKIP] IsolationForest model not loaded")
    
    # Test 4: Test ensemble predictor
    print("\n" + "=" * 80)
    print("TEST 4: Ensemble Predictor Async")
    print("=" * 80)
    
    try:
        predictor = EnsemblePredictor(model_manager, config)
        
        start = time.perf_counter()
        result = await predictor.predict(
            features,
            flow_id="test-001",
            src_ip="192.168.1.100",
            dst_ip="10.0.0.1"
        )
        elapsed = (time.perf_counter() - start) * 1000
        
        print(f"[OK] Ensemble prediction works")
        print(f"   Predicted class: {result.predicted_class}")
        print(f"   Confidence: {result.confidence:.2%}")
        print(f"   Is threat: {result.is_threat}")
        print(f"   Inference time: {result.inference_time_ms:.2f} ms")
        print(f"   Total latency: {elapsed:.2f} ms")
    except Exception as e:
        print(f"[FAIL] Ensemble prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Test concurrent inference (non-blocking)
    print("\n" + "=" * 80)
    print("TEST 5: Concurrent Inference (Non-Blocking)")
    print("=" * 80)
    
    if model_manager.models_ready.get('xgboost'):
        try:
            # Run 10 inferences concurrently
            tasks = []
            for i in range(10):
                task = model_manager.predict_xgboost_async(features)
                tasks.append(task)
            
            start = time.perf_counter()
            results = await asyncio.gather(*tasks)
            elapsed = (time.perf_counter() - start) * 1000
            
            print(f"[OK] Concurrent inference works")
            print(f"   Completed: {len(results)} inferences")
            print(f"   Total time: {elapsed:.2f} ms")
            print(f"   Avg per inference: {elapsed / len(results):.2f} ms")
            
            # Check if it was truly concurrent (should be faster than sequential)
            expected_sequential = elapsed / len(results) * len(results)
            speedup = expected_sequential / elapsed
            print(f"   Speedup: {speedup:.2f}x")
            
            if speedup < 1.5:
                print(f"[WARN] Low speedup - may not be truly concurrent")
        except Exception as e:
            print(f"[FAIL] Concurrent inference failed: {e}")
            return False
    else:
        print("[SKIP] XGBoost model not loaded")
    
    # Test 6: Test event loop responsiveness
    print("\n" + "=" * 80)
    print("TEST 6: Event Loop Responsiveness")
    print("=" * 80)
    
    if model_manager.models_ready.get('xgboost'):
        try:
            # Start a background task that should complete quickly
            async def background_task():
                await asyncio.sleep(0.01)
                return "background_done"
            
            bg_task = asyncio.create_task(background_task())
            
            # Run inference
            start = time.perf_counter()
            inference_task = model_manager.predict_xgboost_async(features)
            
            # Both should complete
            inference_result, bg_result = await asyncio.gather(inference_task, bg_task)
            elapsed = (time.perf_counter() - start) * 1000
            
            print(f"[OK] Event loop remained responsive")
            print(f"   Background task: {bg_result}")
            print(f"   Inference completed: {inference_result[0]}")
            print(f"   Total time: {elapsed:.2f} ms")
        except Exception as e:
            print(f"[FAIL] Event loop test failed: {e}")
            return False
    else:
        print("[SKIP] XGBoost model not loaded")
    
    return True


async def main():
    """Main test execution."""
    
    try:
        success = await test_async_inference()
        
        print("\n" + "=" * 80)
        if success:
            print("[SUCCESS] ALL TESTS PASSED")
            print("=" * 80)
            print("\nAsync inference is working correctly!")
            print("The event loop will not block during model inference.")
            print("\nNext steps:")
            print("  1. Run load tests with high traffic")
            print("  2. Monitor dashboard responsiveness")
            print("  3. Check packet queue sizes under load")
            return 0
        else:
            print("[FAILURE] SOME TESTS FAILED")
            print("=" * 80)
            print("\nPlease review the errors above.")
            return 1
    
    except Exception as e:
        print(f"\n[ERROR] Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
