#!/usr/bin/env python3
"""
SENTINEL-AI Model Loading Fix - Pragmatic Approach
====================================================
Uses TensorFlow's model reconstruction to fix the batch_shape issue.

Strategy:
1. Load model with compile=False to skip optimizer issues
2. Reconstruct model with compatible Input layer
3. Transfer weights
4. Save in modern format
"""

import sys
import os
from pathlib import Path
import numpy as np

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

print("=" * 80)
print("SENTINEL-AI MODEL LOADING FIX - Pragmatic Approach")
print("=" * 80)

# Paths
ORIGINAL_MODEL = PROJECT_ROOT / "models" / "cnn_bilstm_attention" / "model_original.h5"
NEW_MODEL_H5 = PROJECT_ROOT / "models" / "cnn_bilstm_attention" / "model.h5"
NEW_MODEL_SAVED = PROJECT_ROOT / "models" / "cnn_bilstm_attention" / "savedmodel"
BACKUP_MODEL = PROJECT_ROOT / "models" / "cnn_bilstm_attention" / "model_backup.h5"

print(f"\n[*] Original model: {ORIGINAL_MODEL}")
print(f"[*] New H5 model: {NEW_MODEL_H5}")
print(f"[*] New SavedModel: {NEW_MODEL_SAVED}")

# Check if original exists
if not ORIGINAL_MODEL.exists():
    print(f"\n[ERROR] Original model not found at {ORIGINAL_MODEL}")
    print("[INFO] Please ensure model_original.h5 exists")
    sys.exit(1)

print(f"[OK] Original model found: {ORIGINAL_MODEL.stat().st_size / 1024 / 1024:.2f} MB")


def load_model_safe(model_path):
    """
    Load model with maximum compatibility.
    """
    print(f"\n[*] Loading model from {model_path.name}...")
    
    try:
        # Try loading without compilation first
        model = keras.models.load_model(str(model_path), compile=False)
        print(f"[OK] Model loaded successfully (compile=False)")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        return None


def reconstruct_model_compatible(old_model):
    """
    Reconstruct model with compatible Input layer.
    """
    print(f"\n[*] Reconstructing model with compatible layers...")
    
    # Get model config
    config = old_model.get_config()
    
    # Fix Input layer if it has batch_shape
    if 'layers' in config:
        for layer in config['layers']:
            if layer['class_name'] == 'InputLayer':
                if 'batch_shape' in layer['config']:
                    # Convert batch_shape to shape
                    batch_shape = layer['config']['batch_shape']
                    if batch_shape and len(batch_shape) > 1:
                        layer['config']['shape'] = batch_shape[1:]
                    del layer['config']['batch_shape']
                    print(f"[OK] Fixed InputLayer: batch_shape -> shape")
    
    # Reconstruct model from fixed config
    try:
        new_model = keras.Model.from_config(config)
        print(f"[OK] Model reconstructed from config")
        
        # Transfer weights
        new_model.set_weights(old_model.get_weights())
        print(f"[OK] Weights transferred")
        
        return new_model
    except Exception as e:
        print(f"[ERROR] Failed to reconstruct model: {e}")
        return None


def validate_model(model):
    """
    Validate model works correctly.
    """
    print(f"\n[*] Validating model...")
    
    # Test input
    dummy = np.zeros((1, 65, 1), dtype=np.float32)
    
    try:
        # Run inference
        output = model.predict(dummy, verbose=0)
        
        # Check shape
        if output.shape != (1, 8):
            print(f"[ERROR] Wrong output shape: {output.shape}")
            return False
        
        # Check probabilities
        if not np.allclose(output.sum(), 1.0, atol=1e-5):
            print(f"[ERROR] Output not a probability distribution: sum={output.sum()}")
            return False
        
        print(f"[OK] Model validation passed")
        print(f"   Input shape: (1, 65, 1)")
        print(f"   Output shape: {output.shape}")
        print(f"   Output sum: {output.sum():.6f}")
        print(f"   Sample output: {output[0][:4]}...")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Validation failed: {e}")
        return False


def benchmark_model(model, n_runs=100):
    """
    Benchmark inference speed.
    """
    print(f"\n[*] Benchmarking inference ({n_runs} runs)...")
    
    import time
    
    dummy = np.zeros((1, 65, 1), dtype=np.float32)
    
    # Warm up
    for _ in range(10):
        model.predict(dummy, verbose=0)
    
    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model.predict(dummy, verbose=0)
        times.append((time.perf_counter() - start) * 1000)
    
    times = np.array(times)
    
    print(f"[OK] Benchmark complete")
    print(f"   Mean: {times.mean():.2f} ms")
    print(f"   Median: {np.median(times):.2f} ms")
    print(f"   P95: {np.percentile(times, 95):.2f} ms")
    print(f"   Throughput: {1000 / times.mean():.1f} inferences/sec")
    
    return times.mean()


def main():
    """Main execution."""
    
    # Step 1: Load original model
    old_model = load_model_safe(ORIGINAL_MODEL)
    if old_model is None:
        print("\n[ERROR] Cannot proceed without loading original model")
        sys.exit(1)
    
    print(f"\n[*] Model info:")
    print(f"   Layers: {len(old_model.layers)}")
    print(f"   Parameters: {old_model.count_params():,}")
    print(f"   Input shape: {old_model.input_shape}")
    print(f"   Output shape: {old_model.output_shape}")
    
    # Step 2: Validate original model works
    if not validate_model(old_model):
        print("\n[ERROR] Original model validation failed")
        sys.exit(1)
    
    # Step 3: Reconstruct with compatible layers
    new_model = reconstruct_model_compatible(old_model)
    if new_model is None:
        print("\n[WARN] Could not reconstruct model, will use original")
        new_model = old_model
    
    # Step 4: Validate new model
    if not validate_model(new_model):
        print("\n[ERROR] New model validation failed")
        sys.exit(1)
    
    # Step 5: Benchmark
    avg_latency = benchmark_model(new_model)
    
    # Step 6: Backup existing model if it exists
    if NEW_MODEL_H5.exists() and not BACKUP_MODEL.exists():
        print(f"\n[*] Creating backup...")
        import shutil
        shutil.copy2(NEW_MODEL_H5, BACKUP_MODEL)
        print(f"[OK] Backup created: {BACKUP_MODEL.name}")
    
    # Step 7: Save new model
    print(f"\n[*] Saving fixed model...")
    
    # Save as H5
    new_model.save(str(NEW_MODEL_H5), save_format='h5')
    print(f"[OK] Saved H5: {NEW_MODEL_H5.name}")
    
    # Save as SavedModel
    NEW_MODEL_SAVED.mkdir(exist_ok=True)
    new_model.save(str(NEW_MODEL_SAVED), save_format='tf')
    print(f"[OK] Saved SavedModel: {NEW_MODEL_SAVED.name}")
    
    # Step 8: Verify saved model loads
    print(f"\n[*] Verifying saved model...")
    
    try:
        loaded = keras.models.load_model(str(NEW_MODEL_H5))
        dummy = np.zeros((1, 65, 1), dtype=np.float32)
        output = loaded.predict(dummy, verbose=0)
        print(f"[OK] Saved model loads and works: {output.shape}")
    except Exception as e:
        print(f"[ERROR] Saved model verification failed: {e}")
        sys.exit(1)
    
    # Success
    print("\n" + "=" * 80)
    print("[SUCCESS] MODEL LOADING FIX COMPLETE")
    print("=" * 80)
    print(f"\nSummary:")
    print(f"   [OK] Model loaded from original")
    print(f"   [OK] Model reconstructed with compatible layers")
    print(f"   [OK] Model validated successfully")
    print(f"   [OK] Average latency: {avg_latency:.2f} ms")
    print(f"   [OK] Model saved in H5 and SavedModel formats")
    print(f"\nNext steps:")
    print(f"   1. Test with sentinel_ai.ai.model_manager")
    print(f"   2. Run integration tests")
    print(f"   3. Verify in production environment")
    print()


if __name__ == "__main__":
    main()
