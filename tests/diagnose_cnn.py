#!/usr/bin/env python3
"""
CNN Model Loading Diagnosis
"""

import sys
sys.path.insert(0, ".")

from pathlib import Path
import traceback

model_path = Path("models/cnn_bilstm_attention/model.h5")

print(f"Model file: {model_path}")
print(f"Exists: {model_path.exists()}")
print(f"Size: {model_path.stat().st_size / (1024*1024):.2f} MB")
print()

try:
    import tensorflow as tf
    print(f"TensorFlow version: {tf.__version__}")
    print()
    
    print("Attempting to load model...")
    model = tf.keras.models.load_model(str(model_path))
    print("SUCCESS!")
    print(f"Model type: {type(model)}")
    print(f"Input shape: {model.input_shape}")
    print(f"Output shape: {model.output_shape}")
    
except Exception as e:
    print(f"FAILED: {type(e).__name__}")
    print(f"Message: {e}")
    print()
    print("Full traceback:")
    traceback.print_exc()
    print()
    
    # Try to inspect the H5 file structure
    print("=" * 70)
    print("H5 File Structure Inspection")
    print("=" * 70)
    try:
        import h5py
        with h5py.File(model_path, 'r') as f:
            print(f"Root keys: {list(f.keys())}")
            if 'model_config' in f.attrs:
                import json
                config = json.loads(f.attrs['model_config'])
                print(f"Model config keys: {config.keys() if isinstance(config, dict) else 'N/A'}")
    except Exception as h5_err:
        print(f"H5 inspection failed: {h5_err}")
