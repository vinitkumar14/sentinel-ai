"""
SENTINEL-AI Model Trainer
===========================
Full retraining pipeline for the CNN+BiLSTM+Attention model using
the CICIDS2017 dataset (prepared_train_test.npz).

Replicates and extends the training notebook's approach:
  - Loads pre-split X_train/y_train/X_test/y_test from .npz
  - Applies class weights to handle severe imbalance
  - EarlyStopping + ReduceLROnPlateau callbacks
  - Saves model.h5 + training history CSV

Usage:
    python -m sentinel_ai.training.trainer
    # or
    from sentinel_ai.training.trainer import Trainer
    trainer = Trainer(config)
    trainer.train()
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

from sentinel_ai.ai.cnn_bilstm_attention import build_cnn_bilstm_attention
from sentinel_ai.core.constants import ATTACK_CLASSES, NUM_CLASSES
from sentinel_ai.core.logger import get_logger

if TYPE_CHECKING:
    from sentinel_ai.core.config import SentinelConfig

log = get_logger(__name__)


class Trainer:
    """
    Full training pipeline for SENTINEL-AI models.

    Args:
        config: Loaded SentinelConfig.
    """

    def __init__(self, config: "SentinelConfig") -> None:
        self._config = config
        self._model = None
        self._history = None

    def load_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load the pre-processed training data from artifacts.

        Returns:
            Tuple of (X_train, y_train, X_test, y_test).

        Raises:
            FileNotFoundError: If prepared_train_test.npz is not found.
        """
        npz_path = self._config.get_artifact_path("prepared_train_test.npz")
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Training data not found at {npz_path}. "
                "Run scripts/import_artifacts.py first."
            )

        log.info("Loading training data", path=str(npz_path))
        data = np.load(str(npz_path))

        X_train = data["X_train"].astype(np.float32)
        y_train = data["y_train"].astype(np.int32)
        X_test = data["X_test"].astype(np.float32)
        y_test = data["y_test"].astype(np.int32)

        log.info(
            "Training data loaded",
            train_samples=len(X_train),
            test_samples=len(X_test),
            features=X_train.shape[1],
            classes=NUM_CLASSES,
        )

        return X_train, y_train, X_test, y_test

    def _compute_class_weights(self, y: np.ndarray) -> dict[int, float]:
        """Compute inverse-frequency class weights."""
        from sklearn.utils.class_weight import compute_class_weight  # type: ignore
        classes = np.unique(y)
        weights = compute_class_weight("balanced", classes=classes, y=y)
        return {int(cls): float(w) for cls, w in zip(classes, weights)}

    def train(
        self,
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-4,
        early_stopping_patience: int = 4,
    ) -> dict:
        """
        Train the CNN+BiLSTM+Attention model end-to-end.

        Args:
            epochs:                  Maximum training epochs.
            batch_size:              Mini-batch size.
            learning_rate:           Adam optimizer learning rate.
            early_stopping_patience: Stop if val_loss doesn't improve.

        Returns:
            Dict with final metrics (val_accuracy, val_loss, etc.)
        """
        try:
            import tensorflow as tf
        except ImportError:
            raise ImportError("TensorFlow required: pip install tensorflow>=2.15.0")

        # Configure GPU memory growth
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)

        # Load data
        X_train, y_train, X_test, y_test = self.load_data()

        # Reshape for CNN: (N, 65) → (N, 65, 1)
        X_train_3d = X_train[:, :, np.newaxis]
        X_test_3d = X_test[:, :, np.newaxis]

        # Class weights
        class_weights = self._compute_class_weights(y_train)
        log.info("Class weights computed", weights={
            ATTACK_CLASSES[k]: round(v, 3) for k, v in class_weights.items()
        })

        # Build model
        self._model = build_cnn_bilstm_attention(
            num_features=X_train.shape[1],
            num_classes=NUM_CLASSES,
            learning_rate=learning_rate,
            compile_model=True,
        )

        log.info("Model built", params=f"{self._model.count_params():,}")

        # Callbacks
        model_path = (
            self._config.project_root
            / "models" / "cnn_bilstm_attention" / "model.h5"
        )
        model_path.parent.mkdir(parents=True, exist_ok=True)

        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss",
                factor=0.3,
                patience=2,
                min_lr=1e-6,
                verbose=1,
            ),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=str(model_path),
                monitor="val_loss",
                save_best_only=True,
                verbose=1,
            ),
            tf.keras.callbacks.CSVLogger(
                filename=str(
                    self._config.project_root / "artifacts" / "train_epoch_logs_new.csv"
                )
            ),
        ]

        # Train
        log.info("Starting training", epochs=epochs, batch_size=batch_size)
        t_start = time.time()

        self._history = self._model.fit(
            X_train_3d, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test_3d, y_test),
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=2,
        )

        elapsed = time.time() - t_start
        log.info("Training complete", elapsed_seconds=round(elapsed, 1))

        # Final evaluation
        results = self._model.evaluate(X_test_3d, y_test, verbose=0)
        metrics = dict(zip(self._model.metrics_names, results))

        log.info(
            "Final evaluation",
            val_loss=round(metrics.get("loss", 0), 4),
            val_accuracy=round(metrics.get("accuracy", 0), 4),
        )

        return metrics

    def also_train_xgboost(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """Train and save the XGBoost secondary model."""
        from sentinel_ai.ai.xgboost_engine import XGBoostEngine
        engine = XGBoostEngine(self._config)
        engine.train(X_train, y_train)
        engine.save()

    def also_train_isolation_forest(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> None:
        """Train IsolationForest on benign traffic only."""
        from sentinel_ai.ai.isolation_forest import IsolationForestEngine
        engine = IsolationForestEngine(self._config)
        benign_mask = y_train == 0
        X_benign = X_train[benign_mask]
        log.info("Training IsolationForest on benign samples", count=len(X_benign))
        engine.train(X_benign)
        engine.save()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from sentinel_ai.core.config import load_config
    from sentinel_ai.core.logger import setup_logging

    config = load_config()
    setup_logging(config)

    trainer = Trainer(config)
    try:
        X_train, y_train, X_test, y_test = trainer.load_data()
        metrics = trainer.train()
        trainer.also_train_xgboost(X_train, y_train)
        trainer.also_train_isolation_forest(X_train, y_train)
        print(f"\n✅ All models trained. Final val_accuracy: {metrics.get('accuracy', 0):.4f}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
