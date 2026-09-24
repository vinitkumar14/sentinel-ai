"""
SENTINEL-AI Model Evaluation
==============================
Comprehensive evaluation of all trained models on the test set.

Generates:
  - Classification report (precision, recall, F1 per class)
  - Confusion matrix
  - ROC-AUC scores
  - Ensemble vs. individual model comparison
  - Saves results to artifacts/evaluation_report.json

Usage:
    python -m sentinel_ai.training.evaluate
    python training/evaluate.py --model all
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from sentinel_ai.core.constants import ATTACK_CLASSES
from sentinel_ai.core.logger import get_logger

if TYPE_CHECKING:
    from sentinel_ai.core.config import SentinelConfig

log = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluates all SENTINEL-AI models on the held-out test set.

    Args:
        config: Loaded SentinelConfig.
    """

    def __init__(self, config: "SentinelConfig") -> None:
        self._config = config
        self._results: dict = {}

    def load_test_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Load X_test, y_test from artifacts."""
        npz_path = self._config.get_artifact_path("prepared_train_test.npz")
        if not npz_path.exists():
            raise FileNotFoundError(
                f"Test data not found at {npz_path}.\n"
                "Run scripts/import_artifacts.py first."
            )
        data = np.load(str(npz_path))
        X_test = data["X_test"].astype(np.float32)
        y_test = data["y_test"].astype(np.int32)
        log.info("Test data loaded", samples=len(X_test), features=X_test.shape[1])
        return X_test, y_test

    def evaluate_cnn(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate CNN+BiLSTM+Attention model."""
        try:
            import tensorflow as tf
            model_path = self._config.project_root / "models" / "cnn_bilstm_attention" / "model.h5"
            if not model_path.exists():
                return {"error": "Model file not found"}

            model = tf.keras.models.load_model(str(model_path))
            X_3d = X_test[:, :, np.newaxis]

            t0 = time.perf_counter()
            proba = model.predict(X_3d, verbose=0, batch_size=512)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            y_pred = np.argmax(proba, axis=1)
            return self._compute_metrics(y_test, y_pred, proba, "CNN+BiLSTM+Attention", elapsed_ms)
        except Exception as exc:
            return {"error": str(exc)}

    def evaluate_xgboost(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate XGBoost model."""
        try:
            import joblib
            model_path = self._config.project_root / "models" / "xgboost" / "model.pkl"
            if not model_path.exists():
                return {"error": "Model file not found"}

            model = joblib.load(str(model_path))
            t0 = time.perf_counter()
            proba = model.predict_proba(X_test)
            elapsed_ms = (time.perf_counter() - t0) * 1000

            y_pred = np.argmax(proba, axis=1)
            return self._compute_metrics(y_test, y_pred, proba, "XGBoost", elapsed_ms)
        except Exception as exc:
            return {"error": str(exc)}

    def evaluate_isolation_forest(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Evaluate IsolationForest anomaly detector."""
        try:
            import joblib
            model_path = self._config.project_root / "models" / "isolation_forest" / "model.pkl"
            if not model_path.exists():
                return {"error": "Model file not found"}

            model = joblib.load(str(model_path))
            # -1 = anomaly, 1 = normal
            raw_preds = model.predict(X_test)

            # Convert to binary: 0=benign, 1=anomaly
            y_pred_binary = (raw_preds == -1).astype(int)
            y_true_binary = (y_test != 0).astype(int)

            from sklearn.metrics import classification_report, roc_auc_score
            report = classification_report(
                y_true_binary, y_pred_binary,
                target_names=["benign", "anomaly"], output_dict=True
            )
            try:
                auc = roc_auc_score(y_true_binary, y_pred_binary)
            except Exception:
                auc = 0.0

            return {
                "model": "IsolationForest",
                "task": "binary_anomaly",
                "classification_report": report,
                "roc_auc": round(auc, 4),
            }
        except Exception as exc:
            return {"error": str(exc)}

    def _compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        proba: np.ndarray,
        model_name: str,
        inference_ms: float,
    ) -> dict:
        """Compute full classification metrics."""
        from sklearn.metrics import (
            accuracy_score, classification_report,
            confusion_matrix, roc_auc_score,
        )

        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(
            y_true, y_pred,
            target_names=ATTACK_CLASSES,
            output_dict=True,
            zero_division=0,
        )
        cm = confusion_matrix(y_true, y_pred).tolist()

        try:
            auc = roc_auc_score(y_true, proba, multi_class="ovr", average="macro")
        except Exception:
            auc = 0.0

        return {
            "model": model_name,
            "accuracy": round(accuracy, 4),
            "macro_f1": round(report.get("macro avg", {}).get("f1-score", 0), 4),
            "weighted_f1": round(report.get("weighted avg", {}).get("f1-score", 0), 4),
            "roc_auc_macro_ovr": round(auc, 4),
            "inference_ms_total": round(inference_ms, 2),
            "inference_ms_per_sample": round(inference_ms / max(len(y_true), 1), 4),
            "classification_report": report,
            "confusion_matrix": cm,
            "n_samples": len(y_true),
        }

    def run_full_evaluation(self) -> dict:
        """Run evaluation on all available models and save report."""
        log.info("Starting full model evaluation...")
        X_test, y_test = self.load_test_data()

        report = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "n_test_samples": len(X_test),
            "n_features": X_test.shape[1],
            "classes": ATTACK_CLASSES,
            "class_distribution": {
                ATTACK_CLASSES[c]: int(np.sum(y_test == c))
                for c in range(len(ATTACK_CLASSES))
                if np.sum(y_test == c) > 0
            },
            "models": {},
        }

        log.info("Evaluating CNN+BiLSTM+Attention...")
        report["models"]["cnn_bilstm"] = self.evaluate_cnn(X_test, y_test)

        log.info("Evaluating XGBoost...")
        report["models"]["xgboost"] = self.evaluate_xgboost(X_test, y_test)

        log.info("Evaluating IsolationForest...")
        report["models"]["isolation_forest"] = self.evaluate_isolation_forest(X_test, y_test)

        self._results = report

        # Save report
        out_path = self._config.project_root / "artifacts" / "evaluation_report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        log.info("Evaluation complete", report_path=str(out_path))
        return report

    def print_summary(self) -> None:
        """Print a human-readable evaluation summary to console."""
        if not self._results:
            print("No results yet — run run_full_evaluation() first.")
            return

        print("\n" + "=" * 60)
        print("  SENTINEL-AI — Model Evaluation Summary")
        print("=" * 60)
        print(f"  Test samples : {self._results['n_test_samples']:,}")
        print(f"  Features     : {self._results['n_features']}")
        print(f"  Generated at : {self._results['generated_at']}\n")

        for model_key, result in self._results["models"].items():
            if "error" in result:
                print(f"  [{model_key}] ERROR: {result['error']}")
                continue
            print(f"  [{result.get('model', model_key)}]")
            print(f"    Accuracy      : {result.get('accuracy', 0):.4f}")
            print(f"    Macro F1      : {result.get('macro_f1', 0):.4f}")
            print(f"    ROC-AUC (OvR) : {result.get('roc_auc_macro_ovr', 0):.4f}")
            ms = result.get("inference_ms_per_sample", 0)
            print(f"    Latency/sample: {ms:.4f} ms")
            print()


if __name__ == "__main__":
    from sentinel_ai.core.config import load_config
    from sentinel_ai.core.logger import setup_logging

    config = load_config()
    setup_logging(config)

    evaluator = ModelEvaluator(config)
    try:
        evaluator.run_full_evaluation()
        evaluator.print_summary()
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
