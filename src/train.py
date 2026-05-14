"""Training routines for classical models and the basic CNN."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score

from .models import build_cnn, classical_factories


def train_classical_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> dict:
    """Fit each classical model and return a results dict.

    Returns {name: {"model", "predictions", "accuracy"}}.
    """
    results: dict[str, dict] = {}
    for name, factory in classical_factories().items():
        print(f"\nTraining {name}...")
        model = factory()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)
        results[name] = {
            "model": model,
            "predictions": predictions,
            "accuracy": accuracy,
        }
        print(f"{name} accuracy: {accuracy:.4f}")
    return results


def train_cnn(
    X_cnn_train: np.ndarray,
    y_cnn_train: np.ndarray,
    X_cnn_test: np.ndarray,
    y_cnn_test: np.ndarray,
    epochs: int = 8,
    batch_size: int = 32,
    val_split: float = 0.2,
) -> dict:
    """Train the basic CNN. Returns
    {"model", "predictions", "accuracy", "history", "test_loss"}.
    """
    num_classes = int(np.max(y_cnn_train)) + 1
    img_size = X_cnn_train.shape[1]
    model = build_cnn(num_classes=num_classes, img_size=img_size)
    model.summary()

    history = model.fit(
        X_cnn_train,
        y_cnn_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        verbose=1,
    )

    test_loss, test_acc = model.evaluate(X_cnn_test, y_cnn_test, verbose=0)
    probs = model.predict(X_cnn_test, verbose=0)
    predictions = np.argmax(probs, axis=1)

    print(f"\nCNN test loss: {test_loss:.4f}")
    print(f"CNN test accuracy: {test_acc:.4f}")

    return {
        "model": model,
        "predictions": predictions,
        "accuracy": float(test_acc),
        "history": history,
        "test_loss": float(test_loss),
    }
