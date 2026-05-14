"""Evaluation metrics and plotting helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def weighted_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def print_classification_reports(
    results: dict,
    y_test_classical: np.ndarray,
    y_test_cnn: np.ndarray,
    label_names: list[str],
) -> None:
    short_labels = [n.replace("Tomato___", "") for n in label_names]
    for name, info in results.items():
        y_true = y_test_cnn if name == "Basic CNN" else y_test_classical
        print(f"\n===== {name} (acc={info['accuracy']:.4f}) =====")
        print(classification_report(
            y_true,
            info["predictions"],
            target_names=short_labels,
            digits=3,
            zero_division=0,
        ))


def find_best_model(results: dict) -> str:
    return max(results, key=lambda n: results[n]["accuracy"])


def _save(fig_path: Path) -> None:
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")


def plot_training_curves(history, out_path: Path | str) -> None:
    out_path = Path(out_path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["loss"], label="Train", linewidth=2, color="#2C5F2D")
    axes[0].plot(history.history["val_loss"], label="Validation", linewidth=2, color="#D97706")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("CNN Loss"); axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].plot(history.history["accuracy"], label="Train", linewidth=2, color="#2C5F2D")
    axes[1].plot(history.history["val_accuracy"], label="Validation", linewidth=2, color="#D97706")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    axes[1].set_title("CNN Accuracy"); axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    _save(out_path)
    plt.close(fig)


def plot_accuracy_comparison(results: dict, out_path: Path | str) -> None:
    out_path = Path(out_path)
    names = list(results.keys())
    accuracies = [results[n]["accuracy"] for n in names]

    fig = plt.figure(figsize=(10, 5))
    plt.bar(names, accuracies)
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.xticks(rotation=30, ha="right")
    plt.ylim(0, 1)
    plt.tight_layout()
    _save(out_path)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    title: str,
    out_path: Path | str,
) -> None:
    out_path = Path(out_path)
    cm = confusion_matrix(y_true, y_pred)
    short_labels = [n.replace("Tomato___", "") for n in labels]
    fig = plt.figure(figsize=(8, 8))
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=short_labels)
    display.plot(xticks_rotation=45, cmap="Greens")
    plt.title(title)
    plt.tight_layout()
    _save(out_path)
    plt.close(fig)
