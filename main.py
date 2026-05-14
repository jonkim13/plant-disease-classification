"""End-to-end orchestration: load data, train all models, evaluate, save plots."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np

from src.data import (
    DEFAULT_CLASSES,
    list_class_folders,
    load_classical_dataset,
    load_cnn_tensors,
    resolve_data_dir,
    split_and_scale,
    split_cnn,
)
from src.evaluate import (
    find_best_model,
    plot_accuracy_comparison,
    plot_confusion_matrix,
    plot_training_curves,
    print_classification_reports,
)
from src.train import train_classical_models, train_cnn

SEED = 42


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plant disease classification training pipeline.")
    p.add_argument("--data-dir", type=Path, default=None,
                   help="Local dataset path (containing the class folders). "
                        "If absent, kagglehub will download it.")
    p.add_argument("--output-dir", type=Path, default=Path("results"),
                   help="Where to save plots.")
    p.add_argument("--epochs", type=int, default=8, help="CNN training epochs.")
    p.add_argument("--batch-size", type=int, default=32, help="CNN batch size.")
    p.add_argument("--img-size", type=int, default=32,
                   help="Pixel size for the classical feature extractor.")
    p.add_argument("--cnn-img-size", type=int, default=64, help="Pixel size for the CNN.")
    p.add_argument("--max-per-class", type=int, default=400,
                   help="Cap on images sampled from each class.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(SEED)
    np.random.seed(SEED)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Resolving dataset...")
    data_dir = resolve_data_dir(local_data_dir=args.data_dir, classes=DEFAULT_CLASSES)
    print(f"Data directory: {data_dir}")

    classes = list_class_folders(data_dir, requested=DEFAULT_CLASSES)
    print("\nSelected classes:")
    for c in classes:
        print(f"  {c}")

    print("\nLoading classical feature vectors...")
    X, y, image_paths, label_names = load_classical_dataset(
        data_dir=data_dir,
        classes=classes,
        max_per_class=args.max_per_class,
        img_size=args.img_size,
        seed=SEED,
    )
    print(f"X: {X.shape}  y: {y.shape}")

    X_train, X_test, y_train, y_test, _scaler = split_and_scale(X, y, seed=SEED)
    print(f"Training set: {X_train.shape}  Test set: {X_test.shape}")

    print("\nLoading CNN tensors...")
    X_cnn, y_cnn = load_cnn_tensors(image_paths, y, img_size=args.cnn_img_size)
    X_cnn_train, X_cnn_test, y_cnn_train, y_cnn_test = split_cnn(X_cnn, y_cnn, seed=SEED)
    print(f"CNN training set: {X_cnn_train.shape}  CNN test set: {X_cnn_test.shape}")

    print("\n===== Classical models =====")
    results = train_classical_models(X_train, y_train, X_test, y_test)

    print("\n===== Basic CNN =====")
    cnn_result = train_cnn(
        X_cnn_train,
        y_cnn_train,
        X_cnn_test,
        y_cnn_test,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    # Original notebook forgot to merge this — make sure best-model logic sees the CNN.
    results["Basic CNN"] = cnn_result

    print("\n===== Per-model accuracy =====")
    for name, info in results.items():
        print(f"{name}: {info['accuracy']:.4f}")

    print("\n===== Classification reports =====")
    print_classification_reports(results, y_test, y_cnn_test, label_names)

    best_name = find_best_model(results)
    print(f"\nBest model: {best_name} ({results[best_name]['accuracy']:.4f})")

    print("\nSaving plots...")
    plot_training_curves(cnn_result["history"], args.output_dir / "cnn_training_curves.png")
    plot_accuracy_comparison(results, args.output_dir / "accuracy_comparison.png")

    y_true_for_cm = y_cnn_test if best_name == "Basic CNN" else y_test
    plot_confusion_matrix(
        y_true_for_cm,
        results[best_name]["predictions"],
        label_names,
        title=f"Confusion Matrix: {best_name}",
        out_path=args.output_dir / "confusion_matrix.png",
    )
    print(f"Plots written to {args.output_dir}/")


if __name__ == "__main__":
    main()
