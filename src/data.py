"""Dataset discovery, image loading, and feature extraction."""

from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

DEFAULT_CLASSES = [
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___healthy",
]

DEFAULT_LOCAL_DATA_DIR = Path("data/plantvillage_dataset/color")

KAGGLE_DATASET_SLUG = "abdallahalidev/plantvillage-dataset"

DATASET_HELP = (
    "Could not find the PlantVillage dataset.\n"
    "See the 'Dataset Setup' section of README.md for two options:\n"
    "  (A) Kaggle API:    https://www.kaggle.com/settings  -> API -> Create New Token\n"
    "  (B) Manual:        download from "
    "https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset\n"
    "                     and unzip into ./data/plantvillage_dataset/"
)


def _has_expected_classes(directory: Path, classes: list[str]) -> bool:
    if not directory.exists() or not directory.is_dir():
        return False
    available = {p.name for p in directory.iterdir() if p.is_dir()}
    return all(c in available for c in classes)


def resolve_data_dir(
    local_data_dir: Path | str | None = None,
    classes: list[str] | None = None,
) -> Path:
    """Find the dataset locally, otherwise download via kagglehub.

    Order:
      1. Use `local_data_dir` if it contains the expected class folders.
      2. Fall back to `kagglehub.dataset_download(...)`. Credentials are read
         from ~/.kaggle/kaggle.json by kagglehub — we never touch env vars.
      3. Raise FileNotFoundError with setup instructions.
    """
    classes = classes or DEFAULT_CLASSES
    local_dir = Path(local_data_dir) if local_data_dir else DEFAULT_LOCAL_DATA_DIR

    if _has_expected_classes(local_dir, classes):
        return local_dir

    try:
        import kagglehub
    except ImportError as e:
        raise FileNotFoundError(DATASET_HELP) from e

    try:
        download_path = kagglehub.dataset_download(KAGGLE_DATASET_SLUG)
    except Exception as e:
        raise FileNotFoundError(DATASET_HELP) from e

    for root, dirs, _ in os.walk(download_path):
        if "color" in dirs:
            candidate = Path(root) / "color"
            if _has_expected_classes(candidate, classes):
                return candidate

    raise FileNotFoundError(DATASET_HELP)


def list_class_folders(data_dir: Path, requested: list[str] | None = None) -> list[str]:
    """Return the subset of `requested` classes that exist in `data_dir`."""
    available = sorted(p.name for p in data_dir.iterdir() if p.is_dir())
    requested = requested or DEFAULT_CLASSES
    selected = [c for c in requested if c in available]
    if not selected:
        selected = available[:5]
    return selected


def image_to_features(image_path: Path, img_size: int = 32) -> np.ndarray:
    """Build a 3,075-D feature vector: 32x32x3 flattened pixels + mean RGB."""
    img = Image.open(image_path).convert("RGB").resize((img_size, img_size))
    arr = np.array(img) / 255.0
    return np.concatenate([arr.flatten(), arr.mean(axis=(0, 1))])


def load_classical_dataset(
    data_dir: Path,
    classes: list[str],
    max_per_class: int = 400,
    img_size: int = 32,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[Path], list[str]]:
    """Load images as classical feature vectors. Returns X, y, image_paths, label_names."""
    rng = random.Random(seed)

    X: list[np.ndarray] = []
    y: list[int] = []
    image_paths: list[Path] = []
    label_names: list[str] = []

    for label_index, class_name in enumerate(classes):
        class_folder = data_dir / class_name
        files: list[Path] = []
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
            files.extend(class_folder.glob(ext))
        rng.shuffle(files)
        files = files[:max_per_class]
        print(f"{class_name}: using {len(files)} images")

        for image_path in files:
            try:
                X.append(image_to_features(image_path, img_size=img_size))
                y.append(label_index)
                image_paths.append(image_path)
            except Exception as e:
                print(f"Skipping {image_path}: {e}")

        label_names.append(class_name)

    return np.array(X), np.array(y), image_paths, label_names


def load_cnn_tensors(
    image_paths: list[Path],
    y: np.ndarray,
    img_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the same images as float32 tensors of shape (N, img_size, img_size, 3)."""
    import tensorflow as tf

    imgs = []
    for path in image_paths:
        raw = tf.io.read_file(str(path))
        img = tf.image.decode_image(raw, channels=3, expand_animations=False)
        img = tf.image.resize(img, [img_size, img_size])
        img = tf.cast(img, tf.float32) / 255.0
        imgs.append(img.numpy())
    return np.array(imgs), np.array(y)


def split_and_scale(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.25,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def split_cnn(
    X_cnn: np.ndarray,
    y_cnn: np.ndarray,
    test_size: float = 0.25,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return train_test_split(
        X_cnn, y_cnn, test_size=test_size, random_state=seed, stratify=y_cnn
    )
