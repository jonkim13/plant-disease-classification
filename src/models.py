"""Model factory functions for the classical baselines and the basic CNN."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

SEED = 42


def build_knn():
    return KNeighborsClassifier(n_neighbors=5)


def build_logreg():
    return LogisticRegression(max_iter=1000, random_state=SEED)


def build_svm_linear():
    return SVC(kernel="linear", random_state=SEED)


def build_svm_rbf():
    return SVC(kernel="rbf", random_state=SEED)


def build_mlp():
    return MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=SEED)


def build_cnn(num_classes: int, img_size: int = 64):
    """Return a compiled keras Sequential CNN.

    CNN training is non-deterministic on GPU even with seeds set, so accuracy
    will fluctuate across runs by a couple of percent.
    """
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.Input(shape=(img_size, img_size, 3)),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def classical_factories() -> dict[str, callable]:
    """Mapping from display name -> factory function."""
    return {
        "KNN": build_knn,
        "Logistic Regression": build_logreg,
        "SVM Linear Kernel": build_svm_linear,
        "SVM RBF Kernel": build_svm_rbf,
        "Simple Neural Network": build_mlp,
    }
