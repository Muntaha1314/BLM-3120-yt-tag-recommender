from __future__ import annotations
from pathlib import Path
import pickle
from typing import Dict, List, Tuple

import numpy as np

from src.preprocess import clean_text


def _ensure_2d_probs(probs) -> np.ndarray:
    """
    OneVsRestClassifier.predict_proba returns (n_samples, n_classes) for most estimators.
    """
    if isinstance(probs, list):
        # list of (n_samples, 2) per class -> take positive class column
        return np.column_stack([p[:, 1] for p in probs])
    probs = np.asarray(probs)
    if probs.ndim != 2:
        raise ValueError(f"Unexpected prob shape: {probs.shape}")
    return probs


def parse_tags(tag_str) -> List[str]:
    """
    We normalize whitespace and remove empty tags/quotes.
    """
    if not isinstance(tag_str, str):
        return []
    s = tag_str.strip()
    if s == "" or s.lower() == "[none]":
        return []
    parts = [t.strip().strip('"').strip("'") for t in s.split("|")]
    return [t for t in parts if t]


def load_artifacts(models_dir: str | Path) -> Dict[str, object]:
    models_dir = Path(models_dir)
    with open(models_dir / "nb_model.pkl", "rb") as f:
        nb = pickle.load(f)
    with open(models_dir / "dt_model.pkl", "rb") as f:
        dt = pickle.load(f)
    with open(models_dir / "knn_model.pkl", "rb") as f:
        knn = pickle.load(f)
    with open(models_dir / "vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    with open(models_dir / "mlb.pkl", "rb") as f:
        mlb = pickle.load(f)

    artifacts = {"nb": nb, "dt": dt, "knn": knn, "vectorizer": vectorizer, "mlb": mlb}

    # test sets for evaluation
    x_test_path = models_dir / "X_test.pkl"
    y_test_path = models_dir / "Y_test.pkl"
    if x_test_path.exists() and y_test_path.exists():
        with open(x_test_path, "rb") as f:
            artifacts["X_test"] = pickle.load(f)
        with open(y_test_path, "rb") as f:
            artifacts["Y_test"] = pickle.load(f)

    return artifacts


def predict_top_k(
    model,
    vectorizer,
    mlb,
    title: str,
    description: str,
    k: int = 5,
) -> List[Tuple[str, float]]:
    """
    Returns list of (tag, probability) top-k for a single example.
    """
    text = f"{title} {description}".strip()
    clean = clean_text(text)
    X = vectorizer.transform([clean])

    probs = _ensure_2d_probs(model.predict_proba(X))[0]  # (n_classes,)
    top_idx = np.argsort(probs)[::-1][:k]
    tags = mlb.classes_[top_idx]
    return [(str(t), float(probs[i])) for t, i in zip(tags, top_idx)]


def top_k_indices_and_scores(model, X, k: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    For a matrix X, returns:
      - indices: (n_samples, k) int indices of top-k labels
      - scores:  (n_samples, k) float probabilities for those labels
    """
    probs = _ensure_2d_probs(model.predict_proba(X))  # (n, L)
    idx = np.argsort(probs, axis=1)[:, ::-1][:, :k]
    row = np.arange(probs.shape[0])[:, None]
    scores = probs[row, idx]
    return idx, scores


def precision_recall_at_k(y_true: np.ndarray, top_k_idx: np.ndarray) -> Tuple[float, float]:
    """
    Per-sample averaged Precision@k and Recall@k.
    y_true: (n_samples, n_labels) {0,1}
    top_k_idx: (n_samples, k) predicted label indices
    """
    y_true = np.asarray(y_true)
    k = top_k_idx.shape[1]

    pred = np.zeros_like(y_true, dtype=bool)
    rows = np.arange(y_true.shape[0])[:, None]
    pred[rows, top_k_idx] = True

    true = y_true.astype(bool)
    tp = (pred & true).sum(axis=1)  # per sample true positives
    pred_count = np.full((y_true.shape[0],), k, dtype=float)
    true_count = true.sum(axis=1).astype(float)

    precision = np.mean(np.divide(tp, pred_count, out=np.zeros_like(tp, dtype=float), where=pred_count > 0))
    recall = np.mean(np.divide(tp, true_count, out=np.zeros_like(tp, dtype=float), where=true_count > 0))
    return float(precision), float(recall)


def per_example_precision_recall_at_k(y_true_row: np.ndarray, top_k_idx_row: np.ndarray) -> Tuple[float, float]:
    """
    Single-example Precision@k and Recall@k.

    y_true_row: (n_labels,) {0,1}
    top_k_idx_row: (k,) indices of predicted labels
    """
    true_idx = set(np.where(np.asarray(y_true_row).astype(bool))[0].tolist())
    pred_idx = set(np.asarray(top_k_idx_row).tolist())
    if not pred_idx:
        return 0.0, 0.0
    tp = len(true_idx & pred_idx)
    precision = tp / len(pred_idx)
    recall = tp / len(true_idx) if true_idx else 0.0
    return float(precision), float(recall)


def mean_confidence_at_k(top_k_scores: np.ndarray) -> float:
    """
    Mean probability across the top-k predicted tags, averaged over samples.
    """
    return float(np.mean(top_k_scores))


def evaluate_models_on_test(artifacts: Dict[str, object], k: int = 5) -> List[Dict[str, float]]:
    """
    Returns a list of metric dicts (Model, Precision@k, Recall@k, MeanConfidence@k).
    Requires X_test and Y_test in artifacts (saved during training).
    """
    if "X_test" not in artifacts or "Y_test" not in artifacts:
        raise ValueError("X_test/Y_test not found. Re-train to save test artifacts.")

    X_test = artifacts["X_test"]
    Y_test = artifacts["Y_test"]

    results = []
    for name in ["nb", "dt", "knn"]:
        model = artifacts[name]
        idx, scores = top_k_indices_and_scores(model, X_test, k=k)
        p, r = precision_recall_at_k(Y_test, idx)
        c = mean_confidence_at_k(scores)
        results.append(
            {
                "Model": name.upper(),
                f"Precision@{k}": p,
                f"Recall@{k}": r,
                f"MeanConfidence@{k}": c,
            }
        )
    return results
