from __future__ import annotations

from pathlib import Path
import pickle
from collections import Counter
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer

from src.preprocess import clean_text


def _project_root() -> Path:
    
    return Path(__file__).resolve().parents[1]


def _parse_tags(tag_str) -> List[str]:
    """
    Kaggle 'USvideos' tags column is pipe-separated, with possible '[none]'.
    We normalize whitespace and remove empty tags/quotes.
    """
    if not isinstance(tag_str, str):
        return []
    s = tag_str.strip()
    if s == "" or s.lower() == "[none]":
        return []
    parts = [t.strip().strip('"').strip("'") for t in s.split("|")]
    return [t for t in parts if t]


def _build_vectorizer() -> TfidfVectorizer:
   
    return TfidfVectorizer(
        max_features=20000,
        min_df=3,
        max_df=0.90,
        ngram_range=(1, 1),
    )


def train_all_models(
    top_k_tags: int = 300,
    test_size: float = 0.2,
    random_state: int = 42,
    knn_train_cap: int = 3000,
) -> Dict[str, object]:
    """
    Trains three multi-label models (NB, DT, kNN) and saves artifacts under project_root/models.

    Artifacts saved:
      - nb_model.pkl, dt_model.pkl, knn_model.pkl
      - vectorizer.pkl, mlb.pkl
      - X_test.pkl, Y_test.pkl
    """
    root = _project_root()
    data_path = root / "data" / "videos.csv"
    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)

    if not data_path.exists():
        raise FileNotFoundError(f"Could not find dataset at: {data_path}")

    print(f"Loading data: {data_path}")
    df = pd.read_csv(data_path)

    # Build text field
    df["title"] = df.get("title", "").fillna("").astype(str)
    df["description"] = df.get("description", "").fillna("").astype(str)
    df["text"] = (df["title"] + " " + df["description"]).map(clean_text)

    # Parse & clean tags
    if "tags" not in df.columns:
        raise ValueError("Expected a 'tags' column in videos.csv")

    df["tag_list"] = df["tags"].apply(_parse_tags)

    # Keep rows with at least 1 tag
    df = df[df["tag_list"].map(len) > 0].copy()
    if df.empty:
        raise ValueError("After cleaning, dataset has no rows with tags.")

    # Reduce to the most common tags (keeps label space manageable)
    tag_counts = Counter(t for tags in df["tag_list"] for t in tags)
    common_tags = [t for t, _ in tag_counts.most_common(top_k_tags)]
    common_set = set(common_tags)

    df["tag_list"] = df["tag_list"].apply(lambda tags: [t for t in tags if t in common_set])
    df = df[df["tag_list"].map(len) > 0].copy()

    # Train/test split based on text
    X_text_train, X_text_test, y_tags_train, y_tags_test = train_test_split(
        df["text"].values,
        df["tag_list"].values,
        test_size=test_size,
        random_state=random_state,
        shuffle=True,
    )

    # Vectorize
    vectorizer = _build_vectorizer()
    X_train = vectorizer.fit_transform(X_text_train)
    X_test = vectorizer.transform(X_text_test)

    # Binarize labels
    mlb = MultiLabelBinarizer(classes=common_tags)
    Y_train = mlb.fit_transform(y_tags_train)
    Y_test = mlb.transform(y_tags_test)

    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")
    print(f"Features: {X_train.shape[1]} | Labels: {Y_train.shape[1]} (top_k_tags={top_k_tags})")

    # ---------- Model 1: Naive Bayes ----------
    nb_model = OneVsRestClassifier(MultinomialNB())
    print("Training: Naive Bayes (OneVsRest)")
    nb_model.fit(X_train, Y_train)

    # ---------- Model 2: Decision Tree ----------
    dt_base = DecisionTreeClassifier(
        max_depth=20,
        min_samples_split=10,
        random_state=random_state,
    )
    dt_model = OneVsRestClassifier(dt_base)
    print("Training: Decision Tree (OneVsRest)")
    dt_model.fit(X_train, Y_train)

    # ---------- Model 3: kNN (REQUIRED) ----------
    # kNN is expensive on sparse high-dim TF-IDF. We compress with SVD first.
    knn_train_idx = np.arange(X_train.shape[0])
    if knn_train_cap is not None and X_train.shape[0] > knn_train_cap:
        rng = np.random.default_rng(random_state)
        knn_train_idx = rng.choice(X_train.shape[0], size=knn_train_cap, replace=False)

    X_train_knn = X_train[knn_train_idx]
    Y_train_knn = Y_train[knn_train_idx]

    svd = TruncatedSVD(n_components=200, random_state=random_state)
    norm = Normalizer(copy=False)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn_pipeline = make_pipeline(svd, norm, knn)
    knn_model = OneVsRestClassifier(knn_pipeline)

    print(f"Training: kNN (OneVsRest) on {X_train_knn.shape[0]} samples with SVD(200)")
    knn_model.fit(X_train_knn, Y_train_knn)

    # Save artifacts
    print(f"Saving artifacts to: {models_dir}")
    with open(models_dir / "nb_model.pkl", "wb") as f:
        pickle.dump(nb_model, f)
    with open(models_dir / "dt_model.pkl", "wb") as f:
        pickle.dump(dt_model, f)
    with open(models_dir / "knn_model.pkl", "wb") as f:
        pickle.dump(knn_model, f)
    with open(models_dir / "mlb.pkl", "wb") as f:
        pickle.dump(mlb, f)
    with open(models_dir / "vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # Save test split for UI comparison
    with open(models_dir / "X_test.pkl", "wb") as f:
        pickle.dump(X_test, f)
    with open(models_dir / "Y_test.pkl", "wb") as f:
        pickle.dump(Y_test, f)

    print("✅ Training complete.")
    return {
        "nb": nb_model,
        "dt": dt_model,
        "knn": knn_model,
        "vectorizer": vectorizer,
        "mlb": mlb,
        "X_test": X_test,
        "Y_test": Y_test,
    }
