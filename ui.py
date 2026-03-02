from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.preprocess import clean_text
from src.evaluate import (
    load_artifacts,
    predict_top_k,
    evaluate_models_on_test,
    top_k_indices_and_scores,
    per_example_precision_recall_at_k,
    parse_tags,
)

# ---------- Paths ----------
ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "models"
DATA_PATH = ROOT / "data" / "videos.csv"

# ---------- Page setup ----------
st.set_page_config(page_title="YouTube Tag Prediction", layout="wide")
st.title("YouTube Video Tag Prediction (NB vs DT vs kNN)")

# ---------- Load artifacts ----------
@st.cache_resource
def _load_models():
    return load_artifacts(MODELS_DIR)

try:
    artifacts = _load_models()
except Exception as e:
    st.error(f"Couldn't load models from {MODELS_DIR}. Train models first.\n\nError: {e}")
    st.stop()

vectorizer = artifacts["vectorizer"]
mlb = artifacts["mlb"]

# ---------- Load dataset for random example ----------
@st.cache_data
def _load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")
    return pd.read_csv(DATA_PATH)

try:
    df_data = _load_dataset()
except Exception as e:
    df_data = None
    st.warning(f"Random-example mode disabled: {e}")

# ---------- Session state defaults ----------
if "title" not in st.session_state:
    st.session_state["title"] = ""
if "description" not in st.session_state:
    st.session_state["description"] = ""
if "true_tags_raw" not in st.session_state:
    st.session_state["true_tags_raw"] = None

# ---------- Random example section ----------
st.subheader("Quick test from dataset (random example)")
cols = st.columns([1, 2])
with cols[0]:
    pick_random = st.button("Pick random example")
with cols[1]:
    if df_data is not None:
        st.caption("Fills inputs from a random CSV row and enables per-example metrics using its ground-truth tags.")

if pick_random and df_data is not None:
    row = df_data.sample(1).iloc[0]
    st.session_state["title"] = str(row.get("title", "") or "")
    st.session_state["description"] = str(row.get("description", "") or "")
    st.session_state["true_tags_raw"] = str(row.get("tags", "") or "")

if st.session_state.get("true_tags_raw"):
    with st.expander("Ground-truth tags for the selected random example"):
        true_list = parse_tags(st.session_state["true_tags_raw"])
        st.write(true_list if true_list else ["(no tags / [none])"])

st.divider()

# ---------- Inputs ----------
st.subheader("Enter video info")
col1, col2 = st.columns(2)

with col1:
    title = st.text_input("Video name (title)", key="title")
with col2:
    description = st.text_area("Video description", key="description", height=120)

k = st.slider("Top-K tags to display", min_value=3, max_value=10, value=5)

# ---------- Predictions ----------
st.subheader("Predicted tags per model")
predict_clicked = st.button("Predict tags")

def _predict_block(model_key: str, label: str):
    preds = predict_top_k(artifacts[model_key], vectorizer, mlb, title, description, k=k)
    dfp = pd.DataFrame(preds, columns=["Tag", "Confidence"])
    dfp["Confidence"] = dfp["Confidence"].map(lambda x: round(float(x), 4))
    st.markdown(f"### {label}")
    st.table(dfp)

if predict_clicked:
    text = f"{title} {description}".strip()
    if not text:
        st.warning("Please enter at least a title or a description.")
    else:
        pred_cols = st.columns(3)
        with pred_cols[0]:
            _predict_block("nb", "Naive Bayes")
        with pred_cols[1]:
            _predict_block("dt", "Decision Tree")
        with pred_cols[2]:
            _predict_block("knn", "kNN (SVD + kNN)")

st.divider()

# ---------- Per-example dynamic comparison ----------
st.subheader("Per-example comparison (dynamic)")
st.caption("This comparison changes with each random dataset example because it uses that example’s ground-truth tags.")

compare_one = st.button("Compare models on this example (Precision/Recall/Confidence)")

if compare_one:
    text = f"{title} {description}".strip()
    if not text:
        st.warning("Please enter at least a title or a description.")
    elif not st.session_state.get("true_tags_raw"):
        st.warning("Pick a random example first so we have ground-truth tags for this input.")
    else:
        true_list_raw = parse_tags(st.session_state["true_tags_raw"])
        label_set = set(mlb.classes_)
        true_list = [t for t in true_list_raw if t in label_set]

        if len(true_list) == 0:
            st.warning(
                   "This random example has no ground-truth tags inside the model’s label set "
                    f"(top {len(mlb.classes_)} tags). Pick another random example."
            )
            st.stop()

        y_true = mlb.transform([true_list])  # now it's not all-zeros

        X_one = vectorizer.transform([clean_text(text)])

        rows = []
        for key, label in [("nb", "NB"), ("dt", "DT"), ("knn", "kNN")]:
            idx, scores = top_k_indices_and_scores(artifacts[key], X_one, k=k)
            p, r = per_example_precision_recall_at_k(y_true[0], idx[0])
            conf = float(np.mean(scores[0]))
            rows.append(
                {
                    "Model": label,
                    f"Precision@{k} (this example)": p,
                    f"Recall@{k} (this example)": r,
                    f"MeanConfidence@{k} (this example)": conf,
                }
            )

        df = pd.DataFrame(rows)
        for c in df.columns:
            if c != "Model":
                df[c] = df[c].map(lambda x: round(float(x), 4))
        st.dataframe(df, use_container_width=True)

st.divider()

# ---------- Global comparison ----------
st.subheader("Global evaluation (fixed after training)")
st.caption(
    "This computes Precision@K, Recall@K, and MeanConfidence@K on the held-out test set saved during training. "
    "These values are fixed unless models are retrained."
)

compare_global = st.button("Show global comparison on test set")

if compare_global:
    try:
        results = evaluate_models_on_test(artifacts, k=k)
        df = pd.DataFrame(results)
        for c in df.columns:
            if c != "Model":
                df[c] = df[c].map(lambda x: round(float(x), 4))
        st.dataframe(df, use_container_width=True)
    except Exception as e:
        st.error(f"Global comparison failed: {e}")
