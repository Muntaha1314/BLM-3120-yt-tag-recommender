import pickle
import numpy as np
from src.train_models import train_all_models
from src.evaluate import precision_at_k, recall_at_k, top_k_predictions

def show_sample_output(df, Y_test, topk, mlb):
    idx = 0
    true_ids = np.where(Y_test[idx] == 1)[0]
    pred_ids = topk[idx]

    print("\nSample Prediction")
    print("Title:", df.iloc[idx]["title"])
    print("True Tags:", list(mlb.classes_[true_ids]))
    print("Predicted Tags:", list(mlb.classes_[pred_ids]))

print("Training models...")
X_test, Y_test = train_all_models()

nb = pickle.load(open("models/nb_model.pkl", "rb"))
dt = pickle.load(open("models/dt_model.pkl", "rb"))
knn = pickle.load(open("models/knn_model.pkl", "rb"))
mlb = pickle.load(open("models/mlb.pkl", "rb"))
df = pickle.load(open("models/df_sample.pkl", "rb"))

models = {
    "Naive Bayes": nb,
    "Decision Tree": dt,
    "kNN": knn
}

print("\n--- Evaluation (k = 5) ---\n")

for name, model in models.items():
    print(f"Model: {name}")

    topk = top_k_predictions(model, X_test, 5)
    p = precision_at_k(Y_test, topk)
    r = recall_at_k(Y_test, topk)

    print(f"Precision@5: {p:.3f}")
    print(f"Recall@5: {r:.3f}")

    show_sample_output(df, Y_test, topk, mlb)
    print("=" * 40)
