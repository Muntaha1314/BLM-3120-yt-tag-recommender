import numpy as np

def top_k_predictions(model, X_test, k):
    probs = model.predict_proba(X_test)

    if isinstance(probs, list):
        probs = np.column_stack([p[:, 1] for p in probs])

    return np.argsort(probs, axis=1)[:, -k:]

def precision_at_k(y_true, top_k_preds):
    correct = 0
    total = 0

    for i in range(len(y_true)):
        true_tags = set(np.where(y_true[i] == 1)[0])
        pred_tags = set(top_k_preds[i])
        correct += len(true_tags & pred_tags)
        total += len(pred_tags)

    return correct / total if total > 0 else 0

def recall_at_k(y_true, top_k_preds):
    correct = 0
    total = 0

    for i in range(len(y_true)):
        true_tags = set(np.where(y_true[i] == 1)[0])
        pred_tags = set(top_k_preds[i])
        correct += len(true_tags & pred_tags)
        total += len(true_tags)

    return correct / total if total > 0 else 0
