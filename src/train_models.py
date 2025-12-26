import pandas as pd
import pickle
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from src.vectorize import prepare_features

def train_all_models():
    print("Training models...")

    df = pd.read_csv("data/videos.csv")

    # Prepare tags FIRST
    df["tag_list"] = df["tags"].fillna("").apply(lambda x: x.split("|"))

    # Limit tags
    all_tags = [t for tags in df["tag_list"] for t in tags]
    tag_counts = Counter(all_tags)

    TOP_K_TAGS = 300
    common_tags = set(tag for tag, _ in tag_counts.most_common(TOP_K_TAGS))

    df["tag_list"] = df["tag_list"].apply(
        lambda tags: [t for t in tags if t in common_tags]
    )

    df = df[df["tag_list"].map(len) > 0].reset_index(drop=True)

   
    X, vectorizer = prepare_features(df)

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["tag_list"])

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42
    )

    # Models
    nb = OneVsRestClassifier(MultinomialNB())
    dt = OneVsRestClassifier(DecisionTreeClassifier())
    knn = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=5))
    
    
    print("Training Naive Bayes...")
    nb.fit(X_train, Y_train)
    print("Naive Bayes done")
    
    print("Training Decision Tree...")
    dt.fit(X_train, Y_train)
    print("Decision Tree done")
    
    print("Training kNN...")
    knn.fit(X_train, Y_train)
    print("kNN done")

    # Save artifacts
    pickle.dump(nb, open("models/nb_model.pkl", "wb"))
    pickle.dump(dt, open("models/dt_model.pkl", "wb"))
    pickle.dump(knn, open("models/knn_model.pkl", "wb"))
    pickle.dump(mlb, open("models/mlb.pkl", "wb"))
    pickle.dump(df, open("models/df_sample.pkl", "wb"))

    return X_test, Y_test
