from sklearn.feature_extraction.text import TfidfVectorizer
from src.preprocess import clean_text

def prepare_features(df):
    df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    df["clean_text"] = df["text"].apply(clean_text)

    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(df["clean_text"])

    return X, vectorizer
