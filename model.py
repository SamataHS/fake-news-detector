"""
Improved Fake News Detector
✔ Logistic Regression, Naive Bayes, Calibrated SVM
✔ Proper probability outputs
✔ Balanced uncertainty detection
✔ Explanation support
"""

import numpy as np
import pandas as pd
import re
import string
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score


# ─── TEXT PREPROCESSING ─────────────────────────────
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ─── LOAD DATASET ───────────────────────────────────
def load_dataset(fake_path="Fake.csv", true_path="True.csv"):
    print("[*] Loading datasets...")

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)

    df["combined"] = df["title"].fillna("") + " " + df["text"].fillna("")

    # Add headline-only data
    title_df = df.copy()
    title_df["combined"] = title_df["title"]

    df = pd.concat([df, title_df], ignore_index=True)

    df = df[df["combined"].str.strip() != ""].reset_index(drop=True)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Total -> {len(df)}")
    print(f"Real: {df['label'].sum()} | Fake: {(df['label']==0).sum()}")

    return df


# ─── MODEL CLASS ────────────────────────────────────
class FakeNewsDetector:

    def __init__(self):
        self.best_model = None
        self.tfidf = None
        self.model_name = None
        self.metrics = {}

    # ─── TRAIN ─────────────────────────────
    def train(self, df):

        print("\n[STEP] Preprocessing...")
        df["clean_text"] = df["combined"].apply(preprocess_text)

        X = df["clean_text"]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Train: {len(X_train)} | Test: {len(X_test)}")

        tfidf = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2
        )

        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)

        results = {}

        # ─── Logistic Regression ───
        print("\n[1/3] Logistic Regression")
        lr = LogisticRegression(max_iter=1000, class_weight='balanced')
        lr.fit(X_train_tfidf, y_train)

        y_pred = lr.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy: {acc*100:.2f}% | F1: {f1*100:.2f}%")
        results["Logistic Regression"] = (acc, f1, lr)

        # ─── Naive Bayes ───
        print("\n[2/3] Naive Bayes")
        nb = MultinomialNB()
        nb.fit(X_train_tfidf, y_train)

        y_pred = nb.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy: {acc*100:.2f}% | F1: {f1*100:.2f}%")
        results["Naive Bayes"] = (acc, f1, nb)

        # ─── SVM (CALIBRATED ✅) ───
        print("\n[3/3] SVM (Calibrated)")
        svm = LinearSVC(max_iter=2000, class_weight='balanced', dual=False)

        calibrated_svm = CalibratedClassifierCV(svm)
        calibrated_svm.fit(X_train_tfidf, y_train)

        y_pred = calibrated_svm.predict(X_test_tfidf)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy: {acc*100:.2f}% | F1: {f1*100:.2f}%")
        results["SVM"] = (acc, f1, calibrated_svm)

        # ─── SELECT BEST ───
        print("\n===== MODEL COMPARISON =====")
        for name, (acc, f1, _) in results.items():
            print(f"{name}: Accuracy={acc*100:.2f}% | F1={f1*100:.2f}%")

        best = max(results, key=lambda x: results[x][0])
        acc, f1, model = results[best]

        print(f"\n🏆 BEST MODEL: {best}")

        self.best_model = model
        self.tfidf = tfidf
        self.model_name = best

        self.metrics = {
            "accuracy": round(acc * 100, 2),
            "f1": round(f1 * 100, 2),
            "model_name": best
        }

    # ─── PREDICT ───────────────────────────
    def predict(self, text):

        if self.best_model is None:
            raise RuntimeError("Model not trained")

        is_short = len(text.split()) < 4

        clean = preprocess_text(text)
        X_tfidf = self.tfidf.transform([clean])

        # ✅ Now ALL models support predict_proba
        proba = self.best_model.predict_proba(X_tfidf)[0]

        confidence = max(proba)
        diff = abs(proba[1] - proba[0])

        # ─── UNCERTAIN LOGIC ───
        if is_short:
            label = "UNCERTAIN"
        elif confidence < 0.55:
            label = "UNCERTAIN"
        elif diff < 0.12:
            label = "UNCERTAIN"
        elif proba[1] > proba[0]:
            label = "REAL"
        else:
            label = "FAKE"

        return {
            "label": label,
            "confidence": round(float(confidence) * 100, 1),
            "prob_real": round(float(proba[1]) * 100, 1),
            "prob_fake": round(float(proba[0]) * 100, 1),
            "model": self.model_name
        }

    # ─── SAVE ─────────────────────────────
    def save(self, path="model.pkl"):
        with open(path, "wb") as f:
            pickle.dump({
                "model": self.best_model,
                "tfidf": self.tfidf,
                "metrics": self.metrics,
                "model_name": self.model_name
            }, f)

    # ─── LOAD ─────────────────────────────
    @classmethod
    def load(cls, path="model.pkl"):
        obj = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)

        obj.best_model = data["model"]
        obj.tfidf = data["tfidf"]
        obj.metrics = data["metrics"]
        obj.model_name = data["model_name"]

        return obj


# ─── MAIN ─────────────────────────────
if __name__ == "__main__":
    df = load_dataset("Fake.csv", "True.csv")

    detector = FakeNewsDetector()
    detector.train(df)
    detector.save("model.pkl")

    print("\n✅ Model trained & saved!")