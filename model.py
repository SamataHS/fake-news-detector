"""
Improved Fake News Detector
✔ Trains 3 models: LogisticRegression, NaiveBayes, SVM
✔ Automatically selects best performer
✔ Handles short text
✔ Confidence + difference based uncertainty detection
✔ Balanced learning
✔ Explanation support
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report, f1_score
import re
import string
import pickle


# ─── Text Preprocessing ─────────────────────────────

def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)

    # KEEP numbers
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ─── Load Dataset ───────────────────────────────────

def load_dataset(fake_path="fake.csv", true_path="true.csv"):
    print("[*] Loading datasets...")

    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    fake_df["label"] = 0
    true_df["label"] = 1

    df = pd.concat([fake_df, true_df], ignore_index=True)

    # Full text
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


# ─── Model Class ────────────────────────────────────

class FakeNewsDetector:

    def __init__(self):
        self.best_model = None
        self.tfidf = None
        self.model_name = None
        self.metrics = {}

    # ─── TRAIN ALL MODELS ──────────────────
    def train(self, df):

        print("\n[STEP] Preprocessing...")
        df["clean_text"] = df["combined"].apply(preprocess_text)

        X = df["clean_text"]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        print(f"Train: {len(X_train)} | Test: {len(X_test)}")

        # Create TF-IDF vectorizer
        tfidf = TfidfVectorizer(
            max_features=20000,
            ngram_range=(1, 2),
            stop_words="english",
            min_df=2
        )
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)

        models = {}
        results = {}

        # 1 Logistic Regression
        print("\n[1/3] Training Logistic Regression...")
        lr = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
        lr.fit(X_train_tfidf, y_train)
        y_pred_lr = lr.predict(X_test_tfidf)
        acc_lr = accuracy_score(y_test, y_pred_lr)
        f1_lr = f1_score(y_test, y_pred_lr)
        results['Logistic Regression'] = {'accuracy': acc_lr, 'f1': f1_lr, 'model': lr, 'tfidf': tfidf}
        print(f"  Accuracy: {acc_lr*100:.2f}% | F1: {f1_lr*100:.2f}%")

        # 2 Naive Bayes
        print("\n[2/3] Training Naive Bayes...")
        nb = MultinomialNB()
        nb.fit(X_train_tfidf, y_train)
        y_pred_nb = nb.predict(X_test_tfidf)
        acc_nb = accuracy_score(y_test, y_pred_nb)
        f1_nb = f1_score(y_test, y_pred_nb)
        results['Naive Bayes'] = {'accuracy': acc_nb, 'f1': f1_nb, 'model': nb, 'tfidf': tfidf}
        print(f"  Accuracy: {acc_nb*100:.2f}% | F1: {f1_nb*100:.2f}%")

        # 3 SVM (LinearSVC)
        print("\n[3/3] Training SVM (LinearSVC)...")
        svm = LinearSVC(max_iter=2000, class_weight='balanced', random_state=42, dual=False)
        svm.fit(X_train_tfidf, y_train)
        y_pred_svm = svm.predict(X_test_tfidf)
        acc_svm = accuracy_score(y_test, y_pred_svm)
        f1_svm = f1_score(y_test, y_pred_svm)
        results['SVM'] = {'accuracy': acc_svm, 'f1': f1_svm, 'model': svm, 'tfidf': tfidf}
        print(f"  Accuracy: {acc_svm*100:.2f}% | F1: {f1_svm*100:.2f}%")

        # [BEST] SELECT BEST MODEL
        print("\n" + "="*60)
        print("[RESULTS] MODEL COMPARISON")
        print("="*60)
        for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
            print(f"{model_name:20} | Accuracy: {metrics['accuracy']*100:6.2f}% | F1: {metrics['f1']*100:6.2f}%")

        best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_result = results[best_model_name]

        print(f"\n[WINNER] Best Model: {best_model_name}")
        print("="*60)

        # Store best model
        self.best_model = best_result['model']
        self.tfidf = best_result['tfidf']
        self.model_name = best_model_name

        acc = best_result['accuracy']
        prec = precision_score(y_test, best_result['model'].predict(X_test_tfidf))
        rec = recall_score(y_test, best_result['model'].predict(X_test_tfidf))

        print(f"Accuracy : {acc*100:.2f}%")
        print(f"Precision: {prec*100:.2f}%")
        print(f"Recall   : {rec*100:.2f}%")
        print("\n", classification_report(y_test, best_result['model'].predict(X_test_tfidf)))

        self.metrics = {
            "accuracy": round(acc * 100, 2),
            "precision": round(prec * 100, 2),
            "recall": round(rec * 100, 2),
            "model_name": best_model_name
        }

    # ─── PREDICT ───────────────────────────
    def predict(self, text):

        if self.best_model is None:
            raise RuntimeError("Model not trained")

        is_short = len(text.split()) < 6

        clean = preprocess_text(text)
        X_tfidf = self.tfidf.transform([clean])

        # Get prediction
        proba = self.best_model.predict_proba(X_tfidf)[0]

        confidence = max(proba)
        diff = abs(proba[1] - proba[0])

        # ✅ Balanced uncertainty logic
        if is_short:
            label = "UNCERTAIN"
        elif confidence < 0.65:
            label = "UNCERTAIN"
        elif diff < 0.15:
            label = "UNCERTAIN"
        elif proba[1] > proba[0]:
            label = "REAL"
        else:
            label = "FAKE"

        # 🔍 Explanation - get top discriminative words
        feature_names = self.tfidf.get_feature_names_out()

        # For models with coef_ attribute (LogisticRegression, LinearSVC)
        if hasattr(self.best_model, 'coef_'):
            coefs = self.best_model.coef_[0]
            top_fake = [feature_names[i] for i in coefs.argsort()[:5]]
            top_real = [feature_names[i] for i in coefs.argsort()[-5:]]
        # For XGBoost and other models without coef_
        else:
            # Use feature importance as fallback
            if hasattr(self.best_model, 'feature_importances_'):
                importances = self.best_model.feature_importances_
                top_indices = importances.argsort()[-10:][::-1]
                top_fake = [feature_names[i] for i in top_indices[:5]]
                top_real = [feature_names[i] for i in top_indices[5:10]]
            else:
                top_fake = ["N/A"]
                top_real = ["N/A"]

        return {
            "label": label,
            "confidence": round(float(confidence) * 100, 1),
            "prob_real": round(float(proba[1]) * 100, 1),
            "prob_fake": round(float(proba[0]) * 100, 1),
            "confidence_gap": round(float(diff) * 100, 1),
            "top_fake_words": top_fake,
            "top_real_words": top_real,
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
        print(f"[SAVED] Model saved ({self.model_name})")

    # ─── LOAD ─────────────────────────────
    @classmethod
    def load(cls, path="model.pkl"):
        obj = cls()

        with open(path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            obj.best_model = data["model"]
            obj.tfidf = data.get("tfidf")
            obj.metrics = data.get("metrics", {})
            obj.model_name = data.get("model_name", "Unknown")
        else:
            obj.best_model = data

        return obj


    # Main block
if __name__ == "__main__":
    df = load_dataset(fake_path="Fake.csv", true_path="True.csv")
    detector = FakeNewsDetector()
    detector.train(df)
    detector.save("model.pkl")
    print("\n[OK] Training complete! Model saved as model.pkl")