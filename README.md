# 📰 Fake News Detector

A machine learning system that classifies news articles as **Real**, **Fake**, or **Uncertain** using TF-IDF vectorization. Automatically trains 3 models (Logistic Regression, Naive Bayes, SVM) and picks the best performer. Features a beautiful Flask web interface for real-time predictions with confidence scores and explainability.

**Tech Stack:** Python · Pandas · Scikit-learn · NLP · TF-IDF · Flask

---
## 🌐 Live Demo
🚀 Try the app here: https://fake-news-nevp.onrender.com/

## ✨ Key Features

✅ **3 Models Trained** — Logistic Regression, Naive Bayes, SVM (LinearSVC)  
✅ **Auto-Selects Best** — Automatically picks highest-accuracy model  
✅ **97.15% Accuracy** — SVM achieves best performance  
✅ **Uncertainty Detection** — Handles short texts and low-confidence predictions  
✅ **Confidence-Based Scoring** — Probability differences and thresholds  
✅ **Explainability** — Shows top discriminative words for fake vs. real  
✅ **Balanced Learning** — Handles class imbalance with class weighting  
✅ **Data Augmentation** — Trains on full text AND headlines separately  
✅ **Beautiful UI** — Dark-themed Flask interface  
✅ **Fast Training** — ~2-3 minutes for full dataset

---

## 🗂️ Project Structure

```
fake_news_detector/
├── model.py              # Core ML pipeline (3 models + auto-selection)
├── preprocessing.py      # Text cleaning, feature extraction, dataset loading
├── evaluate.py           # Model evaluation utilities & confusion matrix
├── app.py                # Flask web server with interactive UI
├── requirements.txt      # Python dependencies
├── Fake.csv              # Fake news dataset
├── True.csv              # Real news dataset
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Setup
```bash
git clone https://github.com/yourusername/fake-news-detector.git
cd fake-news-detector
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python model.py
```
**Output:**
```
[*] Loading datasets...
Total -> 89796
Real: 42834 | Fake: 46962

[STEP] Preprocessing...
Train: 71836 | Test: 17960

[1/3] Training Logistic Regression...
  Accuracy: 96.51% | F1: 96.38%

[2/3] Training Naive Bayes...
  Accuracy: 93.89% | F1: 93.54%

[3/3] Training SVM (LinearSVC)...
  Accuracy: 97.15% | F1: 97.03%

[RESULTS] MODEL COMPARISON
============================================================
SVM                  | Accuracy:  97.15% | F1:  97.03%
Logistic Regression  | Accuracy:  96.51% | F1:  96.38%
Naive Bayes          | Accuracy:  93.89% | F1:  93.54%

[WINNER] Best Model: SVM
Accuracy : 97.15%
Precision: 96.65%
Recall   : 97.41%

[SAVED] Model saved (SVM)
```

### 3. Launch Web Interface
```bash
python app.py
```
Open **http://localhost:5000** in your browser.

---

## 🎯 How It Works

### Pipeline
```
Raw Text → Preprocessing → TF-IDF Vectorization → 3 Models → Best Model Selected → Prediction
```

### Text Preprocessing
```python
- Lowercase conversion
- URL removal (http/www patterns)
- Punctuation stripping
- Extra whitespace normalization
- Keeps numbers by default
```

### TF-IDF Configuration
```python
TfidfVectorizer(
    max_features=20000,      # Top 20k terms
    ngram_range=(1, 2),      # Unigrams + Bigrams
    stop_words="english",    # Remove common words
    min_df=2                 # Ignore rare terms
)
```

### Models Compared

| Model | Accuracy | F1-Score | Speed |
|-------|----------|----------|-------|
| **SVM** (Winner) | **97.15%** | **97.03%** | Fast |
| Logistic Regression | 96.51% | 96.38% | Very Fast |
| Naive Bayes | 93.89% | 93.54% | Very Fast |

---

## 📊 Model Performance

**Dataset:** 89,796 news articles (Fake.csv + True.csv + augmented headlines)
- Training set: 71,836 samples
- Test set: 17,960 samples

**Selected Model: SVM (LinearSVC)**
- Accuracy: 97.15%
- Precision: 96.65%
- Recall: 97.41%
- F1-Score: 97.03%

---

## 💡 Prediction & Explainability

Each prediction returns:
- **label** — REAL, FAKE, or UNCERTAIN
- **confidence** — 0-100% model certainty
- **prob_real/prob_fake** — Class probabilities
- **confidence_gap** — Difference between predictions
- **top_fake_words** — Terms indicating fake news
- **top_real_words** — Terms indicating real news
- **model** — Which model made the prediction

**Example Response:**
```json
{
  "label": "FAKE",
  "confidence": 89.3,
  "prob_real": 10.7,
  "prob_fake": 89.3,
  "confidence_gap": 78.6,
  "top_fake_words": ["shocking", "leaked", "exposed", "secret", "coverup"],
  "top_real_words": ["study", "research", "published", "data", "evidence"],
  "model": "SVM"
}
```

### Uncertainty Logic
Text is marked **UNCERTAIN** if:
- Short text (< 6 words)
- Low confidence (< 65%)
- Small confidence gap (< 15%)

---

## 🔌 API Endpoints

### POST `/predict`
Classify a news article.
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Breaking news about the scandal..."}'
```

### GET `/metrics`
Get model performance metrics.
```bash
curl http://localhost:5000/metrics
```

### GET `/`
Interactive web interface.

---

## 📚 Using Your Own Dataset

```python
from preprocessing import load_csv_dataset
from model import FakeNewsDetector

# Load CSV (expects columns: 'text' and 'label')
df = load_csv_dataset("your_data.csv", text_col="text", label_col="label")

# Train with auto model selection
detector = FakeNewsDetector()
detector.train(df)
detector.save("custom_model.pkl")

# Load and predict
detector = FakeNewsDetector.load("custom_model.pkl")
result = detector.predict("Some news text...")
print(result)
```

**Recommended Datasets:**
- [Kaggle: Fake and Real News](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset)
- [LIAR Dataset](https://paperswithcode.com/dataset/liar)
- [FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet)

---

## 📦 Requirements

```
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
flask>=3.0.0
```

---

## 🔮 Future Improvements

- [ ] Add BERT/transformer-based models
- [ ] Deploy to cloud (Heroku, AWS, Google Cloud)
- [ ] Add browser extension for in-page detection
- [ ] Integrate fact-checking APIs
- [ ] Real-time model retraining pipeline
- [ ] Multi-language support
- [ ] User feedback loop for continuous improvement

---

## 📝 License

MIT License — Feel free to use in your projects!

---


