"""
app.py — Flask web interface for real-time fake news prediction
Run: python app.py
"""

from flask import Flask, request, jsonify, render_template_string
from model import FakeNewsDetector, load_dataset
import os

app = Flask(__name__)
detector = None


def load_or_train():
    global detector
    if os.path.exists("model.pkl"):
        detector = FakeNewsDetector.load("model.pkl")
        print("✅ Loaded pre-trained model.")
    else:
        raise Exception("❌ model.pkl not found! Cannot start app.")
        df = load_dataset(fake_path="fake.csv", true_path="true.csv")
        detector = FakeNewsDetector()
        detector.train(df)
        detector.save("model.pkl")


@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "No text provided"}), 400
    result = detector.predict(text)
    return jsonify(result)


@app.route("/metrics")
def metrics():
    info = detector.metrics.copy()
    info["model_name"] = detector.model_name
    return jsonify(info)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Fake News Detector</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; min-height: 100vh; display: flex; align-items: center; justify-content: center; padding: 2rem; }
  .card { background: #1e293b; border-radius: 16px; padding: 2.5rem; max-width: 680px; width: 100%; box-shadow: 0 25px 60px rgba(0,0,0,0.5); }
  h1 { font-size: 1.8rem; font-weight: 700; color: #f1f5f9; margin-bottom: 0.3rem; }
  p.sub { color: #94a3b8; font-size: 0.9rem; margin-bottom: 1.8rem; }

  textarea {
    width: 100%;
    height: 140px;
    background: #0f172a;
    border: 2px solid #334155;
    border-radius: 10px;
    color: #e2e8f0;
    font-size: 0.95rem;
    padding: 1rem;
    resize: vertical;
    outline: none;
  }

  button {
    margin-top: 1rem;
    width: 100%;
    padding: 0.85rem;
    background: #6366f1;
    color: white;
    border: none;
    border-radius: 10px;
    font-size: 1rem;
    font-weight: 600;
    cursor: pointer;
  }

  .result {
    margin-top: 1.5rem;
    padding: 1.2rem 1.5rem;
    border-radius: 12px;
    display: none;
  }

  .result.real { background: #052e16; border: 1.5px solid #16a34a; }
  .result.fake { background: #2d0707; border: 1.5px solid #dc2626; }

  /* 🟡 UNCERTAIN STYLE */
  .result.uncertain { background: #1e293b; border: 1.5px solid #facc15; }

  .verdict { font-size: 1.5rem; font-weight: 800; }
  .verdict.real { color: #4ade80; }
  .verdict.fake { color: #f87171; }
  .verdict.uncertain { color: #facc15; }

  .conf { font-size: 0.85rem; color: #94a3b8; margin-top: 0.3rem; }

  .bar-wrap { margin-top: 1rem; }
  .bar { height: 8px; border-radius: 99px; background: #1e3a5f; overflow: hidden; }
  .bar-fill { height: 100%; border-radius: 99px; transition: width 0.6s ease; }
  .bar-fill.real { background: #4ade80; }
  .bar-fill.fake { background: #f87171; }
</style>
</head>

<body>
<div class="card">
  <h1>🔍 Fake News Detector</h1>
  <p class="sub" id="modelInfo">TF-IDF + Best Model · NLP · Scikit-learn</p>

  <textarea id="newsText" placeholder="Paste news here..."></textarea>
  <button onclick="analyze()">Analyze Article</button>

  <div class="result" id="result">
    <div class="verdict" id="verdict"></div>
    <div class="conf" id="confidence"></div>

    <div class="bar-wrap">
      <div>Real: <span id="realPct"></span></div>
      <div class="bar"><div class="bar-fill real" id="realBar"></div></div>

      <div style="margin-top:6px">Fake: <span id="fakePct"></span></div>
      <div class="bar"><div class="bar-fill fake" id="fakeBar"></div></div>
    </div>
  </div>
</div>

<script>
async function loadModelInfo() {
  const res = await fetch('/metrics');
  const data = await res.json();
  const modelName = data.model_name || 'Best Model';
  document.getElementById('modelInfo').textContent = `TF-IDF + ${modelName} · NLP · Scikit-learn`;
}

loadModelInfo();

async function analyze() {
  const text = document.getElementById('newsText').value.trim();
  if (!text) return;

  const res = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });

  const data = await res.json();

  const resultDiv = document.getElementById('result');
  resultDiv.style.display = 'block';
  resultDiv.className = 'result ' + data.label.toLowerCase();

  const verdict = document.getElementById('verdict');
  verdict.className = 'verdict ' + data.label.toLowerCase();

  // 🔥 Handle labels properly
  if (data.label === 'REAL') {
    verdict.textContent = '✅ Likely Real News';
  } else if (data.label === 'FAKE') {
    verdict.textContent = '❌ Likely Fake News';
  } else {
    verdict.textContent = '⚠️ Uncertain Result';
  }

  document.getElementById('confidence').textContent =
    "Confidence: " + data.confidence + "% | Model: " + (data.model || "Unknown");

  // 🔥 Handle UNCERTAIN bars
  if (data.label === 'UNCERTAIN') {
    document.getElementById('realPct').textContent = "--";
    document.getElementById('fakePct').textContent = "--";
    document.getElementById('realBar').style.width = "50%";
    document.getElementById('fakeBar').style.width = "50%";
  } else {
    document.getElementById('realPct').textContent = data.prob_real + "%";
    document.getElementById('fakePct').textContent = data.prob_fake + "%";
    document.getElementById('realBar').style.width = data.prob_real + "%";
    document.getElementById('fakeBar').style.width = data.prob_fake + "%";
  }
}
</script>

</body>
</html>
"""


if __name__ == "__main__":
    load_or_train()
    print("\n🚀 Starting Flask server at http://127.0.0.1:5000")
    app.run(debug=True, host="0.0.0.0", port=5000)