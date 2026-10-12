"""InboxGuard web app.   python app.py   ->  http://127.0.0.1:5000"""
import io
import json

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_from_directory

from inboxguard.config import EXPLAINER_PATH, MODEL_PATH, REPORTS
from inboxguard.explain import predict_many, predict_one, style_summary, top_signals


def load_artifacts():
    """Load saved model; train automatically if it is missing or incompatible."""
    try:
        return joblib.load(MODEL_PATH), joblib.load(EXPLAINER_PATH)
    except Exception:
        import train
        train.main()
        return joblib.load(MODEL_PATH), joblib.load(EXPLAINER_PATH)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
BUNDLE, EXPLAINER = load_artifacts()
MODEL, THRESHOLD, MODEL_NAME = BUNDLE["model"], BUNDLE["threshold"], BUNDLE["name"]


def metrics():
    return json.loads((REPORTS / "metrics.json").read_text())


@app.get("/")
def index():
    return render_template("index.html", model_name=MODEL_NAME, threshold=round(THRESHOLD, 3))


@app.post("/api/predict")
def api_predict():
    text = (request.get_json(silent=True) or {}).get("text", "").strip()
    if not text:
        return jsonify(error="Type or paste a message first."), 400
    prob, label = predict_one(MODEL, text, THRESHOLD)
    return jsonify(label=label, probability=round(prob, 4), threshold=round(THRESHOLD, 3),
                   model=MODEL_NAME, signals=top_signals(EXPLAINER, text), style=style_summary(text))


@app.post("/api/batch")
def api_batch():
    f = request.files.get("file")
    if f is None:
        return jsonify(error="Choose a CSV file."), 400
    try:
        df = pd.read_csv(io.BytesIO(f.read()), encoding="latin-1")
    except Exception as e:
        return jsonify(error=f"Could not read the CSV: {e}"), 400
    col = "text" if "text" in df.columns else None
    if col is None:
        return jsonify(error="The CSV needs a column named 'text'. Found: " + ", ".join(map(str, df.columns))), 400
    df = df.dropna(subset=[col]).head(5000)
    probs, labels = predict_many(MODEL, df[col].astype(str).tolist(), THRESHOLD)
    rows = [{"text": t, "label": l, "probability": round(float(p), 4)}
            for t, l, p in zip(df[col].astype(str), labels, probs)]
    return jsonify(rows=rows, total=len(rows), spam=sum(r["label"] == "spam" for r in rows))


@app.get("/api/metrics")
def api_metrics():
    return jsonify(metrics())


@app.get("/api/terms")
def api_terms():
    return jsonify(pd.read_csv(REPORTS / "top_terms.csv").to_dict(orient="list"))


@app.get("/reports/<path:name>")
def reports(name):
    return send_from_directory(REPORTS, name)


if __name__ == "__main__":
    app.run(debug=False, port=5000)
