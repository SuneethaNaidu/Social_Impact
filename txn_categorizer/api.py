"""
txn_categorizer/api.py
Simple Flask server that loads the saved model and returns predictions with confidence.
"""
from flask import Flask, request, jsonify
import joblib
import yaml
import os
import numpy as np

MODEL_PATH = os.environ.get('MODEL_PATH', 'models/tfidf_lr.joblib')
TAXONOMY_PATH = os.environ.get('TAXONOMY_PATH', 'taxonomy.yaml')

app = Flask(__name__)

# Lazy load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Train model first.")

model = joblib.load(MODEL_PATH)

taxonomy = {}
if os.path.exists(TAXONOMY_PATH):
    with open(TAXONOMY_PATH) as f:
        taxonomy = yaml.safe_load(f)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json or {}
    text = data.get('merchant', '')
    if not text:
        return jsonify({'error': 'merchant field required'}), 400
    # predict
    pred = model.predict([text])[0]
    probs = None
    conf = None
    try:
        probs = model.predict_proba([text])[0]
        classes = model.classes_.tolist()
        conf = float(np.max(probs))
        prob_map = dict(zip(classes, probs.tolist()))
    except Exception:
        prob_map = {}
        conf = None

    return jsonify({
        'merchant': text,
        'category': pred,
        'confidence': conf,
        'probabilities': prob_map
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)
