from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)
model = joblib.load("model.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    merchant = request.json.get("merchant")
    pred = model.predict([merchant])[0]
    return {"merchant": merchant, "category": pred}

app.run(port=8001)
