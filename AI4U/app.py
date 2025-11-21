"""
ai4u/app.py
A simple Flask prototype exposing text and speech endpoints.
Replace rule-based logic with actual on-device models for production.
"""
from flask import Flask, request, jsonify
from offline_inference import NlpInfer, SpeechInfer

app = Flask(__name__)

nlp = NlpInfer()
speech = SpeechInfer()

@app.route('/api/v1/ask', methods=['POST'])
def ask():
    payload = request.json or {}
    text = payload.get('text', '')
    lang = payload.get('lang', 'en')
    if not text:
        return jsonify({'error':'text required'}), 400
    resp = nlp.predict(text, lang=lang)
    return jsonify(resp)

@app.route('/api/v1/speech-to-intent', methods=['POST'])
def speech_to_intent():
    payload = request.json or {}
    speech_text = payload.get('speech_text', '')
    if not speech_text:
        return jsonify({'error':'speech_text required for prototype'}), 400
    resp = speech.recognize(speech_text)
    return jsonify(resp)

if __name__ == '__main__':
    # use 0.0.0.0 to allow remote testing on a device if needed
    app.run(host='0.0.0.0', port=8000, debug=True)
