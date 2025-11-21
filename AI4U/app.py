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
        return jsonify({'error':'speech_text required'}), 400
    resp = speech.recognize(speech_text)
    return jsonify(resp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
