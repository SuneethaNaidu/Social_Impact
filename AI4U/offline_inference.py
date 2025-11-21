"""
ai4u/offline_inference.py
Lightweight inference stubs for prototype.
In production: replace with TFLite models, on-device ASR (Vosk / Whisper small on edge), etc.
"""
from typing import Dict

class NlpInfer:
    def __init__(self):
        # Simple categories used by the prototype
        self.labels = ['health', 'education', 'finance', 'safety', 'other']

    def predict(self, text: str, lang: str = 'en') -> Dict:
        """
        Very simple heuristic-based fallback for demonstration.
        Replace with a lightweight transformer or TFLite model for real use.
        """
        t = text.lower()
        cat = 'other'
        score = 0.6
        if any(k in t for k in ['clinic', 'fever', 'doctor', 'hospital', 'vaccine']):
            cat = 'health'; score = 0.95
        elif any(k in t for k in ['school', 'teacher', 'exam', 'admission', 'class']):
            cat = 'education'; score = 0.92
        elif any(k in t for k in ['loan', 'bank', 'scheme', 'pension', 'subsidy']):
            cat = 'finance'; score = 0.9
        elif any(k in t for k in ['danger', 'flood', 'accident', 'help', 'police']):
            cat = 'safety'; score = 0.92
        answer = self._generate_guidance(cat)
        return {'category': cat, 'confidence': score, 'lang': lang, 'answer': answer}

    def _generate_guidance(self, category: str) -> str:
        guidance = {
            'health': 'If this is an emergency, seek the nearest health center. For common symptoms, rest and hydrate. For more, ask for local clinic details.',
            'education': 'Contact local school authorities or ask about scholarship schemes. We can also provide simple study tips.',
            'finance': 'Check eligibility for government schemes or nearby bank branches. Ask about documentation needed.',
            'safety': 'If immediate danger, call local emergency services. For disasters, follow evacuation instructions.',
            'other': 'Please provide more detail so we can assist you better.'
        }
        return guidance.get(category, 'Please provide more detail.')

class SpeechInfer:
    def __init__(self):
        pass

    def recognize(self, speech_text: str) -> Dict:
        """
        Prototype assumes speech_text is already transcribed (for offline demo).
        In production, use an on-device ASR model and intent classifier.
        """
        # simple mapping: we return the same transcript and a best-guess intent
        transcript = speech_text
        # very naive mapping
        intent = 'query'
        confidence = 0.9
        if 'help' in speech_text.lower() or 'emergency' in speech_text.lower():
            intent = 'emergency'
        return {'transcript': transcript, 'intent': intent, 'confidence': confidence}
