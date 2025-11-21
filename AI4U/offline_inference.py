class NlpInfer:
    def __init__(self):
        self.labels = ['health', 'education', 'finance', 'safety', 'other']

    def predict(self, text, lang='en'):
        t = text.lower()
        if 'clinic' in t or 'doctor' in t:
            cat = 'health'; score = 0.95
        elif 'school' in t or 'exam' in t:
            cat = 'education'; score = 0.92
        elif 'loan' in t or 'bank' in t:
            cat = 'finance'; score = 0.90
        else:
            cat = 'other'; score = 0.6

        guidance = {
            'health': "Visit nearest clinic or ask for symptoms.",
            'education': "Contact school or ask for study guidance.",
            'finance': "Ask for loan eligibility or schemes.",
            'other': "Please provide more details."
        }

        return {
            "category": cat,
            "confidence": score,
            "lang": lang,
            "answer": guidance.get(cat)
        }

class SpeechInfer:
    def recognize(self, speech_text):
        return {
            "transcript": speech_text,
            "intent": "query",
            "confidence": 0.9
        }
