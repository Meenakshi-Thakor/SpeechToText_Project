# # stt.py


import speech_recognition as sr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---- Formality detection training data ----
SAMPLE_TEXTS = [
    ("Good morning Sir, I would like to request your approval.", "Formal"),
    ("Dear Madam, Please find the report attached.", "Formal"),
    ("I hope this message finds you well.", "Formal"),
    ("Can u pls send the file?", "Informal"),
    ("Hey! wassup bro, send that file rn :)", "Informal"),
    ("Thanks a lot! That was awesome!", "Informal"),
    ("Please accept my sincere apologies.", "Formal"),
    ("Sorry, my bad. Won't happen again lol", "Informal"),
]

# Train classifier
texts = [t for t, label in SAMPLE_TEXTS]
labels = [1 if label == "Formal" else 0 for t, label in SAMPLE_TEXTS]
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
clf = LogisticRegression(max_iter=1000).fit(X, labels)


def detect_formality(text):
    X_test = vectorizer.transform([text])
    prob = clf.predict_proba(X_test)[0]
    label = "Formal" if prob[1] > 0.5 else "Informal"
    return label, prob


def speech_to_text(audio_file):
    """Return transcription + formality result as string"""
    r = sr.Recognizer()
    try:
        with sr.AudioFile(audio_file) as source:
            audio = r.record(source)

        text = r.recognize_google(audio)
        label, prob = detect_formality(text)

        # Return clean formatted string
        result = (
            f"🎤 Transcription: {text}\n\n"
            f"📌 Formality: {label}\n"
            f"✅ Probability → Formal: {prob[1]:.2f}, Informal: {prob[0]:.2f}"
        )
        return result

    except sr.UnknownValueError:
        return "❌ Could not understand the audio"
    except sr.RequestError:
        return "❌ API unavailable or quota exceeded"
