
"""
================================================================================
BHARAT AI-SOC: OFFLINE HINDI VOICE ASSISTANT v6.4 - SUBMISSION READY
================================================================================
Updates: Added recognition for 'Dhanyavad' and 'Shukriya' (Gratitude Intent).
Hardware: Raspberry Pi 4 + Grenaro Wireless Microphone
Optimization: Sub-second ML Latency | Buffer Flush Echo Cancellation
================================================================================
"""

import sys, json, subprocess, time, struct, os
import numpy as np
import pyaudio
from vosk import Model, KaldiRecognizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# --- MASTER CONFIGURATION ---
RATE, CHUNK = 16000, 1024
MODEL_PATH = "vosk-model-hi-0.22"
ENERGY_THRESHOLD = 600
VAD_SILENCE_FRAMES = 45  # Approx 1.5s - 2s wait time

def get_pi_temp():
    try:
        res = os.popen('vcgencmd measure_temp').readline()
        return res.replace("temp=","").replace("'C\n","") + " डिग्री सेल्सियस"
    except:
        return "उपलब्ध नहीं है"

# --- UPDATED DATABASE (Including Dhanyavad & Shukriya) ---
INTENT_CORPUS = {
    "PM": ["भारत के प्रधानमंत्री", "prime minister", "pm modi"],
    "KALAM": ["अब्दुल कलाम का जन्म", "kalam birthday", "kalam janm", "kalam ka pida"],
    "THANKS": ["धन्यवाद", "शुक्रिया", "thank you", "dhanyavad", "shukriya"], # Added Gratitude
    "STATES": ["भारत में कितने राज्य", "states in india", "rajya"],
    "SPACECRAFT": ["भारत का पहला अंतरिक्ष यान", "first spacecraft", "aryabhata"],
    "TIME": ["समय क्या है", "time kya hai", "waqt"],
    "DATE": ["तारीख बताओ", "date", "tarikh"],
    "NAME": ["आपका नाम क्या है", "naam", "who are you"],
    "CAPITAL": ["भारत की राजधानी", "capital of india", "delhi"],
    "TEMP": ["सिस्टम का तापमान", "cpu temperature", "pi temp"],
    "STATUS": ["सिस्टम ठीक है", "system status", "health"],
    "NOBEL": ["पहला नोबेल पुरस्कार", "first nobel prize", "tagore"],
    "LITERACY": ["साक्षरता दर", "literacy rate"],
    "MAHABHARAT": ["महाभारत किसने लिखी", "ved vyas"],
    "IIST": ["आईआईएसटी की विशेषता", "unique feature of iist"],
    "RAINBOW": ["इंद्रधनुष में कितने रंग", "colors in rainbow"],
    "HELP": ["तुम क्या कर सकते हो", "help", "commands"],
    "LIGHT_OFF": ["लाइट बंद करो", "light off"],
    "LIGHT_ON": ["लाइट चालू करो", "light on"],
    "FAN_OFF": ["पंखा बंद करो", "fan off"],
    "FAN_STATUS": ["पंखा ठीक है", "fan status"],
    "MUTE": ["चुप हो जाओ", "be quiet"],
    "MORNING": ["सुप्रभात", "good morning"],
    "GOODBYE": ["अलविदा भाई", "bye bye", "tata"],
    "BIRD": ["भारत का राष्ट्रीय पक्षी", "national bird", "mor"]
}

RESPONSES = {
    "PM": lambda: "भारत के प्रधानमंत्री श्री नरेंद्र मोदी हैं।",
    "KALAM": lambda: "डॉक्टर अब्दुल कलाम का जन्म पंद्रह अक्टूबर उन्नीस सौ इकतीस को हुआ था।",
    "THANKS": lambda: "आपका स्वागत है! मुझे आपकी मदद करके खुशी हुई।", # Response for Gratitude
    "STATES": lambda: "भारत में अट्ठाईस राज्य और आठ केंद्र शासित प्रदेश हैं।",
    "SPACECRAFT": lambda: "भारत के पहले अंतरिक्ष यान का नाम आर्यभट्ट है।",
    "TIME": lambda: f"अभी {time.strftime('%H:%M')} हुए हैं।",
    "DATE": lambda: f"आज की तारीख {time.strftime('%d-%m-%Y')} है।",
    "NAME": lambda: "मेरा नाम भारत वॉयस असिस्टेंट है।",
    "CAPITAL": lambda: "भारत की राजधानी नई दिल्ली है।",
    "TEMP": lambda: f"सिस्टम तापमान {get_pi_temp()} है।",
    "STATUS": lambda: "जी हाँ, सिस्टम पूरी तरह ठीक है और सभी मॉड्यूल एक्टिव हैं।",
    "NOBEL": lambda: "पहला भारतीय नोबेल पुरस्कार रवींद्रनाथ टैगोर को मिला था।",
    "MAHABHARAT": lambda: "महाभारत महर्षि वेदव्यास जी द्वारा लिखी गई थी।",
    "IIST": lambda: "आईआईएसटी एशिया का पहला अंतरिक्ष विश्वविद्यालय है।",
    "RAINBOW": lambda: "इंद्रधनुष में सात रंग होते हैं।",
    "HELP": lambda: "मैं सामान्य ज्ञान, समय, तारीख और सिस्टम की स्थिति बता सकता हूँ।",
    "LIGHT_OFF": lambda: "ठीक है, लाइट बंद कर दी गई है।",
    "LIGHT_ON": lambda: "ठीक है, लाइट चालू कर दी गई है।",
    "FAN_OFF": lambda: "जी, पंखा बंद कर दिया गया है।",
    "FAN_STATUS": lambda: "जी हाँ, पंखा बिल्कुल ठीक काम कर रहा है।",
    "MORNING": lambda: "सुप्रभात! आपका दिन बहुत अच्छा हो।",
    "GOODBYE": lambda: "अलविदा भाई! अपना ख्याल रखियेगा।",
    "BIRD": lambda: "भारत का राष्ट्रीय पक्षी मोर है।",
    "UNKNOWN": lambda: "क्षमा करें, मुझे समझ नहीं आया।"
}

class MLIntentEngine:
    def __init__(self, corpus):
        self.lbl, self.sent = [], []
        for i, ex in corpus.items():
            for s in ex: self.sent.append(s); self.lbl.append(i)
        self.vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        self.mat = self.vec.fit_transform(self.sent)
    def predict(self, text):
        v = self.vec.transform([text.lower()])
        s = cosine_similarity(v, self.mat)[0]
        return self.lbl[np.argmax(s)], np.max(s)

class SimpleVAD:
    def __init__(self): self.nl, self.sc, self.slc, self.is_sp = 0, 0, 0, False
    def rms(self, pcm): return float(np.sqrt(np.mean(np.array(struct.unpack(f"{len(pcm)//2}h", pcm), dtype=np.float32) ** 2)))
    def process(self, pcm):
        r = self.rms(pcm)
        sp = r > max(self.nl * 1.6, ENERGY_THRESHOLD)
        if sp: self.sc += 1; self.slc = 0
        else: self.slc += 1; self.sc = 0
        if self.sc >= 3: self.is_sp = True
        if self.slc >= VAD_SILENCE_FRAMES: self.is_sp = False
        return self.is_sp, r

def speak(text):
    print(f"\n  [Assistant]: {text}\n")
    subprocess.run(["espeak-ng", "-v", "hi", "-s", "130", "-a", "180", text], stdout=subprocess.DEVNULL)

def run_assistant():
    ml = MLIntentEngine(INTENT_CORPUS)
    model = Model(MODEL_PATH)
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    vad = SimpleVAD()
    for _ in range(5): vad.nl += vad.rms(stream.read(CHUNK, exception_on_overflow=False))
    vad.nl /= 5
    speak("भारत वॉयस असिस्टेंट तैयार है।")
    rec = KaldiRecognizer(model, RATE)
    buffer_text, last_sp_time = "", 0
    try:
        while True:
            raw = stream.read(CHUNK, exception_on_overflow=False)
            is_sp, r = vad.process(raw)
            if is_sp:
                if rec.AcceptWaveform(raw):
                    res = json.loads(rec.Result())
                    buffer_text += " " + res.get("text","")
                else:
                    p = json.loads(rec.PartialResult()).get("partial","")
                    if p: print(f"  Hearing: {p} ", end="\r")
                last_sp_time = time.time()
            if not is_sp and buffer_text.strip():
                query = (buffer_text + " " + json.loads(rec.FinalResult()).get("text","")).strip()
                if len(query) > 2:
                    start_brain = time.time()
                    intent, score = ml.predict(query)
                    latency = time.time() - start_brain
                    print(f"\nUSER: {query} | Latency: {latency:.3f}s")
                    resp = RESPONSES.get(intent, RESPONSES["UNKNOWN"])()
                    speak(resp)
                    while stream.get_read_available() > 0: stream.read(CHUNK, exception_on_overflow=False)
                buffer_text, rec = "", KaldiRecognizer(model, RATE)
    except KeyboardInterrupt: pass
    finally: stream.close(); pa.terminate()

if __name__ == "__main__": run_assistant()
