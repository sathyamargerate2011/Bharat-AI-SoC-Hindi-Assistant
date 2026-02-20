# Bharat-AI-SoC-Hindi-Assistant
Offline Hindi Voice Assistant for Raspberry Pi 4 - Optimized for ARM CPU.
# 🇮🇳 Bharat AI-SoC: Optimized Offline Hindi Assistant
A low-latency, privacy-preserving voice assistant for the Raspberry Pi 4, built for the **Bharat AI-SoC Student Challenge**.

## 🚀 Technical Highlights
* **Optimized Intent Engine:** Uses TF-IDF with character-level N-grams. This allows the system to recognize colloquialisms (like "Pida" for birthday) and "Hinglish" commands even if the ASR makes minor spelling errors.
* **Echo Cancellation (Safe Flush):** Implemented a hardware buffer-drain loop that "eats" audio feedback while the assistant is speaking, preventing the echo-loop common in wireless mics.
* **ARM SoC Efficiency:** Achieving <10ms Brain Latency on the Cortex-A72 CPU without needing external accelerators or cloud APIs.

## 🧠 Software Architecture
1. **ASR:** Vosk-Kaldi (quantized for 8-bit integer math).
2. **NLP:** TF-IDF Vectorizer + Cosine Similarity (Custom Feature Extraction).
3. **TTS:** eSpeak-NG (optimized for low-latency Hindi synthesis).

## 📊 Performance
- **Brain Latency:** 0.003s - 0.008s
- **Wait Time:** 1.5s (Configurable VAD)
- **Database:** 26 high-accuracy Hindi/English commands.
