# Text to Speech Module
# This module will convert text output back to speech.
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
def text_to_speech(text, rate=150, volume=1.0, voice=None):
    import pyttsx3

    engine = pyttsx3.init()
    engine.setProperty('rate', rate)      # Speed of speech
    engine.setProperty('volume', volume)  # Volume (0.0 to 1.0)
    if voice:
        engine.setProperty('voice', voice)  # Voice id (see below for how to list voices)
    engine.say(text)
    engine.runAndWait()

    engine.stop()
    del engine

# Example: To list available voices, run this code:
def list_voices():
    import pyttsx3
    engine = pyttsx3.init()
    voices = engine.getProperty('voices')
    for idx, v in enumerate(voices):
        print(f"Voice {idx}: {v.id} ({v.name})")