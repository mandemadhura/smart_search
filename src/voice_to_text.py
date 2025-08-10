# Voice to Text Module
# This module will handle voice input and convert it to text.

import speech_recognition as sr
from googletrans import Translator
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def voice_to_text(language_code='en-US'):
    """
    Captures voice input from the microphone, converts it to text using Google Speech Recognition,
    and translates it to English if needed.
    Args:
        language_code (str): The language code for speech recognition (default is 'en-US').
    Returns:
        str: The recognized and translated text in English.
    """
    recognizer = sr.Recognizer()
    translator = Translator()
    with sr.Microphone() as source:
        print(f"Please speak now (language: {language_code})...")
        audio = recognizer.listen(source)
    try:
        text = recognizer.recognize_google(audio, language=language_code)
        print(f"You said: {text}")
        if language_code != 'en-US':
            translated = translator.translate(text, dest='en')
            print(f"Translated to English: {translated.text}")
            return translated.text
        return text
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio.")
        return None
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return None
