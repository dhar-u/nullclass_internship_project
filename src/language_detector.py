# src/language_detector.py

from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException

# Ensure consistent results
DetectorFactory.seed = 0

def detect_language(text: str) -> str:
    """
    Detects the language of the input text.
    Returns the ISO 639-1 code of the language (e.g., 'en', 'fr', 'es', 'hi').
    """
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"
