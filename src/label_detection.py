# Object of Interest (OOI) Detection Module
# This module will analyze text to identify the object of interest.

import re
from typing import Optional
#from src.voice_to_text import voice_to_text
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
def detect_object_label(text: str) -> Optional[str]:
    """
    Detects the object of interest (OOI) from the input text.
    Returns the label (item) being searched for, or None if not found.
    """
    # Example patterns: "find the cat", "show me a car", "where is the bottle"
    patterns = [
        r'find the ([\w\s]+)',
        r'show me (?:a|an|the)? ([\w\s]+)',
        r'where is (?:a|an|the)? ([\w\s]+)',
        r'look for (?:a|an|the)? ([\w\s]+)',
        r'search for (?:a|an|the)? ([\w\s]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    # Fallback: return the last noun-like word (very basic)
    words = text.split()
    if words:
        return words[-1]
    return None

