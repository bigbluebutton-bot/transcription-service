# data.py
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Tuple, Dict

import numpy as np

@dataclass
class Word:
    word: str
    start: float
    end: float
    probability: float

@dataclass
class TextSegment:
    text: str
    start: float
    end: float
    words: Optional[List[Word]] = None
    probability: Optional[float] = None # This is the probability of the word detected at this point in the audio. Not how likely the word is to be correct.

# enum
class Task(Enum):
    TRANSCRIBE = "transcribe"
    TRANSLATE = "translate"

@dataclass
class AudioData:
    raw_audio_data: bytes                                    # Current audio chunk from stream
    task: Task                                               # Task to perform on the audio data
    audio_buffer: Optional[bytes] = None                     # Buffer of n seconds of raw audio data
    audio_buffer_time: Optional[float] = None                # Time duration of the audio buffer
    audio_buffer_start_after: Optional[float] = None         # Time duration of the audio buffer start after the start of the audio stream
    audio_data: Optional[np.ndarray] = None                  # Audio data converted to mono waveform
    audio_data_sample_rate: Optional[int] = None             # Sample rate of the audio data after conversion
    vad_result: Optional[List[Dict[str, float | List[Tuple[float, float]]]]] = None      # Voice activity detection result
    # vad_audio_result: Optional[np.ndarray] = None      # Voice activity detection result
    language: Optional[Tuple[str, float]] = None             # Detected language of the audio data (language code, probability)
    transcribed_segments: Optional[List[TextSegment]] = None # Transcribed segments as text with timestamps
    # cleaned_words: Optional[List[str]] = None                # List of transcribed words. Halicunation removed.
    confirmed_words: Optional[List[Word]] = None              # List of confirmed words
    unconfirmed_words: Optional[List[Word]] = None            # List of unconfirmed words
    translations: Optional[Dict[str, str]] = None            # Translations to target languages
