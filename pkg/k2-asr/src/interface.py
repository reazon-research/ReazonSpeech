import numpy as np
from dataclasses import dataclass

@dataclass
class AudioData:
    """Container for audio waveform"""
    waveform: np.float32
    samplerate: int

@dataclass
class Token:
    """A subword with timestamp"""
    # Currently Subword only has a single-point timestamp.
    # Theoretically, we should be able to compute time ranges.
    seconds: float
    token: str

@dataclass
class TranscribeResult:
    text: str
    tokens: list[Token]

@dataclass
class TranscribeConfig:
    verbose: bool = True
