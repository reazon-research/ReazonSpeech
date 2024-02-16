from dataclasses import dataclass
import numpy as np

@dataclass
class AudioData:
    """A container for audio waveform"""
    waveform: np.float32
    samplerate: int

@dataclass
class Segment:
    """A segment of transcription with timestamps"""
    start_seconds: float
    end_seconds: float
    text: str

@dataclass
class TranscribeResult:
    text: int
    segments: list

@dataclass
class TranscribeConfig:
    verbose: bool = True
