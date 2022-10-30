from dataclasses import dataclass

__all__ = "Caption", "Utterance"

@dataclass
class Caption:
    """A caption packet in MPEG-TS."""
    start_seconds: int
    end_seconds: int
    text: str

@dataclass
class Utterance:
    """A pair of audio wave data and transcription."""
    buffer: list
    samplerate: int
    duration: float
    text: str
    ctc: float
    asr: str = None
    cer: float = None
