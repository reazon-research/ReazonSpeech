from dataclasses import dataclass

__all__ = "Caption", "Utterance", "TranscribeConfig"

@dataclass
class Caption:
    """A caption packet in MPEG-TS."""
    start_seconds: int
    end_seconds: int
    text: str

    def __str__(self):
        return "%.2f\t%.2f\t%s" % (self.start_seconds, self.end_seconds, self.text)

@dataclass
class Utterance:
    """A pair of audio wave data and transcription."""
    buffer: list
    samplerate: int
    duration: float
    start_seconds: float
    end_seconds: float
    text: str
    ctc: float
    asr: str = None
    cer: float = None

@dataclass
class TranscribeConfig:
    """Parameters for transcribe()"""
    samplerate: int = 16000
    window: int = 320000
    blank_threshold: float = 0.98
    padding: tuple = (16000, 4000)
