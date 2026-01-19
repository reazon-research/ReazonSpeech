import numpy as np
import torch
import tqdm
from espnet2.bin.asr_inference import Speech2Text
from reazonspeech.shared.audio import norm_audio
from reazonspeech.shared.interface import AudioData, Segment, TranscribeConfig, TranscribeResult

from .ctc import find_blank, split_text

# Hyper parameters
WINDOW_SECONDS = 20
PADDING = (16000, 8000)


def load_model(device: torch.device | str | None = None) -> Speech2Text:
    """Load ReazonSpeech model

    Args:
      device (torch.device | str | None): Specify "cuda" or "cpu"

    Returns:
      Speech2Text
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    return Speech2Text.from_pretrained(
        "https://huggingface.co/reazon-research/reazonspeech-espnet-v2",
        lm_weight=0,
        device=device,
    )


def transcribe(model: Speech2Text, audio: AudioData, config: TranscribeConfig | None = None) -> TranscribeResult:
    """Interface function to transcribe audio data

    Args:
      model (Speech2Text): ReazonSpeech model
      audio (AudioData): Audio to transcribe
      config (TranscribeConfig): Additional settings

    Returns:
      TranscribeResult
    """
    if config is None:
        config = TranscribeConfig()

    audio = norm_audio(audio)

    pos = 0
    fulltext = ""
    segments = []

    window = int(WINDOW_SECONDS * audio.samplerate)
    pbar = tqdm.tqdm(total=len(audio.waveform), desc='Transcribe',
                     disable=not config.verbose)

    with pbar:
        while pos < len(audio.waveform):
            samples = audio.waveform[pos:]

            # If the audio data is very long, find out the longest
            # non-speech region and perform decoding up to that point.
            if len(samples) > window:
                blank = find_blank(model, samples[:window])
                mid = int((blank.start + blank.end) / 2)
                samples = samples[:mid]

            asr = model(np.pad(samples, PADDING, mode="constant"))[0][0]
            fulltext += asr

            for start, end, text in split_text(model, samples, asr):
                segments.append(Segment(
                    start_seconds=((pos + start) / audio.samplerate),
                    end_seconds=((pos + end) / audio.samplerate),
                    text=text,
                ))
            pos += len(samples)
            pbar.n = pos
            pbar.refresh()

    return TranscribeResult(fulltext, segments)
