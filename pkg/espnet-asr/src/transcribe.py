import tqdm
import numpy as np
import torch
from .audio import norm_audio
from .interface import TranscribeConfig, TranscribeResult, Segment
from .ctc import split_text, find_blank

# Hyper parameters
WINDOW_SECONDS = 20
PADDING = (16000, 8000)

def load_model(device=None):
    """Load ReazonSpeech model

    Args:
      device (str): Specify "cuda" or "cpu"

    Returns:
      espnet2.bin.asr_inference.Speech2Text
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    from espnet2.bin.asr_inference import Speech2Text
    return Speech2Text.from_pretrained(
        "https://huggingface.co/reazon-research/reazonspeech-espnet-v2",
        lm_weight=0,
        device=device,
    )

def transcribe(model, audio, config=None):
    """Interface function to transcribe audio data

    Args:
      model (espnet2.bin.asr_inference.Speech2Text): ReazonSpeech model
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
