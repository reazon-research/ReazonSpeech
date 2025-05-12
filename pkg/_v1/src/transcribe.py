import collections
import librosa
import torch
import numpy as np
import ctc_segmentation as ctc
from .interface import Caption, TranscribeConfig

__all__ = "transcribe", "load_default_model"

# ---------
# ASR Utils
# ---------

def _ctc_decode(audio, speech2text):
    """Get character probabilities per frame using CTC network"""

    # Prepare audio data for encode()
    speech = torch.tensor(audio).unsqueeze(0)
    length = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))

    # Convert to appropriate types
    dtype = getattr(torch, speech2text.dtype)
    speech = speech.to(device=speech2text.device, dtype=dtype)
    length = length.to(device=speech2text.device)

    # Pass audio data through CTC network
    enc = speech2text.asr_model.encode(speech, length)[0]
    lpz = speech2text.asr_model.ctc.softmax(enc)
    return lpz.detach().squeeze(0).cpu().numpy()

def _find_blank(audio, speech2text, threshold=0.98):
    """Find no-speech segment in audio stream.

    The entire point of this function is to detect a reasonable
    audio segment for ASR tasks, and to increase the accuracy of
    ASR tasks.

    See also: arXiv:2002.00551
    """
    Blank = collections.namedtuple('Blank', ['start', 'end'])
    blank_id = speech2text.asr_model.blank_id
    nsamples = len(audio)

    # Get character probability matrix using CTC
    lpz = _ctc_decode(audio, speech2text)

    # Now find all the consecutive nospeech segment
    blanks = [Blank(nsamples, nsamples)]
    start = None
    for idx, prob in enumerate(lpz.T[blank_id]):
        if prob > threshold:
            if start is None:
                start = int(idx / (lpz.shape[0] + 1) * nsamples)
        else:
            if start and start > 0:
                end = int(idx / (lpz.shape[0] + 1) * nsamples)
                blanks.append(Blank(start, end))
            start = None

    return max(blanks, key=lambda b: b.end - b.start)

def _get_timings(text, audio, speech2text):
    """Compute playback timing of each character using CTC segmentation"""
    lpz = _ctc_decode(audio, speech2text)

    opt = ctc.CtcSegmentationParameters(
        index_duration = len(audio) / (lpz.shape[0] + 1),
        char_list = speech2text.asr_model.token_list[:-1]
    )
    matrix, indices = ctc.prepare_text(opt, [text])
    timings = ctc.ctc_segmentation(opt, lpz, matrix)[0]

    # "+1" to skip a preceding blank character.
    return timings[indices[0]+1:indices[1]]

def _split_text(asr, audio, speech2text):
    """Split text according to speech boundaries.

    This works kind of like a sentence splitter. The difference
    is that it splits texts phonemically (by no-speech pauses),
    rather than syntactically.
    """
    if len(asr) < 2:
        return [(0, len(audio), asr)]

    try:
        timings = _get_timings(asr, audio, speech2text)
    except Exception:
        return [(0, len(audio), asr)]

    # A simple heurestics to determine the phonemical threshold
    # to divide texts (98th longest speech pauses).
    threshold = np.percentile(timings[1:] - timings[:-1], 98, interpolation="nearest")

    text, start, prev = '', timings[0], timings[0]
    remain = len(asr)
    ret = []

    for char, curr in zip(asr, timings):
        # CTC Segmentation sometimes returns bogus timings for
        # the first/last characters.
        if len(text) > 1 and remain > 1 and curr - prev > threshold:
            ret.append((start, curr, text))
            start, text = curr, ''
        prev = curr
        text += char
        remain -= 1
    if text:
        ret.append((start, curr, text))
    return ret

# ---------
# Main API
# ---------

def load_default_model():
    from espnet2.bin.asr_inference import Speech2Text
    if torch.cuda.device_count() > 0:
        device = "cuda"
    else:
        device = "cpu"
    return Speech2Text.from_pretrained(
        "https://huggingface.co/reazon-research/reazonspeech-espnet-next",
        ctc_weight=0.3,
        lm_weight=0.3,
        beam_size=20,
        device=device)

def transcribe(audio, speech2text=None, config=None):
    """Interface function to transcribe audio data

    Args:
      audio (str or np.array): Path to audio file, or raw audio data.
      speech2text (espnet2.bin.asr.Speech2Text): ASR model to use

    Yields:
      Decoded captions
    """
    if config is None:
        config = TranscribeConfig()

    if speech2text is None:
        speech2text = load_default_model()

    if isinstance(audio, str):
        audio = librosa.load(audio, sr=config.samplerate)[0]

    nsamples = len(audio)
    pos = 0

    while pos < nsamples:
        segment = audio[pos:]

        # If the audio data is very long, find out the longest
        # non-speech region and perform decoding up to that point.
        if len(segment) > config.window:
            blank = _find_blank(segment[:config.window], speech2text, config.blank_threshold)
            segment = segment[:blank.end]

        asr = speech2text(np.pad(segment, config.padding, mode="constant"))[0][0]

        for start, end, text in _split_text(asr, segment, speech2text):
            yield Caption(
                start_seconds = (pos + start) / config.samplerate,
                end_seconds = (pos + end) / config.samplerate,
                text = text,
            )
        pos += len(segment)
