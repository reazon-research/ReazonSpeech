from .utils import load_audio
from .caption import get_captions
from .sentence import build_sentences
from .interface import Utterance

__all__ = "get_utterances",

_MARGIN = (25.0, 3.0)
_PADDING = (0.0, 0.1)

def _slice(buffer, samplerate, start, end):
    start = max(0, int(start * samplerate))
    end = min(int(end * samplerate), len(buffer))
    return buffer[start:end]

def _align(buffer, samplerate, caption, ctc_segmentation):
    source = _slice(buffer, samplerate,
                    caption.start_seconds - _MARGIN[0],
                    caption.end_seconds + _MARGIN[1])

    try:
        aligned = ctc_segmentation(source, caption.text)
    except (IndexError, ValueError):
        return None

    if aligned.segments:
        start, end, score = aligned.segments[0]
        start -= _PADDING[0]
        end += _PADDING[1]
        duration = (end - start)

        return Utterance(_slice(source, samplerate, start, end),
                         samplerate=samplerate,
                         duration=duration,
                         text=caption.text,
                         score=score)
    return None

def get_utterances(path, ctc_segmentation):
    """Extract utterances from MPEG-TS data

    Args:
      path (str): Path to a M2TS file
      ctc_segmentation (espnet2.bin.asr_align.CTCSegmentation): An audio aligner

    Returns:
      A list of Utterance objects
    """
    samplerate = int(ctc_segmentation.fs)
    captions = build_sentences(get_captions(path))
    buffer = load_audio(path, samplerate)

    utterances = []
    for caption in captions:
        utt = _align(buffer, samplerate, caption, ctc_segmentation)
        if utt:
            utterances.append(utt)
    return utterances
