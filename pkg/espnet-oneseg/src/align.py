from .utils import load_audio
from .caption import get_captions
from .sentence import build_sentences
from .interface import Utterance
from .text import cer, normalize

__all__ = "get_utterances",

# Live programs have ~25 seconds delay when showing captions.
# Take a big chunk and refine it using CTCSegmentation.
_MARGIN = 25

# CTCSegmentation tends to miss the last syllable.
# See lumaku/ctc-segmentation#10
_PADDING = 0.1

def _slice(buffer, samplerate, start, end):
    start = int(start * samplerate)
    end = int(end * samplerate)
    return buffer[start:end]

def _align(buffer, samplerate, caption, ctc_segmentation):
    t0 = max(caption.start_seconds - _MARGIN, 0)
    t1 = caption.end_seconds

    source = _slice(buffer, samplerate, t0, t1)
    try:
        aligned = ctc_segmentation(source, normalize(caption.text))
    except (IndexError, ValueError, RuntimeError):
        return None

    if aligned.segments:
        d0, d1, score = aligned.segments[0]
        start = t0 + d0
        end = t0 + d1 + _PADDING
        del aligned
        return Utterance(buffer=None,
                         samplerate=samplerate,
                         duration=None,
                         start_seconds=start,
                         end_seconds=end,
                         text=caption.text,
                         ctc=score)
    return None

def _add_space(utterances):
    for u0, u1 in zip(utterances, utterances[1:]):
        blank = (u1.start_seconds - u0.end_seconds) / 2
        blank = max(min(blank, 3), 0)
        u0.end_seconds += blank
        u1.start_seconds -= blank

def get_utterances(path, ctc_segmentation, speech2text=None,
                   strategy='optim'):
    """Extract utterances from MPEG-TS data

    This function supports two strategies: "optim" or "lax".

    * "optim" cuts the audio into segments at optimal points with
      least noise. Use this for creating a clean corpus.

    * "lax" includes additional audio that precedes/follows each
      utterance. Use this for robust training.

    Args:
      path (str): Path to a M2TS file
      ctc_segmentation (espnet2.bin.asr_align.CTCSegmentation): An audio aligner
      speech2text (espnet2.bin.asr.Speech2Text): An audio recognizer (optional)
      strategy (str): "optim" or "lax" (default: "optim")

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

    if strategy == 'lax':
        _add_space(utterances)

    for utt in utterances:
        utt.buffer = _slice(buffer, samplerate, utt.start_seconds, utt.end_seconds)
        utt.duration = utt.end_seconds - utt.start_seconds
        utt.samplerate = samplerate
        if speech2text:
            utt.asr = speech2text(utt.buffer)[0][0]
            utt.cer = cer(utt.text, utt.asr)

    return utterances
