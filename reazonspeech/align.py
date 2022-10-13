from .interface import Utterance

__all__ = "align_audio",

_MARGIN = (25.0, 3.0)
_PADDING = (0.0, 0.1)

def _slice(buffer, samplerate, start, end):
    start = max(0, int(start * samplerate))
    end = min(int(end * samplerate), len(buffer))
    return buffer[start:end]

def align_audio(buffer, samplerate, captions, ctc_segmentation,
                margin=_MARGIN, padding=_PADDING):
    """Get utterances corresponding to captions.

    Args:
        buffer (numpy.array): The audio wave data.
        samplerate (int): The sample rate of the audio data.
        captions (list of Caption): The captions to align audio data.
        ctc_segmentation (espnet2.bin.asr_align.CTCSegmentation): The alignment algorithm.
        margin (tuple of int): Add N seconds to segments (pre-alignment)
        padding (tuple of int): Add N seconds to segments (post-alignment)

    Returns:
        A list of `Utterance` instances.
    """

    utterances = []
    for caption in captions:
        source = _slice(buffer, samplerate,
                        caption.start_seconds - margin[0],
                        caption.end_seconds + margin[1])

        try:
            aligned = ctc_segmentation(source, caption.text)
        except (IndexError, ValueError):
            continue

        if aligned.segments:
            start, end, score = aligned.segments[0]
            start -= padding[0]
            end += padding[1]
            duration = (end - start) / samplerate

            utt = Utterance(_slice(source, samplerate, start, end),
                            samplerate=samplerate,
                            duration=duration,
                            text=caption.text,
                            score=score)
            utterances.append(utt)
    return utterances
