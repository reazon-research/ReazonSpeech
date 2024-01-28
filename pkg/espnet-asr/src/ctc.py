import collections
import numpy as np
import torch
import ctc_segmentation

TOKEN_EOS = {'。', '?', '!'}
TOKEN_COMMA = {'、', ','}
TOKEN_PUNC = TOKEN_EOS | TOKEN_COMMA
PHONEMIC_BREAK = 8000
CHARS_PER_SEGMENT = 15

def ctc_decode(model, samples):
    """Get character probabilities per frame using CTC network"""

    # Prepare audio data for encode()
    speech = torch.tensor(samples).unsqueeze(0)
    length = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))

    # Convert to appropriate types
    dtype = getattr(torch, model.dtype)
    speech = speech.to(device=model.device, dtype=dtype)
    length = length.to(device=model.device)

    # Pass audio data through CTC network
    enc = model.asr_model.encode(speech, length)[0]
    lpz = model.asr_model.ctc.softmax(enc)
    return lpz.detach().squeeze(0).cpu().numpy()

def find_blank(model, samples, threshold=0.98):
    """Find no-speech segment in audio stream.

    The entire point of this function is to detect a reasonable
    audio segment for ASR tasks, and to increase the accuracy of
    ASR tasks.

    See also: arXiv:2002.00551
    """
    Blank = collections.namedtuple('Blank', ['start', 'end'])
    blank_id = model.asr_model.blank_id
    nsamples = len(samples)

    # Get character probability matrix using CTC
    lpz = ctc_decode(model, samples)

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

def get_timings(model, samples, text):
    """Compute playback timing of each character using CTC segmentation"""
    lpz = ctc_decode(model, samples)

    opt = ctc_segmentation.CtcSegmentationParameters(
        index_duration = len(samples) / (lpz.shape[0] + 1),
        char_list = model.asr_model.token_list[:-1]
    )
    matrix, indices = ctc_segmentation.prepare_text(opt, [text])
    timings = ctc_segmentation.ctc_segmentation(opt, lpz, matrix)[0]

    # "+1" to skip a preceding blank character.
    return timings[indices[0]+1:indices[1]]

def find_end_of_segment(text, timings, start):
    nchar = len(text)
    for idx in range(start, nchar):
        if idx < nchar - 1:
            cur = text[idx]
            nex = text[idx + 1]
            if nex not in TOKEN_PUNC:
                if cur in TOKEN_EOS:
                    break
                elif idx  - start >= CHARS_PER_SEGMENT:
                    if cur in TOKEN_COMMA or timings[idx + 1] - timings[idx] > PHONEMIC_BREAK:
                        break
    return idx

def split_text(model, samples, text):
    """Split texts into segments (with timestamps)"""
    try:
        timings = get_timings(model, samples, text)
    except Exception:
        return [(0, len(samples), text)]

    ret = []
    start = 0
    while start < len(text):
        end = find_end_of_segment(text, timings, start)
        ret.append((timings[start], timings[end], text[start:end + 1]))
        start = end + 1
    return ret
