from .interface import Subword, Segment, TranscribeResult

# Hyper parameters
PAD_SECONDS = 0.5
SECONDS_PER_STEP = 0.08
SUBWORDS_PER_SEGMENTS = 10
PHONEMIC_BREAK = 0.5

TOKEN_EOS = {'。', '?', '!'}
TOKEN_COMMA = {'、', ','}
TOKEN_PUNC = TOKEN_EOS | TOKEN_COMMA

def _starts_with_sp_whitespace(model, token_id):
    """Return True iff token_id maps to the SentencePiece leading-whitespace
    meta piece (▁, U+2581).

    NeMo's RNN-T beam search (ALSD/MAES/...) sometimes emits this meta piece
    at hyp.y_sequence[0] with its own entry in hyp.timestamp[0]. The piece is
    later dropped (it stringifies to ""), but if we do not also drop
    timestamp[0] the subsequent zip(y_sequence, timestamp) is shifted by one
    step — the first real token inherits step 0 and gets placed at the chunk
    origin, producing phantom prefix segments. The trim must therefore be
    conditional: when the hypothesis does not start with ▁ both arrays must
    be kept intact.
    """
    try:
        # SentencePiece leading-whitespace meta piece (U+2581 "▁")
        return model.tokenizer.tokenizer.id_to_piece(int(token_id)) == "▁"
    except Exception:
        # token_id might be out of SentencePiece vocab range (e.g., RNNT blank idx)
        return False


def find_end_of_segment(subwords, start):
    """Heuristics to identify speech boundaries"""
    length = len(subwords)
    for idx in range(start, length):
        if idx < length - 1:
            cur = subwords[idx]
            nex = subwords[idx + 1]
            if nex.token not in TOKEN_PUNC:
                if cur.token in TOKEN_EOS:
                    break
                elif idx - start >= SUBWORDS_PER_SEGMENTS:
                    if cur.token in TOKEN_COMMA or nex.seconds - cur.seconds > PHONEMIC_BREAK:
                        break
    return idx


def decode_hypothesis(model, hyp):
    """Decode ALSD beam search info into transcribe result

    Args:
        model (EncDecRNNTBPEModel): NeMo ASR model
        hyp (Hypothesis): Hypothesis to decode

    Returns:
        TranscribeResult
    """
    # If the hypothesis starts with the SentencePiece leading-whitespace meta
    # piece (▁), trim it from both y_sequence and timestamp to keep zip
    # aligned. See _starts_with_sp_whitespace() for the rationale.
    y_sequence = hyp.y_sequence.tolist()
    timestamps = hyp.timestamp.tolist() if hasattr(hyp.timestamp, "tolist") else list(hyp.timestamp)
    if y_sequence and _starts_with_sp_whitespace(model, y_sequence[0]):
        y_sequence = y_sequence[1:]
        timestamps = timestamps[1:]

    text = model.tokenizer.ids_to_text(y_sequence)

    subwords = []
    for idx, (token_id, step) in enumerate(zip(y_sequence, timestamps)):
        subwords.append(Subword(
            token_id=token_id,
            token=model.tokenizer.ids_to_text([token_id]),
            seconds=max(SECONDS_PER_STEP * (step - idx - 1) - PAD_SECONDS, 0)
        ))

    # In SentncePiece, whitespace is considered as a normal token and
    # represented with a meta character (U+2581). Trim them.
    subwords = [x for x in subwords if x.token]

    segments = []
    start = 0
    while start < len(subwords):
        end = find_end_of_segment(subwords, start)
        segments.append(Segment(
            start_seconds=subwords[start].seconds,
            end_seconds=subwords[end].seconds + SECONDS_PER_STEP,
            text="".join(x.token for x in subwords[start:end+1]),
        ))
        start = end + 1

    return TranscribeResult(text, subwords, segments)
