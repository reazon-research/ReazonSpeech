import spacy
import librosa

__all__ = "transcribe",

# Assume 16khz audio data.
_SAMPLE_RATE = 16000

# Decode long audio using a 20-second window
_WINDOW_SIZE = 20 * _SAMPLE_RATE

def transcribe(path, speech2text):
    """Interface function to transcribe audio data

    Args:
      path (str): Path to audio file
      speech2text (espnet2.bin.asr.Speech2Text): ASR model to use

    Yields:
      Transcribed texts (strings)
    """
    audio = librosa.load(path, sr=_SAMPLE_RATE)[0]
    pos = 0
    context = ''
    nlp = spacy.load("ja_ginza_electra")

    for pos in range(0, len(audio), _WINDOW_SIZE):
        asr = speech2text(audio[pos:pos + _WINDOW_SIZE])
        if asr:
            morph = nlp(context + asr[0][0])
            sents = [s.text for s in morph.sents]

            # The last sentence might be incomplete because a sentence
            # (or utterance) can span multipe audio windows.
            done, context = sents[:-1], sents[-1]
            for sent in done:
                yield sent

    if context:
        yield context
