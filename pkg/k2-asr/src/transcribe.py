import os
import sherpa_onnx
from .interface import TranscribeConfig, TranscribeResult, Subword
from .audio import audio_to_file, pad_audio, norm_audio

PAD_SECONDS = 0.9

def transcribe(model, audio, config=None):
    """Inference audio data using K2 model

    Args:
        model (sherpa_onnx.OfflineRecognizer): ReazonSpeech model
        audio (AudioData): Audio data to transcribe
        config (TranscribeConfig): Additional settings

    Returns:
        TranscribeResult
    """
    if config is None:
        config = TranscribeConfig()

    audio = pad_audio(norm_audio(audio), PAD_SECONDS)

    stream = model.create_stream()
    stream.accept_waveform(audio.samplerate, audio.waveform)

    model.decode_stream(stream)

    subwords = []
    for t, s in zip(stream.result.tokens, stream.result.timestamps):
        subwords.append(Subword(token=t, seconds=s))

    return TranscribeResult(stream.result.text, subwords)
