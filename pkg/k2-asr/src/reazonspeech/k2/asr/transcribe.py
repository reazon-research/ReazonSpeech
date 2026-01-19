import warnings

import sherpa_onnx
from reazonspeech.shared.audio import AudioData, norm_audio, pad_audio
from reazonspeech.shared.interface import Subword, TranscribeConfig, TranscribeResult

PAD_SECONDS = 0.9
TOO_LONG_SECONDS = 30.0

def transcribe(model: sherpa_onnx.OfflineRecognizer, audio: AudioData, config: TranscribeConfig | None = None) -> TranscribeResult:
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

    # Show warning if a long audio input is detected.
    duration = audio.waveform.shape[0] / audio.samplerate
    if duration > TOO_LONG_SECONDS:
        warnings.warn(
          f"Passing a long audio input ({duration:.1f}s) is not recommended, "
          "because K2 will require a large amount of memory. "
          "Read the upstream discussion for more details: "
          "https://github.com/k2-fsa/icefall/issues/1680"
        )

    stream = model.create_stream()
    stream.accept_waveform(audio.samplerate, audio.waveform)

    model.decode_stream(stream)

    subwords = []
    for t, s in zip(stream.result.tokens, stream.result.timestamps):
        subwords.append(Subword(token=t, seconds=s))

    return TranscribeResult(stream.result.text, subwords)
