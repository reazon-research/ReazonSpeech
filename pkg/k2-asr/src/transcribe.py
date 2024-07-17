import os
import sherpa_onnx
from .interface import TranscribeConfig, TranscribeResult, Subword
from .audio import audio_to_file, pad_audio, norm_audio

PAD_SECONDS = 0.9

def load_model(device='cpu'):
    """Load ReazonSpeech model

    Returns:
      sherpa_onnx
    """
    from huggingface_hub import snapshot_download
    repo_url = 'reazon-research/reazonspeech-zipformer-large'
    local_path = snapshot_download(repo_url)

    return sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=local_path + "/tokens.txt",
        encoder=local_path + "/encoder-epoch-99-avg-1.onnx",
        decoder=local_path + "/decoder-epoch-99-avg-1.onnx",
        joiner=local_path + "/joiner-epoch-99-avg-1.onnx",
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        provider=device,
    )

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
        subwords.append(Subword(t, s))

    return TranscribeResult(stream.result.text, subwords)
