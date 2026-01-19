import torch
from nemo.utils import logging
from reazonspeech.shared.audio import norm_audio, pad_audio
from reazonspeech.shared.interface import AudioData, TranscribeConfig, TranscribeResult

from .decode import PAD_SECONDS, decode_hypothesis

logging.setLevel(logging.ERROR)

from nemo.collections.asr.models import EncDecRNNTBPEModel  # noqa: E402


def load_model(device: torch.device | str | None = None) -> EncDecRNNTBPEModel:
    """Load ReazonSpeech model

    Args:
      device (torch.device | str | None): Specify "cuda" or "cpu"

    Returns:
      nemo.collections.asr.models.EncDecRNNTBPEModel
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    return EncDecRNNTBPEModel.from_pretrained("reazon-research/reazonspeech-nemo-v2", map_location=device)


def transcribe(model: EncDecRNNTBPEModel, audio: AudioData, config: TranscribeConfig | None = None) -> TranscribeResult:
    """Inference audio data using NeMo model

    Args:
        model (nemo.collections.asr.models.EncDecRNNTBPEModel): ReazonSpeech model
        audio (AudioData): Audio data to transcribe
        config (TranscribeConfig): Additional settings

    Returns:
        TranscribeResult
    """
    if config is None:
        config = TranscribeConfig()

    audio = pad_audio(norm_audio(audio), PAD_SECONDS)

    waveform_tensor = torch.from_numpy(audio.waveform)

    result = model.transcribe(
        [waveform_tensor],
        batch_size=1,
        return_hypotheses=True,
        verbose=config.verbose
    )
    hyp = result[0]
    ret = decode_hypothesis(model, hyp)

    if config.raw_hypothesis:
        ret.hypothesis = hyp

    return ret
