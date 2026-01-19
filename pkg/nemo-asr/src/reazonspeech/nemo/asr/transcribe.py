import torch
from reazonspeech.shared.audio import norm_audio, pad_audio
from reazonspeech.shared.interface import TranscribeConfig

from .decode import PAD_SECONDS, decode_hypothesis


def load_model(device=None):
    """Load ReazonSpeech model

    Args:
      device (str): Specify "cuda" or "cpu"

    Returns:
      nemo.collections.asr.models.EncDecRNNTBPEModel
    """
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    from nemo.utils import logging
    logging.setLevel(logging.ERROR)
    from nemo.collections.asr.models import EncDecRNNTBPEModel
    return EncDecRNNTBPEModel.from_pretrained('reazon-research/reazonspeech-nemo-v2',
                                              map_location=device)

def transcribe(model, audio, config=None):
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
