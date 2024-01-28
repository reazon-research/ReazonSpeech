import tempfile
import dataclasses
from .interface import TranscribeConfig
from .decode import decode_hypothesis, PAD_SECONDS
from .audio import audio_to_file, pad_audio, norm_audio

def load_model(device='cuda'):
    """Load ReazonSpeech model

    Args:
      device (str): Specify "cuda" or "cpu"

    Returns:
      nemo.collections.asr.models.EncDecRNNTBPEModel
    """
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

    # TODO Study NeMo's transcribe() function and make it
    # possible to pass waveforms on memory.
    with tempfile.NamedTemporaryFile() as tmpf:
        audio_to_file(tmpf, audio)

        hyp, _ = model.transcribe(
            [tmpf.name],
            batch_size=1,
            return_hypotheses=True,
            verbose=config.verbose
        )
        hyp = hyp[0]

    return decode_hypothesis(model, hyp)
