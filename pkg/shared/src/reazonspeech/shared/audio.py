from typing import BinaryIO

import librosa
import numpy as np
import soundfile
import torch
from numpy.typing import NDArray

from .interface import AudioData

SAMPLERATE = 16000


def audio_from_numpy(array: NDArray[np.float32], samplerate: int) -> AudioData:
    """Load audio from Numpy array

    Args:
      array (NDArray[np.float32]): Audio audio
      samplerate (int): Sample rate of the input array

    Returns:
      AudioData
    """
    return AudioData(array, samplerate)


def audio_from_tensor(tensor: torch.FloatTensor, samplerate: int) -> AudioData:
    """Load audio from PyTorch Tensor

    Args:
      tensor (torch.FloatTensor): Audio audio
      samplerate (int): Sample rate of the input tensor

    Returns:
      AudioData
    """
    return audio_from_numpy(tensor.numpy(), samplerate)


def audio_from_path(path: str) -> AudioData:
    """Load audio from a file

    Args:
      path (str): Path to audio file

    Returns:
      AudioData
    """
    array, samplerate = librosa.load(path, sr=None)
    return audio_from_numpy(array, samplerate)


def audio_to_file(fp: BinaryIO, audio: AudioData, format: str = 'wav'):
    """Write audio data to file

    Args:
      fp (File): output file
      audio (AudioData): Audio data to write
      format (str): Audio encoding
    """
    soundfile.write(fp, audio.waveform, audio.samplerate, format=format)


def norm_audio(audio: AudioData) -> AudioData:
    """Normalize audio into 16khz mono waveform

    Args:
      audio (AudioData): Audio data to normalize

    Returns:
      AudioData (16khz mono waveform)
    """
    waveform = audio.waveform
    if audio.samplerate != SAMPLERATE:
        waveform = librosa.resample(waveform, orig_sr=audio.samplerate, target_sr=SAMPLERATE)
    if len(waveform.shape) > 1:
        waveform = librosa.to_mono(waveform)
    return AudioData(waveform, SAMPLERATE)


def pad_audio(audio: AudioData, seconds: float) -> AudioData:
    """Pad audio with silence

    Args:
      audio (AudioData): Audio data to pad
      seconds (float): Add N seconds padding

    Returns:
      AudioData
    """
    waveform = np.pad(
        audio.waveform,
        pad_width=int(seconds * audio.samplerate),
        mode='constant'
    )
    return AudioData(waveform, audio.samplerate)
