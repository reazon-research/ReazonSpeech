import io
import json
import tempfile
import ffmpeg
import soundfile
import zipfile

__all__ = "save_as_zip",

def _encode(utt, format):
    bio = io.BytesIO()
    soundfile.write(bio, utt.buffer, utt.samplerate, format=format)
    return bytes(bio.getbuffer())

def load_audio(path, samplerate):
    """Read audio wave data from M2TS file

    Args:
        path (str): M2TS file to read from.
        samplerate (int): The samplerate to use

    Returns:
        The wave data in the format of np.array.
    """
    with tempfile.NamedTemporaryFile() as fw:
        (
            ffmpeg.input(path)
                  .output(fw.name, format='wav', ar=samplerate, af="pan=mono|c0=FR")
                  .run(quiet=True, overwrite_output=True)
        )
        return soundfile.read(fw.name)[0]

def save_as_zip(utterances, path, format="flac"):
    """Create a ZIP archive of audio corpus

    Args:
        utterances (list of Utterance): The audio corpus to save.
        path (str): The Zip file path to create.
        format (str): The audio format to encode utterance.

    Returns:
        None
    """

    with zipfile.ZipFile(path, 'w') as zipf:
        dataset = []
        for idx, utt in enumerate(utterances):
            name = "%04i.%s" % (idx, format)
            zipf.writestr(name, _encode(utt, format))
            dataset.append(json.dumps({
                "audio_filepath": name,
                "text": utt.text,
                "duration": utt.duration,
                "ctc": utt.ctc,
                "asr": utt.asr,
                "cer": utt.cer
            }, ensure_ascii=False))
        zipf.writestr("dataset.json", "\n".join(dataset).encode())
