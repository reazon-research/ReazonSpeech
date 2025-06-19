import os
import huggingface_hub as hf
import sherpa_onnx

# The following definitions should match the repository layout
# on Hugging Face Hub. Whenever the HF repo is changed, this
# file should be updated accordingly.
#
# Multi lingual also has precision fp16, but is currently not
# in use.
#
# https://huggingface.co/reazon-research/reazonspeech-k2-v2
# https://huggingface.co/reazon-research/reazonspeech-k2-v2-ja-en
# https://huggingface.co/reazon-research/reazonspeech-k2-v2-ja-en-mls

def load_model(device="cpu", precision="fp32", language="ja"):
    """Load ReazonSpeech model from Hugging Face

    Args:
      device (str): "cpu", "cuda" or "coreml"
      precision (str): Whether to load quantized model ("fp32", "int8" or "int8-fp32")
      language (str): Whether to use japanese or bi-lingual model ("ja" or "ja-en" or "ja-en-mls-5k") 

    Returns:
      sherpa_onnx.OfflineRecognizer
    """

    if language == "ja":
        hf_repo_id = "reazon-research/reazonspeech-k2-v2"
        epochs = 99
    elif language == "ja-en":
        hf_repo_id = "reazon-research/reazonspeech-k2-v2-ja-en"
        epochs = 35
    elif language = "ja-en-mls-5k":
        hf_repo_id = "reazon-research/reazonspeech-k2-v2-ja-en-mls"
        epochs = 24
    else:
        raise ValueError(f"Unknown language: '{language}'")

    hf_repo_files = {
        "fp32": {
            "tokens": "tokens.txt",
            "encoder": f"encoder-epoch-{epochs}-avg-1.onnx",
            "decoder": f"decoder-epoch-{epochs}-avg-1.onnx",
            "joiner": f"joiner-epoch-{epochs}-avg-1.onnx",
        },
        "int8": {
            "tokens": "tokens.txt",
            "encoder": f"encoder-epoch-{epochs}-avg-1.int8.onnx",
            "decoder": f"decoder-epoch-{epochs}-avg-1.int8.onnx",
            "joiner": f"joiner-epoch-{epochs}-avg-1.int8.onnx",
        },
        "int8-fp32": {
            "tokens": "tokens.txt",
            "encoder": f"encoder-epoch-{epochs}-avg-1.int8.onnx",
            "decoder": f"decoder-epoch-{epochs}-avg-1.onnx",
            "joiner": f"joiner-epoch-{epochs}-avg-1.int8.onnx",
        }
    }

    if precision not in hf_repo_files:
        raise ValueError("Unknown precision: '%s'" % precision)

    files = hf_repo_files[precision]

    # If the model is found in the local cache, do not connect
    # to Hugging Face.
    try:
        basedir = hf.snapshot_download(hf_repo_id, local_files_only=True)
    except hf.utils.LocalEntryNotFoundError:
        basedir = hf.snapshot_download(hf_repo_id)

    return sherpa_onnx.OfflineRecognizer.from_transducer(
        tokens=os.path.join(basedir, files["tokens"]),
        encoder=os.path.join(basedir, files['encoder']),
        decoder=os.path.join(basedir, files['decoder']),
        joiner=os.path.join(basedir, files['joiner']),
        num_threads=1,
        sample_rate=16000,
        feature_dim=80,
        decoding_method="greedy_search",
        provider=device,
    )
