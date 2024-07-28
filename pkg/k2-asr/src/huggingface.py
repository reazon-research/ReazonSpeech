import os
import huggingface_hub as hf
import sherpa_onnx

# The following definitions should match the repository layout
# on Hugging Face Hub. Whenever the HF repo is changed, this
# file should be updated accordingly.
#
# https://huggingface.co/reazon-research/reazonspeech-k2-v2

HF_REPO_ID = "reazon-research/reazonspeech-k2-v2"

HF_REPO_FILES = {
    "fp32": {
        "tokens": "tokens.txt",
        "encoder": "encoder-epoch-99-avg-1.onnx",
        "decoder": "decoder-epoch-99-avg-1.onnx",
        "joiner": "joiner-epoch-99-avg-1.onnx",
    },
    "int8": {
        "tokens": "tokens.txt",
        "encoder": "encoder-epoch-99-avg-1.int8.onnx",
        "decoder": "decoder-epoch-99-avg-1.int8.onnx",
        "joiner": "joiner-epoch-99-avg-1.int8.onnx",
    }
}

def load_model(device="cpu", precision="fp32"):
    """Load ReazonSpeech model from Hugging Face

    Args:
      device (str): "cpu", "cuda" or "coreml"
      precision (str): Whether to load quantized model ("fp32" or "int8")

    Returns:
      sherpa_onnx.OfflineRecognizer
    """
    if precision not in HF_REPO_FILES:
        raise ValueError("Unknown precision: '%s'" % precision)

    files = HF_REPO_FILES[precision]

    # If the model is found in the local cache, do not connect
    # to Hugging Face.
    try:
        basedir = hf.snapshot_download(HF_REPO_ID, local_files_only=True)
    except hf.utils.LocalEntryNotFoundError:
        basedir = hf.snapshot_download(HF_REPO_ID)

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
