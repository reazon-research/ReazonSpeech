import argparse
from reazonspeech.k2.asr import audio_from_path, load_model, transcribe

parser = argparse.ArgumentParser(description="Transcribe audio using reazonspeech.")
parser.add_argument("--lang", required=True, choices=["ja", "ja-en"], help="Language to use for transcription (ja or ja-en).")
args = parser.parse_args()
lang = args.lang

if lang == "ja":
    PATHS = ["test_wavs/test_ja_1.wav",    # 13 sec
             "test_wavs/test_ja_2.wav"]    # 11 sec
elif lang == "ja-en":
    PATHS = ["test_wavs/test_ja_1.wav",    # 13 sec
             "test_wavs/test_ja_2.wav",    # 11 sec
             "test_wavs/test_ja_en.wav",   # 16 sec
             "test_wavs/test_en_1.wav",    # 12 sec
             "test_wavs/test_en_2.wav"]    # 16 sec

print("Loading model...")
model = load_model(language=lang)

print("Begin transcribing tests: ")
for path in PATHS:
    audio = audio_from_path(path)
    transcription = transcribe(model,audio)
    print(transcription.text)