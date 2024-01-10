from datetime import datetime
import unicodedata
import os
import re
import scipy
import librosa
import pickle

from espnet2.bin.asr_align import CTCSegmentation
from faster_whisper import WhisperModel
import reazonspeech as rs
from pykakasi import kakasi

from utils import create_csv, extract_audio_from_m2ts

output_dir = "output/ReazonSpeech_cer_data/"
csv_file_dir = "output/dataset/reazonspeech_cer_data/"
audio_dir = "audio_data/"
sampling_rate = 16000

# オブジェクトをインスタンス化
kakashi = kakasi()
# モードの設定：J(Kanji) to H(Hiragana)
kakashi.setMode("J", "H")
kakashi.setMode("K", "H")
conv = kakashi.getConverter()

model_size = "large-v3"
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# Load audio and ASR model
ctc_segmentation = CTCSegmentation(
    asr_train_config="exp/asr_train_asr_conformer_raw_jp_char/exp_asr_train_asr_conformer_raw_jp_char_config.yaml",
    asr_model_file="exp/asr_train_asr_conformer_raw_jp_char/valid.acc.ave_10best.pth",
    kaldi_style_text=False,
    fs=sampling_rate,
)
print("model load successful")


def correct_typo(text: str) -> str:
    # 漢数字と数字を統一させる
    text = convert_kanji_to_int(text)
    # 半角と全角を統一させる
    text = unicodedata.normalize("NFKC", text)
    text = conv.do(text)
    # 句読点と特定の記号を除去する
    punctuation_regex = re.compile(r"[.,!?？！;:()、。ぁぃぅぇぉっ\-ー~〜\'\"-]")
    text = punctuation_regex.sub("", text).replace(" ", "").lower()
    # print("convert delete", text)
    return text


def convert_kanji_to_int(text: str) -> str:
    result = text.translate(str.maketrans("零〇一壱二弐三参四五六七八九拾", "00112233456789十", ""))
    convert_table = {
        "十": "0",
        "百": "00",
        "千": "000",
        "万": "0000",
        "億": "00000000",
        "兆": "000000000000",
        "京": "0000000000000000",
    }
    unit_list = "|".join(convert_table.keys())
    while re.search(unit_list, result):
        for unit in convert_table.keys():
            zeros = convert_table[unit]
            for numbers in re.findall(f"(\d+){unit}(\d+)", result):
                result = result.replace(
                    numbers[0] + unit + numbers[1],
                    numbers[0] + zeros[len(numbers[1]) : len(zeros)] + numbers[1],
                )
            for number in re.findall(f"(\d+){unit}", result):
                result = result.replace(number + unit, number + zeros)
            for number in re.findall(f"{unit}(\d+)", result):
                result = result.replace(unit + number, "1" + zeros[len(number) : len(zeros)] + number)
            result = result.replace(unit, "1" + zeros)
    return result


def calculate_cer(reference: str, hypothesis: str) -> float:
    """
    Calculate the Character Error Rate (CER) while excluding punctuation and certain marks.
    CER is defined as the edit distance between the reference and the hypothesis.

    :param reference: The correct text
    :param hypothesis: The predicted text
    :return: The CER
    """
    # レーベンシュタイン距離（編集距離）を計算
    if len(reference) == 0:
        return 1 if len(hypothesis) > 0 else 0

    v0 = [i for i in range(len(hypothesis) + 1)]
    v1 = [0] * (len(hypothesis) + 1)

    for i in range(len(reference)):
        v1[0] = i + 1
        for j in range(len(hypothesis)):
            deletion_cost = v0[j + 1] + 1
            insertion_cost = v1[j] + 1
            if reference[i] == hypothesis[j]:
                substitution_cost = v0[j]
            else:
                substitution_cost = v0[j] + 1
            v1[j + 1] = min(deletion_cost, insertion_cost, substitution_cost)
        v0, v1 = v1, v0

    # CERを計算
    cer = v0[len(hypothesis)] / len(reference)
    return cer


def transcribe_audio(waveform, model) -> str:
    # 音声ファイルを読み込み、書き起こしを行う
    segments, _ = model.transcribe(audio=waveform, beam_size=5, language="ja", without_timestamps=True)
    text = ""
    for segment in segments:
        text += segment.text
    return text


def text_cleanup(text: str) -> str:
    # 括弧内の全ての文字を削除
    text = re.sub(r"[\(（][^)）]*[)）≫≪＞＞＜＜「」（）<<>>]", "", text)
    # 全ての空白文字を削除
    text = re.sub(r"\s", "", text)
    # 《・》「」を削除
    text = re.sub(r"[《・》「」]", "", text)
    return text


def save_to_dataset(true_text: str, predicted_text: str, threshold: float = 0.1) -> bool:
    cer = calculate_cer(true_text, predicted_text)
    cer = round(cer, 2)
    if cer <= threshold:
        return 0, cer
    elif cer <= 1:
        return 1, cer
    else:
        return 2, cer
