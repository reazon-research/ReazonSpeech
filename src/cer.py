from datetime import datetime
import unicodedata
import os
import re
import scipy
import librosa
from tqdm import tqdm

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


def text_cleanup(text):
    # 漢数字と数字を統一させる
    text = convert_kanji_to_int(text)
    # 半角と全角を統一させる
    text = unicodedata.normalize("NFKC", text)
    # 句読点と特定の記号を除去する
    punctuation_regex = re.compile(r"[.,!?？！;:()、。ぁぃぅぇぉっ《・》「」\'\"-]")
    text = punctuation_regex.sub("", text).replace(" ", "").lower()
    text = conv.do(text)
    return text


def convert_kanji_to_int(string):
    result = string.translate(str.maketrans("零〇一壱二弐三参四五六七八九拾", "00112233456789十", ""))
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


def calculate_cer(reference, hypothesis):
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


def transcribe_audio(waveform, model: WhisperModel):
    # 音声ファイルを読み込み、書き起こしを行う
    segments, info = model.transcribe(audio=waveform, beam_size=5, language="ja")
    text = ""
    for segment in segments:
        text += segment.text
    return text


def save_to_dataset(true_text, predicted_text, threshold=0.2):
    cer = calculate_cer(true_text, predicted_text)
    print(true_text, predicted_text, cer)
    if cer <= threshold:
        return True
    else:
        return False


def get_cer_infer(
    m2ts_file_path,
    wav_file_path,
    output_audio_file_path,
    csv_file_path,
    whisper_model: WhisperModel,
):
    if not os.path.exists(output_audio_file_path):
        os.makedirs(output_audio_file_path)

    # ReazonSpeechを使ってタイムスタンプの予測結果を出力させる
    s1 = datetime.now()
    utterances = get_ctc_segmentation(m2ts_file_path)
    print(f"CTC Segmentation: {datetime.now() - s1}")
    # 音声ファイルを保存&読み込む
    s2 = datetime.now()
    audio, sr = librosa.load(wav_file_path, sr=16000)
    print(f"librosa load: {datetime.now() - s2}")

    # 一つ一つの字幕とタイムスタンプの組み合わせに対して, 閾値の調整＆大きく外れているものを取り除く
    output_dataset = []
    for i, utt in tqdm(enumerate(utterances)):
        output_filename = utt.text
        true_text = text_cleanup(utt.text)
        output_file_path = f"{output_audio_file_path}{i}_{output_filename}.wav"
        utterances[i].start_seconds += 0.22
        utterances[i].end_seconds += 0.5
        if len(output_filename) > 60:
            continue

        whisper_audio = audio[int(utterances[i].start_seconds * 16000) : int(utterances[i].end_seconds * 16000)]
        scipy.io.wavfile.write(output_file_path, 16000, whisper_audio)

        # 終わりを伸ばす
        for _ in range(5):
            infer_text = transcribe_audio(whisper_audio, whisper_model)
            infer_text = text_cleanup(infer_text)
            flag = save_to_dataset(true_text, infer_text)
            if flag:
                break
            else:
                utterances[i].end_seconds += 0.2
                whisper_audio = audio[
                    int(utterances[i].start_seconds * 16000) : int(utterances[i].end_seconds * 16000)
                ]

        if flag:
            flag_text = infer_text
            # 始まりを短くする
            for _ in range(3):
                infer_text = transcribe_audio(whisper_audio, whisper_model)
                infer_text = text_cleanup(infer_text)
                if flag_text[0] != infer_text[0] and flag_text[0] == true_text[0]:
                    utterances[i].start_seconds -= 0.4
                    whisper_audio = audio[
                        int(utterances[i].start_seconds * 16000) : int(utterances[i].end_seconds * 16000)
                    ]
                    break
                else:
                    utterances[i].start_seconds += 0.1
                    whisper_audio = audio[
                        int(utterances[i].start_seconds * 16000) : int(utterances[i].end_seconds * 16000)
                    ]
            scipy.io.wavfile.write(output_file_path, 16000, whisper_audio)
            output_dataset.append(utt)
        else:
            os.remove(output_file_path)

        if utterances[i - 1].end_seconds > utterances[i].start_seconds - 0.4 and i != 0:
            utterances[i - 1].end_seconds = utterances[i].start_seconds - 0.4

    create_csv(output_dataset, csv_file_path)


def get_ctc_segmentation(audio_file_name):
    # Extract audio and transcriptions
    print("audio_file_name: ", audio_file_name)
    utterances = rs.get_utterances(audio_file_name, ctc_segmentation)
    print("get alignment successful")
    return utterances


def single_m2ts_infer(audio_file_path, wav_file_path, output_file_path, csv_file_path, model):
    # m2tsファイルを.mp3に変換
    extract_audio_from_m2ts(audio_file_path, wav_file_path)
    get_cer_infer(
        audio_file_path,
        wav_file_path,
        output_file_path,
        csv_file_path,
        model,
    )


def main():
    model_size = "large-v3"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    for file_name in os.listdir(audio_dir):
        audio_file_path = f"{audio_dir}{file_name}"
        wav_file_path = f"{audio_dir}{file_name[:-5]}.wav"
        output_file_path = output_dir
        csv_file_path = f"{csv_file_dir}{file_name[:-5]}.csv"
        if file_name.endswith(".wav"):
            get_cer_infer(audio_file_path, wav_file_path, output_file_path, csv_file_path, model)
        elif file_name.endswith(".m2ts"):
            single_m2ts_infer(audio_file_path, wav_file_path, output_file_path, csv_file_path, model)


if __name__ == "__main__":
    # main()
    model_size = "large-v3"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    audio_file_path = "audio_data/test.m2ts"
    wav_file_path = f"audio_data/test.wav"
    output_file_path = "output/ReazonSpeech_cer_data/test/"
    csv_file_path = f"output/dataset/reazonspeech_cer_data/test.csv"
    s1 = datetime.now()
    get_cer_infer(audio_file_path, wav_file_path, output_file_path, csv_file_path, model)
    print(f"total time: {datetime.now() - s1}")
