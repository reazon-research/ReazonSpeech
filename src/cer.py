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
    elif cer < 1:
        return 1, cer
    else:
        return 2, cer


def get_timestamps(
    audio_file_path: str,
    wav_file_path: str,
    output_audio_file_path: str,
    csv_file_path: str,
    whisper_model: WhisperModel,
    utt: bool = False,
) -> None:
    if not os.path.exists(output_audio_file_path):
        os.makedirs(output_audio_file_path)
    # 音声ファイルを保存&読み込む
    s2 = datetime.now()
    audio, _ = librosa.load(wav_file_path, sr=16000)
    print(f"librosa load: {datetime.now() - s2}")

    model = whisper_model

    if utt:
        # Load the utterances object from the file
        with open("utterances.pkl", "rb") as f:
            utterances = pickle.load(f)
    else:
        utterances = get_ctc_segmentation(audio_file_path)

    # 一つ一つの字幕とタイムスタンプの組み合わせに対して, 閾値の調整＆大きく外れているものを取り除く
    output_dataset = []
    for idx, utt in enumerate(utterances):
        print(f"\n{idx}番目: {utt.text}の推論を開始")
        true_text = text_cleanup(utt.text)
        output_file_path = f"{output_audio_file_path}{idx}_{true_text}.wav"
        scipy.io.wavfile.write(
            output_file_path, 16000, audio[int(utt.start_seconds * 16000) : int(utt.end_seconds * 16000)]
        )
        # 閾値調整
        if len(utt.text) > 70:
            continue
        utt.start_seconds, utt.end_seconds, predicted_text = get_cer_infer(utt, audio, model)
        true_text = correct_typo(true_text)
        flag, _ = save_to_dataset(true_text, predicted_text)
        # 一つ前の終わりを調整する
        if idx != 0 and utterances[idx - 1].end_seconds > utt.start_seconds and flag != 2:
            utterances[idx - 1].end_seconds = utt.start_seconds - 0.1
            output_file_path = f"{output_audio_file_path}{idx - 1}_{true_text}.wav"
            scipy.io.wavfile.write(
                output_file_path, 16000, audio[int(utterances[idx - 1].start_seconds * 16000) : int(utterances[idx - 1].end_seconds * 16000)]
            )
        if flag != 0:
            os.remove(output_file_path)
            continue
        scipy.io.wavfile.write(
            output_file_path, 16000, audio[int(utt.start_seconds * 16000) : int(utt.end_seconds * 16000)]
        )
        output_dataset.append(utt)

    print("残ったデータの件数: ", len(output_dataset))
    create_csv(output_dataset, csv_file_path)
    print(f"total time: {datetime.now() - s2}")


def get_cer_infer(utt, audio, model):
    utt.start_seconds += 0.22
    true_text = correct_typo(text_cleanup(utt.text))
    # 初期値
    flag = False
    infer_text = ""
    whisper_audio = audio[int(utt.start_seconds * 16000) : int(utt.end_seconds * 16000)]

    s1 = datetime.now()
    # 一つ前の書き起こしと比較をするために用いる
    flag_text = infer_text
    # 始まりを短くする
    for i in range(4):
        infer_text = transcribe_audio(whisper_audio, model)
        infer_text = correct_typo(text_cleanup(infer_text))
        # 書き起こしした文字が0文字の場合はエラーがうまくできないので, バグ回避のため一度
        if len(infer_text) <= 1 or not true_text:
            utt.start_seconds += 0.1
            whisper_audio = audio[int(utt.start_seconds * 16000) : int(utt.end_seconds * 16000)]
            continue
        # 明らかに推論場所がおかしい時は細かい調整を飛ばす
        # TODO: 別モデルを使うorCTCモデルの精度を上げる
        if flag == 2:
            break
        # 一回目の書き起こしは前回との比較がないのでそのまま代入
        if flag_text == "":
            flag_text = infer_text
        # 二回目以降の書き起こしは前との書き起こしで最初の文字が違う, かつ正解データと前の書き起こしが一致している場合は”始まりの予測を遅くしすぎたという”判定で終了させる
        elif flag_text[0] != infer_text[0] and flag_text[0] == true_text[0]:
            utt.start_seconds -= 0.1
            whisper_audio = audio[int(utt.start_seconds * 16000) : int(utt.end_seconds * 16000)]
            break
        # 書き起こし開始の予測が早すぎた場合は一度戻す
        # 毎回文字数が足りなくて戻りすぎている場合があったので今は0.3にしている
        elif len(infer_text) + 3 < len(true_text):
            utt.start_seconds -= 0.3
        else:
            utt.start_seconds += 0.1
            whisper_audio = audio[int(utt.start_seconds * 16000) : int(utt.end_seconds * 16000)]
    flag, cer = save_to_dataset(true_text, infer_text)
    print(
        f"1: 始まり判定 flag: {flag}, cer: {cer}, eval time: {datetime.now() - s1}, infer: {infer_text}, true: {true_text}"
    )
    if cer == 0 or flag == 2:
        utt.end_seconds += 0.5
        print(f"終了, cer: {cer}, total eval time: {datetime.now() - s1}")
        return utt.start_seconds, utt.end_seconds, infer_text

    utt.end_seconds -= 0.4

    # 終わりを伸ばす
    s2 = datetime.now()
    for i in range(10):
        utt.end_seconds += 0.2
        whisper_audio = audio[int(utt.start_seconds * 16000) : int(utt.end_seconds * 16000)]
        infer_text = transcribe_audio(whisper_audio, model)
        infer_text = correct_typo(text_cleanup(infer_text))
        flag, cer = save_to_dataset(true_text, infer_text)
        print(f"flag: {flag}, cer: {cer}, infer: {infer_text}, true: {true_text}")
        # 綺麗に書き起こせた場合
        if flag == 0:
            utt.end_seconds += 0.3
            break
        # あとでbreakのみにする
        elif flag == 2:
            utt.end_seconds -= 0.2 * (i + 1)
            break
    print(
        f"2: 終わりを伸ばす判定 flag: {flag}, cer: {cer}, eval time: {datetime.now() - s2}, infer: {infer_text}, true: {true_text}"
    )

    utt.duration = utt.start_seconds - utt.end_seconds
    # 導入時はreturnの引数を変える
    print(f"終了, cer: {cer}, total eval time: {datetime.now() - s1}")
    return utt.start_seconds, utt.end_seconds, infer_text


def get_ctc_segmentation(audio_file_name):
    # Extract audio and transcriptions
    print("audio_file_name: ", audio_file_name)
    utterances = rs.get_utterances(audio_file_name, ctc_segmentation)
    print("get alignment successful")
    # Save the utterances object to a file
    with open("utterances.pkl", "wb") as f:
        pickle.dump(utterances, f)

    return utterances


def single_m2ts_infer(audio_file_path, wav_file_path, output_file_path, csv_file_path, model):
    # m2tsファイルを.mp3に変換
    extract_audio_from_m2ts(audio_file_path, wav_file_path)
    get_timestamps(
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
        print(f"Start eval {file_name}")
        if file_name != "test2.wav":
            continue
        if file_name.endswith(".wav"):
            audio_file_path = f"{audio_dir}{file_name[:-4]}.m2ts"
            wav_file_path = f"{audio_dir}{file_name}"
            output_file_path = output_dir + file_name[:-4] + "/"
            csv_file_path = f"{csv_file_dir}{file_name[:-4]}.csv"
            get_timestamps(audio_file_path, wav_file_path, output_file_path, csv_file_path, model)
        elif file_name.endswith(".m2ts"):
            audio_file_path = f"{audio_dir}{file_name}"
            wav_file_path = f"{audio_dir}{file_name[:-5]}.wav"
            output_file_path = output_dir + file_name[:-5] + "/"
            csv_file_path = f"{csv_file_dir}{file_name[:-5]}.csv"
            single_m2ts_infer(audio_file_path, wav_file_path, output_file_path, csv_file_path, model)


if __name__ == "__main__":
    main()
    exit()
    model_size = "large-v3"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")

    audio_file_path = "audio_data/test.m2ts"
    wav_file_path = f"audio_data/test.wav"
    output_file_path = "output/ReazonSpeech_cer_data/test/"
    csv_file_path = f"output/dataset/reazonspeech_cer_data/test.csv"
    get_timestamps(wav_file_path, output_file_path, csv_file_path, model)
