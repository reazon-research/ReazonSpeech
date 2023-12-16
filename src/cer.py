import os
import re
import scipy
import librosa
from tqdm import tqdm

from espnet2.bin.asr_align import CTCSegmentation
from faster_whisper import WhisperModel
import reazonspeech as rs

from utils import create_csv, extract_audio_from_m2ts

output_dir = "output/"
cer_dir = "ReazonSpeech_cer_data/"
audio_dir = "audio_data/"
sampling_rate = 16000


def calculate_cer(reference, hypothesis):
    """
    Calculate the Character Error Rate (CER) while excluding punctuation and certain marks.
    CER is defined as the edit distance between the reference and the hypothesis.

    :param reference: The correct text
    :param hypothesis: The predicted text
    :return: The CER
    """
    # 句読点と特定の記号を除去する
    punctuation_regex = re.compile(r"[.,!?？！;:()、。ぁぃぅぇぉっ《・》「」\'\"-]")
    reference = punctuation_regex.sub("", reference).replace(" ", "").lower()
    hypothesis = punctuation_regex.sub("", hypothesis).replace(" ", "").lower()

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


def transcribe_audio(waveform):
    # 音声ファイルを読み込み、書き起こしを行う
    model_size = "large-v2"
    model = WhisperModel(model_size, device="cuda", compute_type="float16")
    segments, info = model.transcribe(audio=waveform, beam_size=5, language="ja")
    text = ""
    for segment in segments:
        text += segment.text
    return text


def save_to_dataset(true_text, predicted_text, threshold=0.3):
    cer = calculate_cer(true_text, predicted_text)
    if cer <= threshold:
        return True
    else:
        return False


def get_cer_infer(audio_file_name, csv_file_path):
    if not os.path.exists(f"{output_dir}{cer_dir}{audio_file_name[:-5]}/"):
        os.makedirs(f"{output_dir}{cer_dir}{audio_file_name[:-5]}/")

    # ReazonSpeechを使ってタイムスタンプの予測結果を出力させる
    audio_file_path = f"{audio_dir}{audio_file_name}"
    utterances = get_ctc_segmentation(audio_file_path)
    # 音声ファイルを保存&読み込む
    audio, sr = librosa.load(f"{audio_file_path[:-5]}.wav", sr=16000)

    # 一つ一つの字幕とタイムスタンプの組み合わせに対して, 閾値の調整＆大きく外れているものを取り除く
    output_dataset = []
    for i, utt in tqdm(enumerate(utterances)):
        output_filename = utt.text
        output_file_path = f"{output_dir}{cer_dir}{audio_file_name[:-5]}/{i}_{output_filename}.wav"
        utterances[i].start_seconds += 0.22
        utterances[i].end_seconds += 0.5
        if len(output_filename) > 60:
            continue

        cer_audio = audio[
            int(utterances[i].start_seconds * 16000) : int(
                utterances[i].end_seconds * 16000
            )
        ]
        infer_text = transcribe_audio(cer_audio)

        if save_to_dataset(utt.text, infer_text):
            scipy.io.wavfile.write(output_file_path, 16000, cer_audio)
            output_dataset.append(utt)

    create_csv(output_dataset, csv_file_path)


def get_ctc_segmentation(audio_file_name):
    # Load audio and ASR model
    ctc_segmentation = CTCSegmentation(
        asr_train_config="exp/asr_train_asr_conformer_raw_jp_char/exp_asr_train_asr_conformer_raw_jp_char_config.yaml",
        asr_model_file="exp/asr_train_asr_conformer_raw_jp_char/valid.acc.ave_10best.pth",
        kaldi_style_text=False,
        fs=sampling_rate,
    )
    print("model load successful")

    # Extract audio and transcriptions
    utterances = rs.get_utterances(audio_file_name, ctc_segmentation)
    print("get alignment successful")
    return utterances


def main():
    for file_name in os.listdir(audio_dir):
        if file_name.endswith(".m2ts"):
            # m2tsファイルを.mp3に変換
            extract_audio_from_m2ts(
                f"{audio_dir}{file_name}", f"{audio_dir}{file_name[:-5]}.wav"
            )
            get_cer_infer(
                file_name, f"output/dataset/reazonspeech_cer_data/{file_name[:-5]}.csv"
            )


if __name__ == "__main__":
    main()
