import pandas as pd
import pickle
from espnet2.bin.asr_align import CTCSegmentation
import re
import reazonspeech as rs
import csv
import subprocess


def extract_audio_from_m2ts(input_file, output_file):
    """
    M2TSファイルから音声を抽出し、WAV形式で保存する。

    :param input_file: M2TSファイルのパス
    :param output_file: 保存するWAVファイルのパス
    """
    command = [
        "ffmpeg",
        "-y",
        "-i", input_file,  # 入力ファイル
        "-vn",             # ビデオトラックを無視
        "-acodec", "pcm_s16le", # オーディオコーデック指定
        "-ar", "44100",    # サンプリングレート
        "-ac", "2",        # オーディオチャネル数
        output_file        # 出力ファイル
    ]

    # ffmpegコマンドを実行
    subprocess.run(command, check=True)


def calculate_error_metrics(true_data_path, predicted_data_path):
    """
    2つのCSVファイルを比較して、'start_seconds' と 'end_seconds' カラムの平均誤差と標準偏差を計算します。

    :param true_data_path: 正解データのCSVファイルパス
    :param predicted_data_path: 予測データのCSVファイルパス
    :return: カラムごとの平均誤差と標準偏差を含む辞書
    """

    # CSVファイルの読み込み
    true_data = pd.read_csv(true_data_path)
    predicted_data = pd.read_csv(predicted_data_path)

    # 評価指標の計算
    metrics = {}
    for column in ["start_seconds", "end_seconds"]:
        diff = true_data[column] - predicted_data[column]
        metrics[column] = {"mean_error": diff.mean(), "std_deviation": diff.std()}

    return metrics


# 予測されたtimestampからデータセットを作成（評価用）
def create_csv(dataset, file_name):
    with open(file_name, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        # Write the header
        writer.writerow(["start_seconds", "end_seconds", "text"])

        # Write each utterance
        for data in dataset:
            writer.writerow(
                [data.start_seconds, data.end_seconds, data.text]
            )


def text_cleanup(text: str) -> str:
    # 括弧内の全ての文字を削除
    text = re.sub(r"[\(（][^)）]*[)）≫≪＞＞＜＜！？?!、。「」（）<<>>]", "", text)
    # 全ての空白文字を削除
    text = re.sub(r"\s", "", text)
    # 《・》「」を削除
    text = re.sub(r"[《・》「」]", "", text)
    return text


def get_ctc_segmentation(audio_file_name):
    sampling_rate = 16000
    # Load audio and ASR model
    ctc_segmentation = CTCSegmentation(
        asr_train_config="exp/next_inference_model/exp_asr_train_asr_conformer_raw_jp_char_config.yaml",
        asr_model_file="exp/next_inference_model/valid.acc.ave_3best.pth",
        kaldi_style_text=False,
        fs=sampling_rate,
    )
    print("model load successful")
    # Extract audio and transcriptions
    print("audio_file_name: ", audio_file_name)
    utterances = rs.get_utterances(audio_file_name, ctc_segmentation)
    print("get alignment successful")
    # Save the utterances object to a file
    with open("utterances.pkl", "wb") as f:
        pickle.dump(utterances, f)

    return utterances
