import pandas as pd
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
