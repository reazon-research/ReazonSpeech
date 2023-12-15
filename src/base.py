import reazonspeech as rs
from espnet2.bin.asr_align import CTCSegmentation
from pydub import AudioSegment
from tqdm import tqdm
import os

from utils import calculate_error_metrics, create_csv, extract_audio_from_m2ts

sampling_rate = 16000
output_dir = "output/"
base_dir = "ReazonSpeech_base_data/"
audio_dir = "audio_data/"


def get_base_infer(audio_file_name, csv_file_path):
    print("start base model")
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

    # 新しいディレクトリを作成（存在しない場合のみ）
    if not os.path.exists(f"{output_dir}{base_dir}"):
        print("create new directory(base model)")
        os.makedirs(f"{output_dir}{base_dir}")
    base_file_path = f"{output_dir}{base_dir}"

    # 音声ファイルを保存&読み込む
    audio = AudioSegment.from_file(f"{audio_file_name[:-5]}.wav")

    create_csv(utterances, csv_file_path)
    calculate_error_metrics("output/dataset/test_true.csv", csv_file_path)

    for i, utt in tqdm(enumerate(utterances)):
        # 指定された時間で音声を切り取って保存
        base_audio = audio[utt.start_seconds * 1000 : utt.end_seconds * 1000]
        if len(utt.text) <= 40:
            base_audio.export(f"{base_file_path}{i}_{utt.text}.wav", format="wav")

    return utterances


def main():
    for file_name in os.listdir(audio_dir):
        if file_name.endswith(".m2ts"):
            # m2tsファイルを.mp3に変換
            extract_audio_from_m2ts(f"{audio_dir}{file_name}", f"{audio_dir}{file_name[:-5]}.wav")
            get_base_infer(f"{audio_dir}{file_name}", "output/dataset/reazonspeech_cer_data.csv")


if __name__ == "__main__":
    main()
