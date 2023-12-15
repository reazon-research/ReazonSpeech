import io
from glob import glob
from typing import Literal, Optional

import os
import scipy
import torch
from faster_whisper import decode_audio
from tqdm import tqdm

from immutable import ImmutableModel
from utils import extract_audio_from_m2ts, create_csv
from cer import get_ctc_segmentation

vad_model, utils = torch.hub.load(
    repo_or_dir="snakers4/silero-vad",
    model="silero_vad",
    onnx=True,
)
(
    get_speech_timestamps,
    _,
    _,
    VADIterator,
    _,
) = utils

sampling_rate = 16000
output_dir = "output/"
base_dir = "ReazonSpeech_base_data/"
audio_dir = "audio_data/"
vad_dir = "ReazonSpeech_vad_data/"


def load_vad_model():
    # 大体100-200ms ぐらいなので、sessionを受け付けるタイミングでloadしてしまって問題ない
    # initしたモデルの数だけmemory, gpu memoryを消費するので注意(ただし多分そんなに大きくないはず:未検証)
    model, _ = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        onnx=True,
    )  # type: ignore
    return model


class VADAction(ImmutableModel):
    action: Literal["start", "end"]


class VADStart(VADAction):
    action: str = "start"
    timestamp: int


class VADEnd(VADAction):
    action: str = "end"
    start_timestamp: int
    end_timestamp: int
    duration_ms: float


class VADStream:
    def __init__(
        self,
        model,
        min_silence_duration_ms: int = 300,
        speech_pad_ms: int = 10,
    ):
        self.model = model
        self.vad_iterator = VADIterator(
            model=self.model,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )
        self.last_start: Optional[int] = None

    def is_speaking(self) -> bool:
        return self.last_start is not None

    def __call__(self, x: torch.Tensor) -> VADStart | VADEnd | None:
        # res: { start: 10000 } | { end: 12000}
        result = self.vad_iterator(x)
        if result is None:
            return None

        if result.get("start"):
            self.last_start = result["start"]
            return VADStart(
                timestamp=self.last_start,
            )
        elif result.get("end") and self.last_start is not None:
            action = VADEnd(
                start_timestamp=self.last_start,
                end_timestamp=result["end"],
                duration_ms=(result["end"] - self.last_start) / 16000 * 1000,
            )
            self.last_start = None
            return action
        else:
            raise Exception("unexpected result", result)


def get_voice_duration_ms(audio: torch.Tensor, sampling_rate: int, file_path: str) -> float:
    from datetime import datetime
    s = datetime.now()
    timestamps = get_speech_timestamps(
        audio=audio, model=vad_model, sampling_rate=sampling_rate
    )
    for timestamp in timestamps:
        print(timestamp["start"] / 16000, timestamp["end"] / 16000)
    duration_frames = sum([timestamp["end"]-timestamp["start"] for timestamp in timestamps])
    duration = duration_frames / sampling_rate
    print(f"get_voice_duration_ms: {(datetime.now() - s).total_seconds()}")
    return duration *1000


def get_vad_infer(audio_file_name, csv_file_path):
    # 新しいディレクトリを作成（存在しない場合のみ）
    if not os.path.exists(f"{output_dir}{vad_dir}"):
        os.makedirs(f"{output_dir}{vad_dir}")

    # ReazonSpeechを使ってタイムスタンプの予測結果を出力させる
    utterances = get_ctc_segmentation(audio_file_name)

    # 一つ一つの字幕とタイムスタンプの組み合わせに対して, 閾値の調整＆大きく外れているものを取り除く
    for i, utt in tqdm(enumerate(utterances)):
        output_filename = utt.text
        base_file_path = f"{output_dir}{base_dir}{i}_{output_filename}.wav"
        output_file_path = f"{output_dir}{vad_dir}{i}_{output_filename}.wav"

        with open(
            base_file_path,
            "rb",
        ) as f:
            wav = decode_audio(io.BytesIO(f.read()))
        timestamps = get_speech_timestamps(
            audio=wav, model=vad_model, sampling_rate=sampling_rate
        )

        if timestamps:
            output_audio = wav[timestamps[0]["start"] : timestamps[-1]["end"]]
            # 新しいファイルとして保存
            scipy.io.wavfile.write(output_file_path, sampling_rate, output_audio)
            utterances[i].start_seconds = timestamps[0]["start"]
            utterances[i].end_seconds = timestamps[-1]["end"]

    create_csv(utterances, csv_file_path)


def main():
    for file_name in os.listdir(audio_dir):
        if file_name.endswith(".m2ts"):
            # m2tsファイルを.mp3に変換
            extract_audio_from_m2ts(f"{audio_dir}{file_name}", f"{audio_dir}{file_name[:-5]}.wav")
            get_vad_infer(f"{audio_dir}{file_name}", "output/dataset/reazonspeech_vad_data.csv")


if __name__ == "__main__":
    main()
