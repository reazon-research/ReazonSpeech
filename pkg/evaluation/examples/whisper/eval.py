import warnings

import torch
import whisper
from datasets import load_dataset, Audio
from reazonspeech.evaluation import (
    BaseEvaluator,
    EvaluationResult,
    EvaluationResultBatch,
)

warnings.filterwarnings("ignore")

temperature = (0.0, 0.2, 0.4, 0.6000000000000001, 0.8, 1.0)
model_args = {
    "append_punctuations": "\"'.。,，!！?？:：”)]}、",
    "beam_size": 5,
    "best_of": 5,
    "clip_timestamps": "0",
    "compression_ratio_threshold": 2.4,
    "condition_on_previous_text": True,
    "fp16": True,
    "hallucination_silence_threshold": None,
    "initial_prompt": None,
    "language": "ja",
    "length_penalty": None,
    "logprob_threshold": -1.0,
    "no_speech_threshold": 0.6,
    "patience": None,
    "prepend_punctuations": "\"'“¿([{-",
    "suppress_tokens": "-1",
    "task": "transcribe",
    "word_timestamps": False,
}


class WhisperEvaluator(BaseEvaluator):
    def __init__(self, model_name: str = "base", **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name

    def _evaluate(self, example, rank: int | None = None, num_gpus: int | None = None, **kwargs) -> EvaluationResult:
        if rank is None:
            rank = 0
        if num_gpus is None:
            num_gpus = 1
        if self.model is None:
            print(f"Loading model on GPU {rank % num_gpus}")
            self.model = whisper.load_model(self.model_name, device=f"cuda:{rank % num_gpus}" if torch.cuda.is_available() else "cpu")
        ret = whisper.transcribe(self.model, example["audio"]["path"], temperature=temperature, **model_args)
        return {"prediction": ret["text"]}

    def _evaluate_batch(self, batch) -> EvaluationResultBatch:
        raise NotImplementedError("Batch evaluation is not supported")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="base")
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    evaluator = WhisperEvaluator(output_file=args.output_file)
    dataset = load_dataset("reazon-research/reazonspeech", "tiny", split="train", num_proc=6)
    dataset = dataset.cast_column("audio", Audio(decode=False)).select(range(3))
    evaluated = evaluator.evaluate(dataset, text_column="transcription", num_gpus=args.num_gpus, num_proc=args.num_proc)
