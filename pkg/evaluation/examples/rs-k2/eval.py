import warnings

import torch
from datasets import load_dataset, Audio
from reazonspeech.evaluation import (
    BaseEvaluator,
    EvaluationResult,
    EvaluationResultBatch,
)
from reazonspeech.k2.asr import load_model, transcribe, audio_from_path, TranscribeConfig

warnings.filterwarnings("ignore")


class RSK2Evaluator(BaseEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _evaluate(self, example, rank: int | None = None, num_gpus: int | None = None, **kwargs) -> EvaluationResult:
        if rank is None:
            rank = 0
        if num_gpus is None:
            num_gpus = 1
        if self.model is None:
            print(f"Loading model on GPU {rank % num_gpus}")
            with torch.cuda.device(rank % num_gpus):
                self.model = load_model(device="cuda" if torch.cuda.is_available() else "cpu")
            self.config = TranscribeConfig(verbose=False)
        ret = transcribe(self.model, audio_from_path(example["audio"]["path"]))
        return {"prediction": ret.text}

    def _evaluate_batch(self, batch) -> EvaluationResultBatch:
        raise NotImplementedError("Batch evaluation is not supported")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int, default=None)
    parser.add_argument("--num_proc", type=int, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    evaluator = RSK2Evaluator(output_file=args.output_file)
    dataset = load_dataset("reazon-research/reazonspeech", "tiny", split="train")
    dataset = dataset.cast_column("audio", Audio(decode=False)).select(range(10))
    evaluated = evaluator.evaluate(dataset=dataset, text_column="transcription", num_gpus=args.num_gpus, num_proc=args.num_proc)
