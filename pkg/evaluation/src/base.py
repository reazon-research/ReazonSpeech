import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypedDict, Optional, Any, Callable

from datasets import Dataset, load_dataset
from .utils import calculate_cer


class EvaluationResult(TypedDict):
    prediction: str


class EvaluationResultBatch(TypedDict):
    predictions: list[str]


class BaseEvaluator(ABC):
    def __init__(
        self,
        model=None,
        processor=None,
        dataset=None,
        output_file: Optional[os.PathLike] = None,
        batch_size: Optional[int] = None,
        num_proc: Optional[int] = None,
        num_gpus: Optional[int] = None,
        text_column: str = "text",
    ):
        self.model = model
        self.processor = processor
        self.output_file = output_file
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.num_gpus = num_gpus
        self.text_column = text_column
        self.dataset = self._load_dataset(dataset) if dataset is not None else None

    def _load_dataset(self, dataset: Dataset | dict[str, Any] | Callable | os.PathLike | str) -> Dataset:
        if isinstance(dataset, Dataset):
            return dataset
        elif isinstance(dataset, dict):
            return Dataset.from_dict(dataset)
        elif isinstance(dataset, Callable):
            return Dataset.from_generator(dataset)
        elif isinstance(dataset, os.PathLike) or isinstance(dataset, str):
            if not isinstance(dataset, Path):
                dataset = Path(dataset)
            if dataset.is_file():
                ext = dataset.suffix
                if ext == ".jsonl":
                    ext = ".json"
                ext = ext.removeprefix(".")
                return load_dataset(ext, data_files=dataset.as_posix(), num_proc=self.num_proc)
            elif dataset.is_dir():
                return load_dataset(
                    dataset.as_posix(),
                    split="train",
                    trust_remote_code=True,
                    num_proc=self.num_proc,
                )
            else:
                raise ValueError(f"Invalid dataset path: {dataset}")
        else:
            raise ValueError(f"Invalid dataset type: {type(dataset)}")

    def _calculate_cer(self, example: dict[str, Any], text_column: str) -> dict[str, Any]:
        return {"cer": calculate_cer(example[text_column], example["prediction"])}

    def evaluate(
        self,
        dataset: Dataset | list[dict[str, Any]] | os.PathLike | str | None = None,
        batch_size: Optional[int] = None,
        num_proc: Optional[int] = None,
        num_gpus: Optional[int] = None,
        text_column: Optional[str] = None,
        output_file: Optional[os.PathLike] = None,
    ) -> Dataset:
        dataset: Dataset = self._load_dataset(dataset) if dataset is not None else self.dataset
        batch_size = batch_size or self.batch_size
        num_proc = num_proc or self.num_proc
        num_gpus = num_gpus or self.num_gpus
        text_column = text_column or self.text_column
        output_file = output_file or self.output_file

        if num_gpus is not None:
            raise NotImplementedError("Multi-gpu inference is not supported")

        # TODO: Multi-gpu inference support
        if batch_size is None:
            evaluated = dataset.map(self._evaluate, num_proc=num_proc)
        else:
            evaluated = dataset.map(self._evaluate_batch, batch_size=batch_size, num_proc=num_proc)
        evaluated = evaluated.map(
            self._calculate_cer,
            num_proc=num_proc,
            fn_kwargs={"text_column": text_column},
        )

        cer = evaluated["cer"]
        print(f"CER: {sum(cer) / len(cer) * 100:.2f}%")

        if output_file is not None:
            evaluated.to_json(output_file, num_proc=num_proc, force_ascii=False)

        return evaluated

    @abstractmethod
    def _evaluate(self, example: dict[str, Any]) -> EvaluationResult:
        raise NotImplementedError("Subclasses must implement _evaluate method")

    @abstractmethod
    def _evaluate_batch(self, batch: dict[str, Any]) -> EvaluationResultBatch:
        raise NotImplementedError("Subclasses must implement _evaluate_batch method")
