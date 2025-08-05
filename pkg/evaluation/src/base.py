"""Base evaluator module for speech recognition evaluation.

This module provides abstract base classes and utilities for evaluating
speech recognition models on various datasets.
"""

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TypedDict, Optional, Any, Callable

from datasets import Dataset, load_dataset
from multiprocess import set_start_method
from .utils import calculate_cer, CERResult


class EvaluationResult(TypedDict):
    """Result of evaluating a single example.

    Attributes:
        prediction: The predicted text from the speech recognition model.
    """

    prediction: str


class EvaluationResultBatch(TypedDict):
    """Result of evaluating a batch of examples.

    Attributes:
        predictions: List of predicted texts from the speech recognition model.
    """

    predictions: list[str]


class BaseEvaluator(ABC):
    """Abstract base class for speech recognition evaluators.

    This class provides a framework for evaluating speech recognition models
    on various datasets with support for batch processing and multi-GPU evaluation.

    Attributes:
        model: The speech recognition model to evaluate.
        processor: The processor/tokenizer for the model.
        dataset: The loaded dataset for evaluation.
        output_file: Path to save evaluation results.
        batch_size: Batch size for evaluation.
        num_proc: Number of processes for parallel processing.
        num_gpus: Number of GPUs to use for evaluation.
        text_column: Name of the column containing reference text.
    """

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
        """Initialize the BaseEvaluator.

        Args:
            model: The speech recognition model to evaluate.
            processor: The processor/tokenizer for the model.
            dataset: The dataset to evaluate on. Can be a Dataset object, dict,
                generator function, or path to a dataset file/directory.
            output_file: Path to save evaluation results (JSONL format).
            batch_size: Batch size for evaluation. If None, processes one example at a time.
            num_proc: Number of processes for parallel processing.
            num_gpus: Number of GPUs to use for evaluation.
            text_column: Name of the column containing reference text in the dataset.
        """
        self.model = model
        self.processor = processor
        self.output_file = output_file
        self.batch_size = batch_size
        self.num_proc = num_proc
        self.num_gpus = num_gpus
        self.text_column = text_column
        self.dataset = self._load_dataset(dataset) if dataset is not None else None

    def _load_dataset(self, dataset: Dataset | dict[str, Any] | Callable | os.PathLike | str) -> Dataset:
        """Load dataset from various input formats.

        Args:
            dataset: Dataset input in various formats:
                - Dataset: Returns as-is.
                - dict: Converts to Dataset using from_dict.
                - Callable: Uses as generator for Dataset.from_generator.
                - os.PathLike or str: Loads from file or directory.

        Returns:
            Dataset: Loaded dataset object.

        Raises:
            ValueError: If dataset path is invalid or doesn't exist.
            ValueError: If dataset type is not supported.
        """
        if isinstance(dataset, Dataset):
            return dataset
        elif isinstance(dataset, dict):
            return Dataset.from_dict(dataset)
        elif isinstance(dataset, Callable):
            return Dataset.from_generator(dataset)
        elif isinstance(dataset, (os.PathLike, str)):
            if not isinstance(dataset, Path):
                dataset = Path(dataset)
            if dataset.is_file():
                ext = dataset.suffix
                if ext == ".jsonl":
                    ext = ".json"
                ext = ext.removeprefix(".")
                return load_dataset(ext, data_files={"train": dataset.as_posix()}, num_proc=self.num_proc)["train"]
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

    def _calculate_cer(self, example: dict[str, Any], text_column: str) -> CERResult:
        """Calculate Character Error Rate for a single example.

        Args:
            example: Dictionary containing prediction and reference text.
                Must have 'prediction' key and the key specified by text_column.
            text_column: Name of the column containing reference text.

        Returns:
            CERResult: Object containing distance and length for CER calculation.
        """
        return calculate_cer(example[text_column], example["prediction"])

    def evaluate(
        self,
        dataset: Dataset | list[dict[str, Any]] | os.PathLike | str | None = None,
        batch_size: Optional[int] = None,
        num_proc: Optional[int] = None,
        num_gpus: Optional[int] = None,
        text_column: Optional[str] = None,
        output_file: Optional[os.PathLike] = None,
    ) -> Dataset:
        """Evaluate the model on a dataset.

        This method runs the speech recognition model on the dataset and calculates
        Character Error Rate (CER) for each example.

        Args:
            dataset: Dataset to evaluate on. If None, uses self.dataset.
                Can be Dataset, list of dicts, or path to dataset.
            batch_size: Batch size for evaluation. If None, processes one at a time.
                Overrides self.batch_size.
            num_proc: Number of processes for parallel processing.
                Overrides self.num_proc.
            num_gpus: Number of GPUs to use. Multi-GPU requires num_gpus > 1.
                Overrides self.num_gpus.
            text_column: Name of the reference text column.
                Overrides self.text_column.
            output_file: Path to save results (JSONL format).
                Overrides self.output_file.

        Returns:
            Dataset: Dataset with added columns:
                - prediction: Model predictions
                - distance: Edit distance for each example
                - length: Length of reference text

        Raises:
            ValueError: If no dataset is provided and self.dataset is None.

        Note:
            The overall CER percentage is printed to stdout.
        """
        dataset: Dataset = self._load_dataset(dataset) if dataset is not None else self.dataset
        batch_size = batch_size or self.batch_size
        num_proc = num_proc or self.num_proc
        num_gpus = num_gpus or self.num_gpus
        text_column = text_column or self.text_column
        output_file = output_file or self.output_file

        if dataset is None:
            raise ValueError("No dataset provided and self.dataset is None.")

        use_gpus = num_gpus is not None and num_proc is not None and num_gpus > 1
        if use_gpus:
            set_start_method("spawn", force=True)

        if batch_size is None:
            evaluated = dataset.map(
                self._evaluate,
                with_rank=use_gpus,
                num_proc=num_proc,
                fn_kwargs={"num_gpus": num_gpus, "num_proc": num_proc},
            )
        else:
            evaluated = dataset.map(
                self._evaluate_batch,
                batch_size=batch_size,
                with_rank=use_gpus,
                num_proc=num_proc,
                fn_kwargs={"num_gpus": num_gpus, "num_proc": num_proc},
            )

        if use_gpus:
            set_start_method("forkserver", force=True)

        evaluated = evaluated.map(
            self._calculate_cer,
            num_proc=num_proc,
            fn_kwargs={"text_column": text_column},
        )

        dist = sum(evaluated["distance"])
        length = sum(evaluated["length"])
        print(f"CER: {dist / length * 100:.2f}%")

        if output_file is not None:
            evaluated.to_json(output_file, num_proc=num_proc, force_ascii=False)

        return evaluated

    def calculate_cer(self, dataset: Dataset, text_column: Optional[str] = None, num_proc: Optional[int] = None) -> float:
        """Calculate Character Error Rate for a dataset.

        This method assumes the dataset already contains 'prediction' column
        with model outputs.

        Args:
            dataset: Dataset containing predictions and reference text.
                Must have 'prediction' column and text_column.
            text_column: Name of the reference text column.
                If None, uses self.text_column.
            num_proc: Number of processes for parallel processing.
                If None, uses self.num_proc.

        Returns:
            float: Character Error Rate as a value between 0 and 1.
                CER = total_edit_distance / total_reference_length

        Note:
            To get percentage, multiply the result by 100.
        """
        text_column = text_column or self.text_column
        num_proc = num_proc or self.num_proc
        evaluated = dataset.map(self._calculate_cer, num_proc=num_proc, fn_kwargs={"text_column": text_column})
        dist = sum(evaluated["distance"])
        length = sum(evaluated["length"])
        return dist / length

    @abstractmethod
    def _evaluate(self, example: dict[str, Any], *args, **kwargs) -> EvaluationResult:
        """Evaluate a single example.

        This method must be implemented by subclasses to define how to
        process a single example through the model.

        Args:
            example: Dictionary containing input data for evaluation.
                The expected keys depend on the specific implementation.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments. May include:
                - num_gpus: Number of GPUs available
                - num_proc: Number of processes
                - rank: Process rank for multi-GPU

        Returns:
            EvaluationResult: Dictionary with 'prediction' key containing
                the model's output text.
        """
        raise NotImplementedError("Subclasses must implement _evaluate method")

    @abstractmethod
    def _evaluate_batch(self, batch: dict[str, Any], *args, **kwargs) -> EvaluationResultBatch:
        """Evaluate a batch of examples.

        This method must be implemented by subclasses to define how to
        process a batch of examples through the model.

        Args:
            batch: Dictionary containing batched input data.
                Each value should be a list with length equal to batch size.
                The expected keys depend on the specific implementation.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments. May include:
                - num_gpus: Number of GPUs available
                - num_proc: Number of processes
                - rank: Process rank for multi-GPU

        Returns:
            EvaluationResultBatch: Dictionary with 'predictions' key containing
                a list of model output texts.
        """
        raise NotImplementedError("Subclasses must implement _evaluate_batch method")
