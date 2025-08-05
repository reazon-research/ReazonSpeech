=======================
reazonspeech.evaluation
=======================

This supplies the main interface for evaluating ReazonSpeech models and also other speech recognition models.

Install
=======

.. code::

    $ pip install git+https://github.com/reazon-research/ReazonSpeech.git#subdirectory=pkg/evaluation

or

.. code::

    $ git clone https://github.com/reazon-research/ReazonSpeech
    $ pip install ReazonSpeech/pkg/evaluation

Usage
=====

Python interface
----------------

.. code:: python3

  from reazonspeech.evaluation import BaseEvaluator

  # Override the _evaluate method to use your own model
  # You should use the `rank` and `num_gpus` arguments to load the model on the correct GPU in multi-GPU inference
  class RSNemoEvaluator(BaseEvaluator):
    def _evaluate(self, example, rank: int | None = None, num_gpus: int | None = None, **kwargs) -> EvaluationResult:
      if rank is None:
        rank = 0
      if num_gpus is None:
        num_gpus = 1
      if self.model is None:
        self.model = load_model(device=f"cuda:{rank % num_gpus}" if torch.cuda.is_available() else "cpu")
      ret = transcribe(self.model, audio_from_path(example["audio"]["path"]))
      return {"prediction": ret.text}
    
    def _evaluate_batch(self, examples, rank: int | None = None, num_gpus: int | None = None, **kwargs) -> EvaluationResult:
      raise NotImplementedError("This method is not implemented")

  evaluator = RSNemoEvaluator()
  dataset = load_dataset("reazon-research/reazonspeech", "tiny", split="train")
  dataset = dataset.cast_column("audio", Audio(decode=False)).select(range(10))
  evaluated = evaluator.evaluate(dataset=dataset, text_column="transcription", num_gpus=args.num_gpus, num_proc=args.num_proc)

Examples and reproducibility
============================

We provide examples for evaluating ReazonSpeech models and other speech recognition models.

- `whisper <https://github.com/reazon-research/ReazonSpeech/blob/main/pkg/evaluation/examples/whisper>`_
- `reazonspeech v2 espnet-asr <https://github.com/reazon-research/ReazonSpeech/blob/main/pkg/evaluation/examples/rs-espnet>`_
- `reazonspeech v2 nemo-asr <https://github.com/reazon-research/ReazonSpeech/blob/main/pkg/evaluation/examples/rs-nemo>`_
- `reazonspeech v2 k2-asr <https://github.com/reazon-research/ReazonSpeech/blob/main/pkg/evaluation/examples/rs-k2>`_