===================
reazonspeech.k2.asr
===================

This supplies the main interface for using ReazonSpeech K2 models.

**More information can be found at** https://research.reazon.jp/projects/ReazonSpeech

Install
=======

.. code::

    $ git clone https://github.com/reazon-research/ReazonSpeech
    $ pip install ReazonSpeech/pkg/k2-asr

Usage
=====

Python interface
----------------

.. code:: python3

  from reazonspeech.k2.asr import load_model, transcribe, audio_from_path

  # Load ReazonSpeech model from Hugging Face
  model = load_model()

  # Read a local audio file
  audio = audio_from_path("speech.wav")

  # Recognize speech
  ret = transcribe(model, audio)

Comnand-line interface
----------------------

.. code::

    $ reazonspeech-k2-asr speech.wav

Use ``-h`` to show a full help.

.. code::

   $ reazonspeech-k2-asr -h
