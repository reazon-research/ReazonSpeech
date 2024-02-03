==========================
reazonspeech.espnet.oneseg
==========================

This supplies a set of functions to analyze
`Japanese one-segment digital television streams <https://en.wikipedia.org/wiki/1seg>`_
(so-called oneseg).

This module requires
`ESPnet <https://github.com/espnet/espnet>`_
and
`ffmpeg <https://ffmpeg.org/>`_.

Install
=======

.. code:: console

    $ git clone https://github.com/reazon-research/reazonspeech
    $ pip install reazonspeech/pkg/espnet-oneseg

Usage
=====

Extract captions
----------------

Here is the most basic usage of this module:

.. code-block:: python

   from reazonspeech.espnet.oneseg import get_captions

   captions = rs.get_captions("test.m2ts")

Given recorded stream data, get_captions() extracts captions from the stream.

.. code-block:: python

    Caption(start_seconds=21.3, end_seconds=25.1, text='こんにちは。正午のニュースです。')
    Caption(start_seconds=30.3, end_seconds=34.2, text='本日十時に北海道に')
    Caption(start_seconds=34.2, end_seconds=35.1, text='陸上機が到着しました。')

The start_seconds/end_seconds fields represent the display timings of the caption.

Format captions
---------------

Often a caption packet only contains a part of the original utterance.

You can use build_sentences() to merge/split captions according to the
sentence boundaries.

.. code-block:: python

   from reazonspeech.espnet.oneseg import build_sentences

   captions = build_sentences(captions)

Here are example outputs:

.. code-block:: python

    Caption(start_seconds=21.3, end_seconds=25.1, text='こんにちは。')
    Caption(start_seconds=21.3, end_seconds=25.1, text='正午のニュースです。')
    Caption(start_seconds=30.3, end_seconds=35.1, text='本日十時に北海道に陸上機が到着しました。')

Create corpus
-------------

Install ``ffmpeg`` and set up a ReazonSpeech model:

.. code:: console

    $ sudo apt install ffmpeg git-lfs
    $ git clone https://huggingface.co/reazon-research/reazonspeech-espnet-v2
    $ ln -s reazonspeech-espnet-v2/exp

Use the following code to generate a corpus:

.. code-block:: python

   from espnet2.bin.asr_align import CTCSegmentation
   from reazonspeech.espnet.oneseg import get_utterances, save_as_zip

   # Load audio and ASR model
   ctc_segmentation = CTCSegmentation(
       asr_train_config="exp/asr_train_asr_conformer_raw_jp_char/config.yaml",
       asr_model_file="exp/asr_train_asr_conformer_raw_jp_char/valid.acc.ave_10best.pth",
       kaldi_style_text=False,
       fs=16000,
   )

   # Extract audio and transcriptions
   utt = get_utterances("test.m2ts", ctc_segmentation)
   save_as_zip(utt, path="corpus.zip")
