============
ReazonSpeech
============

ReazonSpeech is a toolkit to build audio corpus from Japanese digital
television stream.

Installation
============

Use ``pip`` to install::

    $ pip install git+https://github.com/reazon-research/reazonspeech

QuickStart
==========

Extract captions from stream
----------------------------

Here is the most basic usage of ReazonSpeech:

.. code-block:: python

   import reazonspeech as rs

   captions = rs.get_captions("test.m2ts")

Given broadcast data, read_captions() parses and extracts caption data
from the stream.

.. code-block:: python

    Caption(start_seconds=21.3, end_seconds=25.1, text='こんにちは。正午のニュースです。')
    Caption(start_seconds=30.3, end_seconds=34.2, text='本日十時に北海道に')
    Caption(start_seconds=34.2, end_seconds=35.1, text='陸上機が到着しました。')

Each Caption instance corresponds to a packet in the stream. The
start_seconds/end_seconds fields represent the display timings of the
caption (counting from the beginning of the stream).

Build sentences from captions
-----------------------------

Often a caption packet only contains a part of the original utterance.
You can use build_sentences() to merge/split captions according to the
sentence boundaries.

.. code-block:: python

   captions = rs.build_sentences(captions)

This should make the caption data more suitable to ASR tasks.

.. code-block:: python

    Caption(start_seconds=21.3, end_seconds=25.1, text='こんにちは。')
    Caption(start_seconds=21.3, end_seconds=25.1, text='正午のニュースです。')
    Caption(start_seconds=30.3, end_seconds=35.1, text='本日十時に北海道に陸上機が到着しました。')

Build audio corpus
------------------

First, prepare ffmpeg and ESPNet2 model::

    $ sudo apt install ffmpeg
    $ ln -s /path/to/my-espnet-model/exp

You can create an audio corpus from a M2TS file very easily:

.. code-block:: python

   from espnet2.bin.asr_align import CTCSegmentation
   import reazonspeech as rs

   # Load audio and ASR model
   ctc_segmentation = CTCSegmentation(
       asr_train_config="exp/asr_train/config.yaml",
       asr_model_file="exp/asr_train/valid.acc.best.pth",
       kaldi_style_text=False,
       fs=16000,
   )

   # Extract audio and transcriptions
   utterances = rs.get_utterances("test.m2ts", ctc_segmentation)

   rs.save_as_zip(utterances, path="corpus.zip")

LICENSE
=======

::

    Copyright 2022 Reazon Holdings, inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
