============
ReazonSpeech
============

ReazonSpeech is a toolkit to build audio corpus from Japanese digital
television stream.

Installation
============

Use ``pip`` to install::

    $ git clone https://github.com/reazon-research/reazonspeech
    $ pip install reazonspeech

QuickStart
==========

Extract captions from stream
----------------------------

Here is the most basic usage of ReazonSpeech:

.. code-block:: python

   from reazonspeech import *

   with open("sample.m2ts", "rb") as fp:
       captions = read_captions(fp)
       for caption in captions:
            print(caption)

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

   captions = build_sentences(captions)

This should make the caption data more suitable to ASR tasks.

.. code-block:: python

    Caption(start_seconds=21.3, end_seconds=25.1, text='こんにちは。')
    Caption(start_seconds=21.3, end_seconds=25.1, text='正午のニュースです。')
    Caption(start_seconds=30.3, end_seconds=35.1, text='本日十時に北海道に陸上機が到着しました。')

Create your audio corpus
------------------------

This sections shows the full usage of ReazonSpeech.

First, prepare your audio data and ESPNet2 model::

    # Extract 16khz mono audio stream from MPEG-TS
    $ ffmpeg -i sample.m2ts -ar 16000 -af "pan=mono|c0=FR" sample.wav

    # Create a symbolic link to ESPNet2 model
    $ ln -s /path/to/my-espnet-model/exp

Process the data using the following script:

.. code-block:: python

   import soundfile
   from espnet2.bin.asr_align import CTCSegmentation
   from reazonspeech import *

   # Parse captions from M2TS file
   with open("sample.m2ts", "rb") as fp:
       captions = read_captions(fp)

   # Format captions into sentences
   captions = build_sentences(captions)

   # Load audio and ASR model
   buffer, samplerate = soundfile.read("sample.wav")
   ctc_segmentation = CTCSegmentation(
       asr_train_config="exp/asr_train/config.yaml",
       asr_model_file="exp/asr_train/valid.acc.best.pth",
       kaldi_style_text=False,
       fs=samplerate,
   )

   # Align audio to captions
   utterances = align_audio(buffer, samplerate, captions, ctc_segmentation)

   # Save as a ZIP archive
   save_as_zip(utterances, path="corpus.zip", format="flac")

Once done, an archived named "corpus.zip" will be created. It contains
(1) a transcription file (dataset.json) and (2) corresponding audio
files (e.g. "0001.flac").
