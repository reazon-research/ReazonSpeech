=================
reazonspeech.avsr
=================

This supplies the main interface for using AVSR models and pretrained models designed for AVSR.
We currently release AVHuBERT models.

Our released models and their performance results are on Hugging Face Hub: [AVista Models](https://huggingface.co/collections/enactic/avista-68edd84eea1e4049f7e72d8d)

Install
=======

.. code::

    $ pip install git+https://github.com/reazon-research/ReazonSpeech.git#subdirectory=pkg/avsr

or

.. code::

    $ git clone https://github.com/reazon-research/ReazonSpeech
    $ pip install ReazonSpeech/pkg/avsr

Usage
=====

AVSR Models
-----------

We also release AVSR models, which are encoder-decoder models. This models are implemented by following Hugging Face transformers interface, so you can easily transcribe audio with video using generate method.

You can load AVSR models by directly using Hugging Face transformers if you trust our remote code.

.. code:: python3

  from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

  processor = AutoProcessor.from_pretrained("path/to/avsr/model", trust_remote_code=True)
  model = AutoModelForSpeechSeq2Seq.from_pretrained("path/to/avsr/model", trust_remote_code=True)

  inputs = processor(raw_audio="path/to/audio", raw_video="path/to/video")
  # If mouth extraction is not performed, you can add `extract_mouth=True`
  inputs = processor(raw_audio="path/to/audio", raw_video="path/to/video", extract_mouth=True)

  outputs = model.generate(**inputs, num_beams=5, max_new_tokens=256)
  transcription = processor.decode(outputs[0], skip_special_tokens=True)

You can also load AVSR models by using reazonspeech.avsr.
If you don't want to use remote code for security reasons for example, you can use the following code.

.. code:: python3

  from reazonspeech.avsr import AVHubertProcessor, AVHubertForConditionalGeneration

  processor = AVHubertProcessor.from_pretrained("path/to/avsr/model")
  model = AVHubertForConditionalGeneration.from_pretrained("path/to/avsr/model")

  inputs = processor(raw_audio="path/to/audio", raw_video="path/to/video")
  # If mouth extraction is not performed, you can add `extract_mouth=True`
  inputs = processor(raw_audio="path/to/audio", raw_video="path/to/video", extract_mouth=True)

  outputs = model.generate(**inputs, num_beams=5, max_new_tokens=256)
  transcription = processor.decode(outputs[0], skip_special_tokens=True)

Pretrained Models
-----------------

We also release pretrained models, which are encoder-only models.
You can load pretrained models by directly using Hugging Face transformers.

.. code:: python3

  from transformers import AutoFeatureExtractor, AutoModel

  extractor = AutoFeatureExtractor.from_pretrained("path/to/pretrained/model", trust_remote_code=True)
  model = AutoModel.from_pretrained("path/to/pretrained/model", trust_remote_code=True)

  inputs = extractor(raw_audio="path/to/audio", raw_video="path/to/video")
  # If mouth extraction is not performed, you can add `extract_mouth=True`
  inputs = extractor(raw_audio="path/to/audio", raw_video="path/to/video", extract_mouth=True)

  outputs = model(**inputs)

You can also load pretrained models by using reazonspeech.avsr.
If you don't want to use remote code for security reasons for example, you can use the following code.

.. code:: python3

  from reazonspeech.avsr import AVHubertFeatureExtractor, AVHubertModel

  extractor = AVHubertFeatureExtractor.from_pretrained("path/to/pretrained/model")
  model = AVHubertModel.from_pretrained("path/to/pretrained/model")

  inputs = extractor(raw_audio="path/to/audio", raw_video="path/to/video")
  # If mouth extraction is not performed, you can add `extract_mouth=True`
  inputs = extractor(raw_audio="path/to/audio", raw_video="path/to/video", extract_mouth=True)

  outputs = model(**inputs)
