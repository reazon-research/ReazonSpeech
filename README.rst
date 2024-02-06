============
ReazonSpeech
============

This repository provides access to the main user tooling of ReazonSpeech project.

* https://research.reazon.jp/projects/ReazonSpeech/

Install
=======

.. code:: console

   $ git clone https://github.com/reazon-research/ReazonSpeech
   $ pip install ReazonSpeech/pkg/nemo-asr  # or espnet-oneseg, espnet-asr

Packages
========

`reazonspeech.nemo.asr <pkg/nemo-asr>`_

* Implements a fast, accurate speech recognition based on FastConformer-RNNT.
* The total number of parameters is 619M. Requires `Nvidia Nemo <https://github.com/NVIDIA/NeMo>`_.

`reazonspeech.espnet.asr <pkg/espnet-asr>`_

* Speech recognition with a Conformer-Transducer model.
* The total number of parameters is 120M. Requires `ESPnet <https://github.com/espnet/espnet>`_.

`reazonspeech.espnet.oneseg <pkg/espnet-oneseg>`_

* Provides a set of tools to analyze Japanese "one-segment" TV stream.
* Use this package to create Japanese audio corpus.

LICENSE
=======

::

    Copyright 2022-2024 Reazon Holdings, inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
