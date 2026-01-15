import warnings
from contextlib import contextmanager

import numpy as np
from transformers import ProcessorMixin


class AVHubertProcessor(ProcessorMixin):
    r"""
    Constructs a AVHubert processor which wraps a AVHubert feature extractor and a AVHubert CTC tokenizer into a single
    processor.

    [`AVHubertProcessor`] offers all the functionalities of [`AVHubertFeatureExtractor`] and [`PreTrainedTokenizer`].
    See the docstring of [`~AVHubertProcessor.__call__`] and [`~AVHubertProcessor.decode`] for more information.

    Args:
        feature_extractor (`AVHubertFeatureExtractor`):
            An instance of [`AVHubertFeatureExtractor`]. The feature extractor is a required input.
        tokenizer ([`PreTrainedTokenizer`]):
            An instance of [`PreTrainedTokenizer`]. The tokenizer is a required input.
    """

    feature_extractor_class = "AutoFeatureExtractor"
    tokenizer_class = "PreTrainedTokenizerFast"

    def __init__(self, feature_extractor, tokenizer):
        super().__init__(feature_extractor, tokenizer)
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False

    def __call__(
        self,
        raw_audio: np.ndarray | str | list[np.ndarray] | list[str] | None = None,
        raw_video: np.ndarray | str | list[np.ndarray] | list[str] | None = None,
        text: str | list[str] | None = None,
        **kwargs,
    ):
        """
        When used in normal mode, this method forwards all its arguments to AVHubertFeatureExtractor's
        [`~AVHubertFeatureExtractor.__call__`] and returns its output. If used in the context
        [`~AVHubertProcessor.as_target_processor`] this method forwards all its arguments to PreTrainedTokenizer's
        [`~PreTrainedTokenizer.__call__`]. Please refer to the docstring of the above two methods for more information.
        """
        is_batched = isinstance(raw_audio, list)
        # For backward compatibility
        if self._in_target_context_manager:
            return self.current_processor(raw_audio, raw_video, text)

        if raw_audio is None and raw_video is None and text is None:
            raise ValueError("You need to specify either an `raw_audio`, `raw_video` or `text` input to process.")

        if raw_audio is not None or raw_video is not None:
            inputs = self.feature_extractor(raw_audio, raw_video, **kwargs)
        if text is not None:
            if "return_tensors" not in kwargs.keys():
                kwargs["return_tensors"] = "pt"
            if not is_batched:
                text = [text]
            text = [
                (
                    tokens
                    if tokens.startswith("<s>") and tokens.endswith("</s>")
                    else (
                        tokens + "</s>"  # append </s>
                        if tokens.startswith("<s>")
                        else (
                            "<s>" + tokens  # prepend <s>
                            if tokens.endswith("</s>")
                            else "<s>" + tokens + "</s>"  # add <s>/</s>
                        )
                    )
                )
                for tokens in text
            ]

            kwargs.pop("extract_mouth", None)
            encodings = self.tokenizer(text, **kwargs)

        if text is None:
            return inputs
        elif raw_audio is None and raw_video is None:
            return encodings
        else:
            inputs["decoder_input_ids"] = encodings["input_ids"][:, :-1].clone()
            inputs["decoder_attention_mask"] = encodings["attention_mask"][:, :-1]
            inputs["labels"] = encodings["input_ids"][:, 1:]
            return inputs

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.batch_decode`]. Please
        refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to PreTrainedTokenizer's [`~PreTrainedTokenizer.decode`]. Please refer
        to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @contextmanager
    def as_target_processor(self):
        """
        Temporarily sets the tokenizer for processing the input. Useful for encoding the labels when fine-tuning
        AVHubert.
        """
        warnings.warn(
            "`as_target_processor` is deprecated and will be removed in v5 of Transformers. You can process your "
            "labels by using the argument `text` of the regular `__call__` method (either in the same call as "
            "your audio inputs, or in a separate call."
        )
        self._in_target_context_manager = True
        self.current_processor = self.tokenizer
        yield
        self.current_processor = self.feature_extractor
        self._in_target_context_manager = False
