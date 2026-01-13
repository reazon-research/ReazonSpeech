from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
)

from .avhubert import (
    AVHubertConfig,
    AVHubertFeatureExtractor,
    AVHubertForConditionalGeneration,
    AVHubertModel,
    AVHubertProcessor,
)

__all__ = [
    "AVHubertConfig",
    "AVHubertModel",
    "AVHubertForConditionalGeneration",
    "AVHubertFeatureExtractor",
    "AVHubertProcessor",
]

AutoConfig.register(AVHubertConfig.model_type, AVHubertConfig)
AutoModel.register(AVHubertConfig, AVHubertModel)
AutoModelForSpeechSeq2Seq.register(AVHubertConfig, AVHubertForConditionalGeneration)
AutoFeatureExtractor.register(AVHubertConfig.model_type, AVHubertFeatureExtractor)
AutoProcessor.register(AVHubertConfig.model_type, AVHubertProcessor)

AVHubertConfig.register_for_auto_class()
AVHubertModel.register_for_auto_class("AutoModel")
AVHubertForConditionalGeneration.register_for_auto_class("AutoModelForSpeechSeq2Seq")
AVHubertFeatureExtractor.register_for_auto_class("AutoFeatureExtractor")
AVHubertProcessor.register_for_auto_class("AutoProcessor")
