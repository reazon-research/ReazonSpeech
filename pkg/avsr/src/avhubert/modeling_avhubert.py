import logging
from dataclasses import dataclass
from typing import Optional

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel
from transformers.cache_utils import StaticCache
from transformers.generation import GenerationMixin
from transformers.generation.utils import GenerationConfig, GenerationMode
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.hubert.modeling_hubert import (
    HubertEncoder,
    HubertEncoderStableLayerNorm,
)
from transformers.utils import ModelOutput

from .configuration_avhubert import AVHubertConfig
from .configuration_resnet import ResEncoderConfig
from .decoder import AVHubertDecoder, AVHubertDecoderStableLayerNorm
from .modeling_resnet import ResEncoder

logger = logging.getLogger(__name__)

NEED_SETUP_CACHE_CLASSES_MAPPING = {
    "static": StaticCache,
}


@dataclass
class AVHubertOutput:
    last_hidden_state: Optional[torch.Tensor] = None
    hidden_states: Optional[torch.Tensor] = None
    attentions: Optional[torch.Tensor] = None


class AudioFeatureExtractor(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super(AudioFeatureExtractor, self).__init__()
        self.proj = nn.Linear(in_features=input_dim, out_features=output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, T, F]
        return einops.rearrange(x, "b t f -> b f t")  # [B, F, T]


class VideoFeatureExtractor(nn.Module):
    def __init__(self, config: ResEncoderConfig, output_dim: int) -> None:
        super(VideoFeatureExtractor, self).__init__()
        self.resnet = ResEncoder(config=config)
        self.proj = nn.Linear(
            in_features=self.resnet.backend_out,
            out_features=output_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.resnet(einops.rearrange(x, "b t c h w -> b c t h w"))  # [B, F, T]
        x = self.proj(einops.rearrange(x, "b f t -> b t f"))  # [B, T, F]
        return einops.rearrange(x, "b t f -> b f t")  # [B, F, T]


class AVHubertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = AVHubertConfig
    base_model_prefix = "avhubert"
    supports_gradient_checkpointing = False

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if is_deepspeed_zero3_enabled():
                import deepspeed

                if hasattr(module, "weight_v") and hasattr(module, "weight_g"):
                    with deepspeed.zero.GatheredParameters([module.weight_v, module.weight_g], modifier_rank=0):
                        nn.init.kaiming_normal_(module.weight.data)
                else:
                    with deepspeed.zero.GatheredParameters(module.weight, modifier_rank=0):
                        nn.init.kaiming_normal_(module.weight.data)
            else:
                if hasattr(module, "parametrizations"):
                    nn.init.kaiming_normal_(module.parametrizations.weight.original0.data)
                    nn.init.kaiming_normal_(module.parametrizations.weight.original1.data)
                nn.init.kaiming_normal_(module.weight.data)

        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)) and module.bias is not None:
            module.bias.data.zero_()

    def _get_feat_extract_output_lengths(self, input_lengths: torch.LongTensor | int):
        """
        Computes the output length of the convolutional layers
        """

        def _conv_out_length(input_length, kernel_size, stride):
            # 1D convolutional layer output length formula taken
            # from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            return torch.div(input_length - kernel_size, stride, rounding_mode="floor") + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths


class AVHubertModel(AVHubertPreTrainedModel):
    def __init__(self, config: AVHubertConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.config = config
        self.feat2tar_ratio = config.label_rate / config.sample_rate

        # feature extractor
        resnet_config = ResEncoderConfig(relu_type=config.resnet_relu_type)
        self.feature_extractor_audio = AudioFeatureExtractor(
            input_dim=config.audio_feat_dim,
            output_dim=config.encoder_embed_dim,
        )
        self.feature_extractor_video = VideoFeatureExtractor(config=resnet_config, output_dim=config.encoder_embed_dim)

        self.encoder_embed_dim = config.encoder_embed_dim
        if config.modality_fuse == "concat":
            embed = config.encoder_embed_dim * 2
        elif config.modality_fuse == "add":
            embed = config.encoder_embed_dim
        self.post_extract_proj = (
            nn.Linear(embed, config.encoder_embed_dim) if embed != config.encoder_embed_dim else None
        )

        # dropout
        self.dropout_input = nn.Dropout(config.dropout_input)

        # transformer encoder
        transformer_config = config.encoder_config
        if transformer_config.do_stable_layer_norm:
            self.encoder = HubertEncoderStableLayerNorm(config=transformer_config)
        else:
            self.encoder = HubertEncoder(config=transformer_config)
        self.layer_norm = nn.LayerNorm(embed)

    def forward_mask(self, features: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        extra = attention_mask.size(1) % features.size(1)
        if extra > 0:
            attention_mask = attention_mask[:, :-extra]
        attention_mask = attention_mask.view(attention_mask.size(0), features.size(1), -1)
        attention_mask = attention_mask.all(-1)
        return attention_mask

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        padding_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
    ) -> ModelOutput:
        if input_values is not None and pixel_values is None:
            features_audio = self.feature_extractor_audio(input_values)  # [B, F, T]
            features_video = torch.zeros_like(features_audio)  # [B, F, T]
        elif input_values is None and pixel_values is not None:
            features_video = self.feature_extractor_video(pixel_values)  # [B, F, T]
            features_audio = torch.zeros_like(features_video)  # [B, F, T]
        elif input_values is not None and pixel_values is not None:
            features_audio = self.feature_extractor_audio(input_values)  # [B, F, T]
            features_video = self.feature_extractor_video(pixel_values)  # [B, F, T]
        else:
            raise ValueError("Either `input_values` or `pixel_values` must be passed")

        # fuse modality
        if self.config.modality_fuse == "concat":
            features = torch.cat([features_audio, features_video], dim=1)
        elif self.config.modality_fuse == "add":
            features = features_audio + features_video

        features = features.transpose(1, 2)
        features = self.layer_norm(features)

        if attention_mask is not None:
            attention_mask = self.forward_mask(features, attention_mask)
        else:
            attention_mask = torch.ones(features.size()[:2], dtype=torch.bool, device=features.device)

        if self.post_extract_proj is not None:
            features = self.post_extract_proj(features)

        features = self.dropout_input(features)

        # transformer encoder
        encoder_out = self.encoder(
            hidden_states=features,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        return AVHubertOutput(
            last_hidden_state=encoder_out.last_hidden_state,
            hidden_states=encoder_out.hidden_states,
            attentions=encoder_out.attentions,
        )


class AVHubertForConditionalGeneration(AVHubertPreTrainedModel, GenerationMixin):
    def __init__(
        self,
        config: AVHubertConfig,
        **kwargs,
    ) -> None:
        super().__init__(config=config, **kwargs)
        self.config = config

        self.avhubert = AVHubertModel(config=config)
        if config.freeze_base_model:
            self.freeze_base_model()
        if config.freeze_feature_encoder:
            self.freeze_feature_encoder()

        if config.vocab_size is None:
            raise ValueError(
                f"You are trying to instantiate {self.__class__} with a configuration that "
                "does not define the vocabulary size of the language model head. Please "
                "instantiate the model as follows: `AVHubertForCTC.from_pretrained(..., vocab_size=vocab_size)`. "
                "or define `vocab_size` of your model's configuration."
            )

        self.embed_tokens = nn.Embedding(config.vocab_size, config.decoder_embed_dim, padding_idx=config.pad_token_id)
        transformer_config = config.decoder_config
        if transformer_config.do_stable_layer_norm:
            self.decoder = AVHubertDecoderStableLayerNorm(config=transformer_config)
        else:
            self.decoder = AVHubertDecoder(config=transformer_config)

        self.lm_head = nn.Linear(config.decoder_embed_dim, config.vocab_size, bias=False)
        if config.share_decoder_input_output_embed:
            # If this model shares lm head weights with the token embeddings,
            # you can access lm head weights that is the same as the token embeddings but
            # the token embeddings are directly referred to instead of lm heads when training!
            self.lm_head.weight = self.embed_tokens.weight
        else:
            nn.init.normal_(self.lm_head.weight, mean=0, std=config.decoder_embed_dim**-0.5)

        self.post_init()

    def freeze_feature_encoder(self):
        """
        Calling this function will disable the gradient computation for the feature encoder so that its parameter will
        not be updated during training.
        """
        for param in self.avhubert.feature_extractor_audio.parameters():
            param.requires_grad = False
        for param in self.avhubert.feature_extractor_video.parameters():
            param.requires_grad = False

    def freeze_base_model(self):
        """
        Calling this function will disable the gradient computation for the base model so that its parameters will not
        be updated during training. Only the classification head will be updated.
        """
        for param in self.avhubert.parameters():
            param.requires_grad = False

    def get_encoder(self):
        return self.avhubert

    def forward(
        self,
        input_values: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> ModelOutput:
        encoder_outs = self.avhubert(
            input_values=input_values,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        embed_tokens = self.embed_tokens(decoder_input_ids)
        hidden_states = self.decoder(
            inputs_embeds=embed_tokens,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        if self.config.share_decoder_input_output_embed:
            logits = F.linear(hidden_states.last_hidden_state, weight=self.embed_tokens.weight)
        else:
            logits = self.lm_head(hidden_states.last_hidden_state)

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fn(logits.view(-1, self.config.vocab_size), labels.reshape(-1))

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=None,
            decoder_hidden_states=hidden_states.hidden_states,
            decoder_attentions=hidden_states.attentions,
            cross_attentions=None,
            encoder_last_hidden_state=encoder_outs.last_hidden_state,
            encoder_hidden_states=encoder_outs.hidden_states,
            encoder_attentions=encoder_outs.attentions,
        )

    def _get_generation_mode(
        self,
        generation_config: GenerationConfig,
        assistant_model: PreTrainedModel | None,
    ) -> GenerationMode:
        """
        Returns the generation mode triggered by a [`GenerationConfig`] instance.
        """
        if generation_config.constraints is not None or generation_config.force_words_ids is not None:
            generation_mode = GenerationMode.CONSTRAINED_BEAM_SEARCH
        elif generation_config.num_beams == 1:
            if generation_config.do_sample is False:
                if (
                    generation_config.top_k is not None
                    and generation_config.top_k > 1
                    and generation_config.penalty_alpha is not None
                    and generation_config.penalty_alpha > 0
                ):
                    generation_mode = GenerationMode.CONTRASTIVE_SEARCH
                else:
                    generation_mode = GenerationMode.GREEDY_SEARCH
            else:
                generation_mode = GenerationMode.SAMPLE
        else:
            if generation_config.num_beam_groups > 1:
                generation_mode = GenerationMode.GROUP_BEAM_SEARCH
            elif generation_config.do_sample is True:
                generation_mode = GenerationMode.BEAM_SAMPLE
            else:
                generation_mode = GenerationMode.BEAM_SEARCH

        # Assisted generation may extend some generation modes
        if assistant_model is not None or generation_config.prompt_lookup_num_tokens is not None:
            if generation_mode in ("greedy_search", "sample"):
                generation_mode = GenerationMode.ASSISTED_GENERATION
            else:
                raise ValueError(
                    "You've set `assistant_model`, which triggers assisted generate. Currently, assisted generate "
                    "is only supported with Greedy Search and Sample."
                )
        return generation_mode

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor = None,
        input_values: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if decoder_input_ids is None:
            decoder_input_ids = input_ids
            decoder_attention_mask = torch.ones_like(input_ids)
        return {
            "input_values": input_values,
            "pixel_values": pixel_values,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "attention_mask": attention_mask,
        }
