from transformers import HubertConfig, PretrainedConfig


class AVHubertConfig(PretrainedConfig):
    model_type: str = "avhubert"

    def __init__(
        self,
        label_rate: int = 100,
        encoder_layers: int = 12,
        encoder_embed_dim: int = 768,
        encoder_ffn_embed_dim: int = 3072,
        encoder_attention_heads: int = 12,
        activation_fn: str = "gelu",
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
        encoder_layerdrop: float = 0.0,
        dropout_input: float = 0.0,
        conv_dim: tuple[int, ...] = (512, 512, 512, 512, 512, 512, 512),
        conv_stride: tuple[int, ...] = (5, 2, 2, 2, 2, 2, 2),
        conv_kernel: tuple[int, ...] = (10, 3, 3, 3, 3, 2, 2),
        conv_bias: bool = False,
        conv_pos: int = 128,
        conv_pos_groups: int = 16,
        resnet_relu_type: str = "prelu",
        audio_feat_dim: int = 104,
        modality_fuse: str = "concat",
        decoder_embed_dim: int = 768,
        decoder_ffn_embed_dim: int = 3072,
        decoder_layers: int = 6,
        decoder_layerdrop: float = 0.0,
        decoder_attention_heads: int = 4,
        decoder_learned_pos: bool = False,
        decoder_normalize_before: bool = False,
        no_token_positional_embeddings: bool = False,
        decoder_dropout: float = 0.1,
        decoder_attention_dropout: float = 0.1,
        decoder_activation_dropout: float = 0.0,
        max_target_positions: int = 2048,
        share_decoder_input_output_embed: bool = False,
        no_scale_embedding: bool = True,
        sample_rate: int = 25,
        num_labels: int = 100,
        initializer_range: float = 0.02,
        do_stable_layer_norm: bool = False,
        vocab_size: int | None = None,
        freeze_feature_encoder: bool = False,
        freeze_base_model: bool = False,
        ctc_loss_reduction: str = "mean",
        ctc_zero_infinity: bool = False,
        ctc_loss_weight: float = 0.3,
        special_ids: list[int] | None = None,
        is_encoder_decoder: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.label_rate = label_rate
        self.encoder_layers = encoder_layers
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_ffn_embed_dim = encoder_ffn_embed_dim
        self.encoder_attention_heads = encoder_attention_heads
        self.activation_fn = activation_fn
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.encoder_layerdrop = encoder_layerdrop
        self.dropout_input = dropout_input
        self.conv_dim = conv_dim
        self.conv_kernel = conv_kernel
        self.conv_stride = conv_stride
        self.conv_bias = conv_bias
        self.conv_pos = conv_pos
        self.conv_pos_groups = conv_pos_groups
        self.resnet_relu_type = resnet_relu_type
        self.audio_feat_dim = audio_feat_dim
        self.modality_fuse = modality_fuse
        self.decoder_embed_dim = decoder_embed_dim
        self.decoder_ffn_embed_dim = decoder_ffn_embed_dim
        self.decoder_layers = decoder_layers
        self.decoder_layerdrop = decoder_layerdrop
        self.decoder_attention_heads = decoder_attention_heads
        self.decoder_learned_pos = decoder_learned_pos
        self.decoder_normalize_before = decoder_normalize_before
        self.no_token_positional_embeddings = no_token_positional_embeddings
        self.decoder_dropout = decoder_dropout
        self.decoder_attention_dropout = decoder_attention_dropout
        self.decoder_activation_dropout = decoder_activation_dropout
        self.max_target_positions = max_target_positions
        self.share_decoder_input_output_embed = share_decoder_input_output_embed
        self.no_scale_embedding = no_scale_embedding
        self.sample_rate = sample_rate
        self.num_labels = num_labels
        self.initializer_range = initializer_range
        self.do_stable_layer_norm = do_stable_layer_norm
        self.vocab_size = vocab_size
        self.freeze_feature_encoder = freeze_feature_encoder
        self.freeze_base_model = freeze_base_model
        self.ctc_loss_reduction = ctc_loss_reduction
        self.ctc_zero_infinity = ctc_zero_infinity
        self.ctc_loss_weight = ctc_loss_weight
        self.special_ids = special_ids
        self.is_encoder_decoder = is_encoder_decoder

    @property
    def encoder_config(self) -> HubertConfig:
        return HubertConfig(
            hidden_size=self.encoder_embed_dim,
            num_hidden_layers=self.encoder_layers,
            num_attention_heads=self.encoder_attention_heads,
            intermediate_size=self.encoder_ffn_embed_dim,
            hidden_act=self.activation_fn,
            hidden_dropout=self.dropout,
            activation_dropout=self.activation_dropout,
            attention_dropout=self.attention_dropout,
            layerdrop=self.encoder_layerdrop,
            conv_dim=self.conv_dim,
            conv_kernel=self.conv_kernel,
            conv_stride=self.conv_stride,
            conv_bias=self.conv_bias,
            num_conv_pos_embeddings=self.conv_pos,
            num_conv_pos_embedding_groups=self.conv_pos_groups,
            feat_extract_activation="gelu",
            do_stable_layer_norm=self.do_stable_layer_norm,
            max_position_embeddings=self.max_target_positions,
            learned_pos=self.decoder_learned_pos,
            share_input_output_embed=self.share_decoder_input_output_embed,
        )

    @property
    def decoder_config(self) -> HubertConfig:
        return HubertConfig(
            hidden_size=self.decoder_embed_dim,
            num_hidden_layers=self.decoder_layers,
            num_attention_heads=self.decoder_attention_heads,
            intermediate_size=self.decoder_ffn_embed_dim,
            hidden_act=self.activation_fn,
            hidden_dropout=self.decoder_dropout,
            activation_dropout=self.decoder_activation_dropout,
            attention_dropout=self.decoder_attention_dropout,
            layerdrop=self.decoder_layerdrop,
            conv_dim=self.conv_dim,
            conv_kernel=self.conv_kernel,
            conv_stride=self.conv_stride,
            conv_bias=self.conv_bias,
            num_conv_pos_embeddings=self.conv_pos,
            num_conv_pos_embedding_groups=self.conv_pos_groups,
            feat_extract_activation="gelu",
            do_stable_layer_norm=self.do_stable_layer_norm,
            max_position_embeddings=self.max_target_positions,
            learned_pos=self.decoder_learned_pos,
            share_input_output_embed=self.share_decoder_input_output_embed,
        )
