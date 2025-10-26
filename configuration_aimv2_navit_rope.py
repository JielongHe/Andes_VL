from typing import Any

from transformers.configuration_utils import PretrainedConfig

__all__ = ["Aimv2VisionConfig"]


class Aimv2VisionConfig(PretrainedConfig):
    model_type: str = "aimv2"

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 2816,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 8,
        num_channels: int = 3,
        image_size: int = 224,
        patch_size: int = 14,
        rms_norm_eps: float = 1e-5,
        attention_dropout: float = 0.0,
        projection_dropout: float = 0.0,
        qkv_bias: bool = False,
        use_bias: bool = False,
        hidden_stride: int = 2,
        window_size: int = 112,
        fullatt_block_indexes: list = None,
        temporal_patch_size: int = 1,
        preserve_original_pe: bool = False,
        interpolate_pe_method: str = 'one_dim',
        disable_rope: bool = False,
        min_pixels: int = 3136,
        max_pixels: int = 1960000,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_channels = num_channels
        self.patch_size = patch_size
        self.image_size = image_size
        self.attention_dropout = attention_dropout
        self.rms_norm_eps = rms_norm_eps

        self.projection_dropout = projection_dropout
        self.qkv_bias = qkv_bias
        self.use_bias = use_bias

        self.hidden_stride = hidden_stride
        self.window_size = window_size
        self.fullatt_block_indexes = fullatt_block_indexes
        self.temporal_patch_size = temporal_patch_size
        self.preserve_original_pe = preserve_original_pe
        self.interpolate_pe_method = interpolate_pe_method
        self.disable_rope = disable_rope
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels