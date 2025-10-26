import copy

from transformers import  Qwen3Config 
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from .configuration_aimv2_navit_rope import Aimv2VisionConfig

logger = logging.get_logger(__name__)


class AndesVLConfig(PretrainedConfig):
    model_type = 'andesvl-aimv2-qwen3'

    def __init__(
        self,
        vision_config=None,
        text_config=None,
        **kwargs):
        super().__init__(**kwargs)

        self.vision_config = Aimv2VisionConfig(**vision_config) if vision_config is not None else Aimv2VisionConfig()
        self.text_config = Qwen3Config(**text_config) if text_config is not None else Qwen3Config()

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].
        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output['vision_config'] = self.vision_config.to_dict()
        output['text_config'] = self.text_config.to_dict()
        output['model_type'] = self.__class__.model_type
        return output