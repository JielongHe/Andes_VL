from torch import nn
import torch.utils.checkpoint
from transformers import Qwen3ForCausalLM
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import  logging
from .configuration_andesvl import AndesVLConfig
from .modeling_aimv2_navit_rope import Aimv2VisionModel

logger = logging.get_logger(__name__)

class AndesVLForConditionalGeneration(PreTrainedModel):
    config_class = AndesVLConfig
    main_input_name = 'pixel_values'
    _supports_flash_attn_2 = True
    _no_split_modules = ['Aimv2VisionModel','Qwen3DecoderLayer']


    def __init__(self, config: AndesVLConfig):
        super().__init__(config)
        
        self.config = config
        self.vision_encoder = Aimv2VisionModel(config.vision_config)
        self.language_model = Qwen3ForCausalLM(config.text_config)
        
        vit_hidden_size = self.vision_encoder.config.hidden_size
        llm_hidden_size = self.language_model.config.hidden_size
        self.patch_size = self.vision_encoder.config.patch_size
        self.mlp = nn.Sequential(
            nn.Linear(vit_hidden_size * 4, vit_hidden_size * 4),
            nn.GELU(),
            nn.Linear(vit_hidden_size * 4, llm_hidden_size),
        )

    def get_input_embeddings(self):
        return self.language_model.model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.language_model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.language_model.lm_head = new_embeddings

    def get_flated_pixel_values(self, pixel_values):
        flated_pixel_values = []
        image_grid_hw = []
        for pv in pixel_values:
            c, h, w = pv.shape
            assert c==3 and h%self.patch_size==0 and w%self.patch_size==0, f"{c}, {w}, {h}, {self.patch_size}"
            image_grid_hw.append((h//self.patch_size, w//self.patch_size))
            fpv = pv.reshape(c, h//(2*self.patch_size), 2, self.patch_size, w//(2*self.patch_size), 2, self.patch_size)
            flated_pixel_values.append(fpv.permute(1, 4, 2, 5, 0, 3, 6).reshape(-1, c*self.patch_size*self.patch_size))
        flated_pixel_values = torch.cat(flated_pixel_values, dim=0) # (Len_img, C, H, W)
        image_grid_hw = torch.tensor(image_grid_hw, device=flated_pixel_values.device) # (N_img, 2)
        return flated_pixel_values, image_grid_hw


    def get_vit_embeds_and_merge(self, pixel_values, image_grid_hw, input_embeds, image_flags):
        """
        Args:
            pixel_values: (Len_img, H_vit0)， 拉平后的初始patch特征，按照序列维度拼接在一起
            image_grid_hw: (N_img, 2)， 每个图片的宽高
            input_embeds: (Bt, Lt, Ht)， 每个token的embedding
            image_flags: (Bt, Lt)， 每个token是否是图片
        """
        vit_embeds = self.vision_encoder(pixel_values, image_grid_hw)  # (Len_img, H_vit)
        vit_embeds = vit_embeds.view(-1, vit_embeds.shape[-1]*4) # (Len_img//4, H_vit*4)
        vit_embeds = self.mlp(vit_embeds) # (Len_img//4, H_llm)
        vit_embeds = vit_embeds[:image_flags.sum()]
        Bt, Lt, Ht = input_embeds.shape
        input_embeds = input_embeds.reshape(-1, Ht)
        image_flags = image_flags.view(-1)
        input_embeds[image_flags == 1] = vit_embeds
        input_embeds = input_embeds.view(Bt, Lt, Ht)
        return input_embeds

    def forward(
            self,
            pixel_values=None,
            input_ids=None,
            attention_mask=None,
            image_flags=None,
            labels=None,
            return_dict=None,
            **kwargs
    ):
        """
        前向传播函数，用于训练和微调

        Args:
            pixel_values: 图像的像素值列表
            input_ids: 输入token ids (Bt, Lt)
            attention_mask: 注意力掩码 (Bt, Lt)
            image_flags: 图像标志位 (Bt, Lt)，标记哪些位置是图像token
            labels: 标签token ids (Bt, Lt)，用于计算loss
            return_dict: 是否返回字典格式

        Returns:
            包含loss和logits的输出
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # 获取输入的embeddings
        input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # 如果有图像输入，处理图像并融合到embeddings中
        if pixel_values is not None and image_flags is not None and (image_flags == 1).sum() > 0:
            flated_pixel_values, image_grid_hw = self.get_flated_pixel_values(pixel_values)
            input_embeds = self.get_vit_embeds_and_merge(
                flated_pixel_values, image_grid_hw, input_embeds, image_flags
            )

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=return_dict,
            **kwargs
        )
        return outputs

    @torch.inference_mode()
    @torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    def generate(
        self,
        pixel_values=None,
        input_ids=None,
        attention_mask=None,
        image_flags=None,  # (Bt, Lt)
        generation_config=None,
        **generate_kwargs,
    ) -> torch.LongTensor:

        input_embeds = self.language_model.get_input_embeddings()(input_ids)  # (Bt, Lt, Ht)
        if image_flags != None and (image_flags == 1).sum() > 0:
            flated_pixel_values, image_grid_hw = self.get_flated_pixel_values(pixel_values)
            input_embeds = self.get_vit_embeds_and_merge(flated_pixel_values, image_grid_hw, input_embeds, image_flags)

        print(f"input_ids shape: {input_ids.shape}")
        # print(f"labels shape: {labels.shape}")
        print(f"attention_mask shape: {attention_mask.shape}")

        outputs = self.language_model.generate(
            input_ids=input_ids,
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            use_cache=True,
            **generate_kwargs,
        )
        return outputs
    
    #NOTE: completion和chat接口暂不支持batch推理，需要手动构建self.generate函数的输入来实现。
    def completion(self, prompt, images, tokenizer, image_processor, **kwargs):
        """输入一段文字和一组图片（其中文字中的图片用占位符标记为<image>），输出补全的文本"""
        assert prompt.count("<image>") == len(images), "图片数量和占位符数量不匹配"
        def replacement(m):
            token_count = image_tokens.pop(0)
            return f"<img>{'<|vision_pad|>' * token_count}</img>"
        #首先对所有的图像进行处理，获取对应的size
        max_size = kwargs.get("max_size", 733) # max_size**2为支持的最大的面积
        base = self.patch_size*2
        image_token_id = tokenizer.vocab['<|vision_pad|>'] # 图像token的占位符
        background_color = tuple(int(x*255) for x in image_processor.image_mean)
        transform = T.Compose([T.ToTensor(),T.Normalize(mean=image_processor.image_mean, std=image_processor.image_std)])
        pixel_values = []
        image_tokens = []
        for image in images:
            if isinstance(image, (tuple, list)):
                image, detail = image
            else:
                detail = "low"
            image = load_image(image)
            if detail=="low":
                image = native_preprocess(image, max_size, base, background_color, min_tokens=4)
                pixel_values.append(transform(image))
                image_tokens.append(image.size[0]*image.size[1]//(base*base))
            else:
                raise NotImplementedError("暂未实现")
        new_prompt = re.sub(r"<image>", replacement, prompt)
        input_ids = tokenizer(new_prompt, return_tensors="pt", add_special_tokens=False).input_ids.to(self.device)
        image_flags = (input_ids == image_token_id).int()
        input_ids = input_ids.to(self.vision_encoder.device)
        pixel_values = [pv.to(self.vision_encoder.device) for pv in pixel_values]
        image_flags = image_flags.to(self.vision_encoder.device)
        output_ids = self.generate(pixel_values=pixel_values, input_ids=input_ids, image_flags=image_flags, **kwargs)[0][input_ids.shape[1]:]
        return tokenizer.decode(output_ids, skip_special_tokens=True)
            
    def chat(self, messages, tokenizer, image_processor, **kwargs):
        """输入是一组对话信息（openai格式），输出是回复"""
        prompt = ""
        images = []
        for message in messages:
            role = message["role"]
            assert role in ["user", "assistant", "system"], f"非法的角色{role}"
            content = message['content']
            if isinstance(content, str):
                prompt += f"<|im_start|>{role}\n{content}{tokenizer.eos_token}\n"
            elif isinstance(content, list):
                temp = ""
                for sub_content in content:
                    if sub_content['type']=='text':
                        temp += f"{sub_content['text']}"
                    elif sub_content['type']=='image_url':
                        temp += "<image>"
                        images.append([load_image(sub_content['image_url']['url']), sub_content['image_url'].get("detail",'low')])
                    elif sub_content['type']=='image':
                        temp += "<image>"
                        images.append(load_image(sub_content['image']))
                prompt += f"<|im_start|>{role}\n{temp}{tokenizer.eos_token}\n"
            else:
                raise ValueError(f"非法的内容{content}")
        prompt += f"<|im_start|>assistant\n"
        # return self.completion(prompt, images, tokenizer, image_processor, **kwargs)
        thinking = 'thinking' in kwargs and kwargs['thinking']
        if 'thinking' in kwargs:
            kwargs.pop('thinking')
        prompt += f"<|im_start|>assistant\n" + ('<think>' if thinking else '')
        return ('<think>' if thinking else '') + self.completion(prompt, images, tokenizer, image_processor, **kwargs)

########################
###下面是图像处理的代码###
########################

import os
import math
import re
from typing import Union
import requests
import base64
from io import BytesIO
from PIL import Image
import torchvision.transforms as T

def load_image(source: Union[str, Image.Image]) -> Image.Image:
    """加载图像"""
    if isinstance(source, Image.Image):
        img = source
    elif isinstance(source, str):
        if source.startswith('http'):
            response = requests.get(source)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
        elif os.path.exists(source):
            img = Image.open(source)
        elif source.startswith('data:image'):
            img = Image.open(BytesIO(base64.b64decode(source.split(',')[1])))
        else:
            raise ValueError("Unsupported image source")
    else:
        raise ValueError("Unsupported image source")
    return img.convert('RGB')

def get_scaled_img_size(image_size, max_area, base, max_resolution=4172, upper=True):
    """计算缩放后的图片大小和包裹矩形的大小"""
    # 计算原始图片的宽高比
    aspect_ratio = image_size[0] / image_size[1]
    # 计算包裹矩形的最大可能宽度和高度
    max_width = math.floor(math.sqrt(max_area * aspect_ratio))
    max_height = math.floor(math.sqrt(max_area / aspect_ratio))
    max_width, max_height = min(max_width, max_resolution), min(
        max_height, max_resolution
    )
    max_width, max_height = max(max_width, base), max(max_height, base)
    # 确保包裹矩形的宽度和高度都是base的整数倍
    if not upper:
        # 向下取整, 保证面积不会超过max_area
        max_width = max_width - max_width % base
        max_height = max_height - max_height % base
    else:
        # 向上取整，同时不超过max_resolution（单边最大长度）
        max_width = min(max_width + (base - max_width % base), max_resolution)
        max_height = min(max_height + (base - max_height % base), max_resolution)
    # 计算缩放因子
    scale_factor = min(max_width / image_size[0], max_height / image_size[1])
    # 计算缩放后的图片大小
    new_image_size = (
        round(image_size[0] * scale_factor),
        round(image_size[1] * scale_factor),
    )
    # 计算包裹矩形的大小
    bounding_box_size = (max_width, max_height)
    return new_image_size, bounding_box_size


def max_preprocess(
    img, max_size, base, background_color, max_resolution=4172, upper=True, force_resize=False
):
    """对图片进行预处理，使其面积接近max_size**2"""
    # 首先把图片resize到长度和宽度都低于max_resolution
    w, h = img.size
    if max(w, h) > max_resolution:
        scale = max_resolution / max(w, h)
        w, h = int(w * scale), int(h * scale)
    # 获取缩放后的图片大小和包裹矩形的大小
    new_image_size, bounding_box_size = get_scaled_img_size(
        (w, h), max_size**2, base, max_resolution, upper
    )
    if force_resize:
        return img.resize(bounding_box_size)
    # 创建一个新的画布
    canvas = Image.new("RGB", bounding_box_size, background_color)
    # 计算将图像粘贴到画布上的位置
    paste_width = (bounding_box_size[0] - new_image_size[0]) // 2
    paste_height = (bounding_box_size[1] - new_image_size[1]) // 2
    # 将图像粘贴到画布上
    canvas.paste(img.resize(new_image_size), (paste_width, paste_height))
    return canvas

def native_preprocess(
    img, max_size, base, background_color, max_resolution=4172, min_tokens=64
):
    # 对图片进行处理，使其宽度和高度都是base的整数倍
    # 如果图片的最长边超过max_resolution，就把图片resize到max_resolution以内
    w, h = img.size
    # 首先保证图片的最长边不超过max_resolution(ViT在极限长度)
    if max(w, h) > max_resolution:
        scale = max_resolution / max(w, h)
        w, h = int(w * scale), int(h * scale)
        img = img.resize((w, h))
    if w * h > max_size**2:
        return max_preprocess(img, max_size, base, background_color, max_resolution)
    if w * h < (base * base * min_tokens):
        return max_preprocess(
            img,
            int(base * (min_tokens**0.5)),
            base,
            background_color,
            max_resolution,
        )  
    w1, h1 = w + base - w % base, h + base - h % base
    if w1 == w and h1 == h:
        return img
    else:
        # 创建一个新的(w1, h1)的画布，并把图片resize保证只有一侧存在白边的情况
        scale = min(w1 / w, h1 / h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h))
        canvas = Image.new("RGB", (w1, h1), background_color)
        canvas.paste(img, ((w1 - new_w) // 2, (h1 - new_h) // 2))
        return canvas