import os
from .convnext_encoder import ConvNeXtCLIPVisionTower


def build_vision_tower(vision_tower_cfg, **kwargs):
    vision_tower = getattr(vision_tower_cfg, 'mm_vision_tower', getattr(vision_tower_cfg, 'vision_tower', None))
    print(f"now we are building vision tower, the model is {vision_tower}")
    if 'convnext' in vision_tower:
        print(f'building ConvNeXtCLIPVisionTower')
        return ConvNeXtCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)
    return ConvNeXtCLIPVisionTower(vision_tower, args=vision_tower_cfg, **kwargs)

