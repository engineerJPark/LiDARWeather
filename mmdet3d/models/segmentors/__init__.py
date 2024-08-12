# Copyright (c) OpenMMLab. All rights reserved.
from .base import Base3DSegmentor
from .cylinder3d import Cylinder3D
from .encoder_decoder import EncoderDecoder3D
from .lasermix import LaserMix
from .minkunet import MinkUNet
from .seg3d_tta import Seg3DTTAModel
from .semi_base import SemiBase3DSegmentor

### new added
from .minkunet_weather_dropper import MinkUNetWeatherDropper

__all__ = [
    'Base3DSegmentor', 'EncoderDecoder3D', 'Cylinder3D', 'MinkUNet',
    'Seg3DTTAModel', 'SemiBase3DSegmentor', 'LaserMix',
    ### new added
    'MinkUNetWeatherDropper'
]