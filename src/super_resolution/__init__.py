### __init__.py ###

from .model import Model

from .methods import Method
from .methods import AdversarialNetwork
from .methods import SingleNetwork

from .upsampling import bicubic
from .upsampling import bilinear
from .upsampling import lanczos
from .upsampling import Framework
from .upsampling import PreUpsampling
from .upsampling import ProgressiveUpsampling

from .presets import build_SRGAN
from .presets import build_SRResNet

from pipeline import Pipeline