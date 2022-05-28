### __init__.py ###

# All the classes from model.py
from .model import Model

# All the Methods
from .methods import Method
from .methods import AdversarialNetwork
from .methods import SingleNetwork

# All the upsampling functions
from .upsampling import bicubic
from .upsampling import bilinear
from .upsampling import lanczos
from .upsampling import none

# All the Frameworks
from .upsampling import Framework
from .upsampling import PreUpsampling
from .upsampling import ProgressiveUpsampling

# All the presets
from .presets import build_SRGAN
from .presets import build_SRResNet
from .presets import build_SRDemo

# All the classes from pipeline.py
from .pipeline import Pipeline