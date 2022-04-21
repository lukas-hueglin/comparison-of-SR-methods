### __init__.py ###

from .architectures import build_ResNet
from .architectures import build_SRGAN_disc
from .architectures import build_Demo

from .loss_functions import SRGAN_loss
from .loss_functions import gen_loss
from .loss_functions import disc_loss