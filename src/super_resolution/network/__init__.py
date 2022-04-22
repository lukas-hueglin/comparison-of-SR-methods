### __init__.py ###

# The functions for building a network
from .architectures import make_SRResNet
from .architectures import make_SRGAN_disc
from .architectures import make_Demo

# The functoins defining a loss function
from .loss_functions import SRGAN_loss
from .loss_functions import gen_loss
from .loss_functions import disc_loss