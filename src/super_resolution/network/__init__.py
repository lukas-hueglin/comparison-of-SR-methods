### __init__.py ###

# The functions for building a network
from .architectures import make_SRResNet_4x
from .architectures import make_SRResNet_2x
from .architectures import make_SRGAN_disc
from .architectures import make_Demo

# The functoins defining a loss function
from .loss_functions import build_SRGAN_loss
from .loss_functions import build_SRGAN_Limited_loss
from .loss_functions import build_gen_loss
from .loss_functions import build_disc_loss
from .loss_functions import build_MSE_loss