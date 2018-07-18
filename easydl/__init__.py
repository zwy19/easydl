import sys
import os
import traceback

sys.path.append(os.path.abspath('./'))
from common import *

try:
    import torch
    from pytorch import *
except ImportError as e:
    print('pytorch not available!')
    traceback.print_exc()

try:
    import tensorflow
    from tf import *
except ImportError as e:
    print('tensorflow not available!')
    traceback.print_exc()

import warnings

warnings.filterwarnings('ignore', '.*')