from .Mars import *
from .Duke import *
from .LSVID import *

__factory = {
    'mars': Mars,
    'duke':DukeMTMCVidReID,
    'lsvid':LSVID,
}

def get_names():
    return __factory.keys()

def init_dataset(name, *args, **kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown dataset: {}".format(name))
    return __factory[name](*args, **kwargs)