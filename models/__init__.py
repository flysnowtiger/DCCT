from __future__ import absolute_import

from models.ViTBase import ViTBase
from models.ResNet50 import ResNet50
from models.DCCT import DeepCoupledCnnTransformer

__factory = {
    'ResNet50' : ResNet50,
    'ViTBase' : ViTBase,
    'DCCT': DeepCoupledCnnTransformer,
}

def get_names():
    return __factory.keys()

def init_model(name,*args,**kwargs):
    if name not in __factory.keys():
        raise KeyError("Unknown model :{}".format(name))
    return __factory[name](*args,**kwargs)