from torch import optim
from models.frostnet import FrostNet


def get_model(model_name):
    model_dict = {'FrostNet': FrostNet}
    return model_dict.get(model_name)


def get_optimizer(optimizer_name):
    optim_dict = {'sgd': optim.SGD, 'adam': optim.Adam}
    return optim_dict.get(optimizer_name)
