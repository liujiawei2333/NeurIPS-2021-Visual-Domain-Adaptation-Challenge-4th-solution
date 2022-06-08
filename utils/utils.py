from operator import ne
from models.basenet import *
import os
import torch
import socket
from efficientnet_pytorch import EfficientNet


def get_model_mme(net, num_class=13, temp=0.05, top=False, norm=True):
    dim = 2048
    if "resnet" in net:
        model_g = ResBase(net, top=top)
        if "resnet18" in net:
            dim = 512
        if net == "resnet34":
            dim = 512
    elif "resnext" in net:
        model_g = ResBase(net, top=top)
        dim = 2048
    elif "efficient" in net:
        model_g = EfficientPlus2CNN(net)
        dim = 2048
    elif "effiwhole" in net:
        model_g = EfficientNet.from_pretrained('efficientnet-b7', num_classes=2048)
        dim = 2048
    elif "adveffi-b8" in net:
        model_g = EfficientNet.from_pretrained('efficientnet-b8', num_classes=2048, advprop=True)
        dim = 2048
    elif "adveffi-b6" in net:
        model_g = EfficientNet.from_pretrained('efficientnet-b6', num_classes=500, advprop=True)
        dim = 500
    elif "adveffi-b5" in net:
        model_g = EfficientNet.from_pretrained('efficientnet-b5', num_classes=500, advprop=True)
        dim = 500
    elif "adveffi-b4" in net:
        model_g = EfficientNet.from_pretrained('efficientnet-b4', num_classes=500, advprop=True)
        dim = 500
    elif "adveffi-b3" in net:
        model_g = EfficientNet.from_pretrained('efficientnet-b3', num_classes=2048, advprop=True)
        dim = 2048
    elif "vgg" in net:
        model_g = VGGBase(option=net, pret=True, top=top)
        dim = 4096
    if top:
        dim = 1000
    print("selected network %s"%net)
    return model_g, dim

def log_set(kwargs, name=None):
    source_data = kwargs["source_data"]
    target_data = kwargs["target_data"]
    network = kwargs["network"]
    conf_file = kwargs["config_file"]
    script_name = kwargs["script_name"]
    multi = kwargs["multi"]
    #args = kwargs["args"]

    target_data = os.path.splitext(os.path.basename(target_data))[0]

    if name is not None:
        logname = name
    else:
        logname = "{file}_{source}2{target}_{network}_hp_{hp}".format(file=script_name.replace(".py", ""),
                                                                               source="imagenet-1k",
                                                                               target=target_data,
                                                                               network=network,
                                                                               hp=str(multi))
    logname = os.path.join("record", kwargs["exp_name"],
                           os.path.basename(conf_file).replace(".yaml", ""), logname)
    if not os.path.exists(os.path.dirname(logname)):
        os.makedirs(os.path.dirname(logname))
    print("record in %s " % logname)
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(filename=logname, format="%(message)s")
    logger.setLevel(logging.INFO)
    logger.info("{}_2_{}".format(source_data, target_data))
    return logname


def save_model(model_g, model_c1, model_c2, save_path):
    save_dic = {
        'g_state_dict': model_g.state_dict(),
        'c1_state_dict': model_c1.state_dict(),
        'c2_state_dict': model_c2.state_dict(),
    }
    torch.save(save_dic, save_path)

def load_model(model_g, model_c, load_path):
    checkpoint = torch.load(load_path)
    model_g.load_state_dict(checkpoint['g_state_dict'])
    model_c.load_state_dict(checkpoint['c_state_dict'])
    return model_g, model_c

def load_model2(model_g, model_c1, model_c2, load_path):
    checkpoint = torch.load(load_path)
    model_g.load_state_dict(checkpoint['g_state_dict'])
    model_c1.load_state_dict(checkpoint['c1_state_dict'])
    model_c2.load_state_dict(checkpoint['c2_state_dict'])
    return model_g, model_c1, model_c2