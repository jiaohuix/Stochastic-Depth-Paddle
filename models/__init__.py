import os
import paddle
from .resnet import ResNet
from utils import logger

def build_model(conf):
    num_blocks=(conf['model']['layers']-2)//6
    model=ResNet(num_classes=conf['data']['class_dim'],num_blocks=num_blocks,
                 P0=conf['hparas']['P0'],PL=conf['hparas']['PL'])

    #加载训练好的模型
    model_path = os.path.join(conf['hparas']["save_dir"], "final.pdparams")
    if os.path.exists(model_path):
        model.set_dict(paddle.load(model_path))
        logger.info('Prep | Model weights loaded!')
    return model
