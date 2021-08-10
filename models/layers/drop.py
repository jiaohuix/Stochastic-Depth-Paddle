import paddle
import paddle.nn as nn

def drop_path(x,act_layer,drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    assert drop_prob>=0 and drop_prob<=1
    if not training: #  如果是test，则直接输出激活输出  f(x),否则依概率失活
        return act_layer(x)
    else: # 如果有存活概率pl，则bl为p=pl的伯努利随机变量，bl=1时x被激活，反之返回x
        keep_prob = paddle.to_tensor(1 - drop_prob) # pl 第l层的存活概率
        bl=paddle.bernoulli(keep_prob)
        if bl:
            return act_layer(x)
        else:
            return 0
        # return bl*act_layer(x)

class StochasticLayer(nn.Layer):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, act_layer,drop_prob=None):
        super(StochasticLayer, self).__init__()
        self.act_layer=act_layer
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x,self.act_layer,self.drop_prob, self.training) # self.training可以获取模型的训练与否
