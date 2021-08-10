import paddle
import paddle.nn as nn

class ChannelPad(nn.Layer):
    def __init__(self):
        super(ChannelPad, self).__init__()

    def forward(self, x):
        zeros=paddle.zeros_like(x,dtype=x.dtype)
        x=paddle.concat([x,zeros],axis=1)
        return x