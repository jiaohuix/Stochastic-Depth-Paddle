'''
n=18时，res110，用于跑cifar10
'''
import paddle.nn as nn
from .layers import StochasticLayer,ChannelPad
from paddle.nn.initializer import Constant,KaimingNormal,TruncatedNormal

kaiming_normal_=KaimingNormal()
zeros_ = Constant(value=0.)
ones_ = Constant(value=1.)
trunc_normal_=TruncatedNormal(std=.02)

class ResidualBlock(nn.Layer):
    def __init__(self,inchannel,outchannel,stride=1,downsample=None,norm_layer=None,drop_prob=0.):
        super(ResidualBlock,self).__init__()
        if norm_layer is None:
            norm_layer=nn.BatchNorm2D
        fl=nn.Sequential( #第l层的function
            nn.Conv2D(inchannel,outchannel,3,stride,1,bias_attr=False), # 为啥不用bias
            norm_layer(outchannel),
            nn.ReLU(),
            nn.Conv2D(outchannel, outchannel, 3, 1, 1,bias_attr=False),
            norm_layer(outchannel))
        self.sd_layer = StochasticLayer(act_layer=fl,drop_prob=drop_prob)
        self.relu = nn.ReLU()
        self.downsample=downsample # 对x下采样保证与f(x)可以相加

    def forward(self,x):
        identity=x
        out=self.sd_layer(x)
        if self.downsample is not None:
            identity=self.downsample(identity)
        out+=identity
        return self.relu(out)

class ResNet(nn.Layer):
    def __init__(self,num_classes=10,norm_layer=None,num_blocks=18,P0=1,PL=0.5):
        super(ResNet,self).__init__()
        # calc drop prob
        L=num_blocks*3 # 总共多少个block
        self.num_blocks=num_blocks # 一层block数
        self.calc_drop_prob = lambda l: 1 - P0 + (l / L) * (1 - PL) # 失活概率
        if norm_layer is None:
            norm_layer=nn.BatchNorm2D
        self._norm_layer = norm_layer

        self.inplanes = 16
        self.pre=nn.Sequential(
            nn.Conv2D(in_channels=3,out_channels=16,kernel_size=3,stride=1,padding=1,bias_attr=False),
            norm_layer(16),
            nn.ReLU())
        self.layer1=self._make_layer(16,num_blocks)
        self.layer2=self._make_layer(32,num_blocks,stride=2)
        self.layer3=self._make_layer(64,num_blocks,stride=2)
        self.avgpool = nn.AdaptiveAvgPool2D((1, 1))
        self.fc = nn.Linear(64 , num_classes)
        self.apply(self._init_weights)

    def _init_weights(self,m):
            if isinstance(m, nn.Conv2D):
                kaiming_normal_(m.weight)
            elif isinstance(m,nn.Linear):
                trunc_normal_(m.weight)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2D, nn.GroupNorm)):
                ones_(m.weight)
                zeros_(m.bias)

    def _make_layer(self,planes,blocks,stride=1): # 添加droppath
        norm_layer = self._norm_layer
        downsample=None
        if stride!=1 or self.inplanes!=planes: # 如果要stride=2，且通道改变，对x进行下采样
            # downsample=nn.Sequential(
            #     nn.Conv2D(self.inplanes,planes,kernel_size=1,stride=stride,bias_attr=False),
            #     norm_layer(planes))
            downsample=nn.Sequential(
                nn.AvgPool2D(kernel_size=stride, stride=stride),
                ChannelPad()
            )

        layers=[]
        n=int(planes)//16  # 第n个layer
        l=(n-1)*self.num_blocks+1 # 第l个block，动态计算drop_prob,由于浅层抽取低级的信息，比较可靠，drop概率低
        layers.append(ResidualBlock(self.inplanes,planes,stride,downsample,norm_layer,drop_prob=self.calc_drop_prob(l)))
        self.inplanes=planes
        for i in range(1,blocks):
            layers.append(ResidualBlock(self.inplanes,planes,norm_layer=norm_layer,drop_prob=self.calc_drop_prob(l+i))) # 此处inplanes=planes

        return nn.Sequential(*layers)

    def forward(self, x):
        x=self.pre(x)
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=self.avgpool(x)
        x=x.flatten(1)
        x=self.fc(x)
        return x

