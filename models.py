import paddle
from x2paddle import torch2paddle
import math
from paddle import nn
from paddleseg.cvlibs import param_init

class ESPCN(nn.Layer):

    def __init__(self, scale_factor, num_channels=1):
        super(ESPCN, self).__init__()
        self.first_part = nn.Sequential(
            nn.Conv2D(num_channels, 64,kernel_size=5, padding=5 // 2), 
            nn.Tanh(), 
            nn.Conv2D(64, 32,kernel_size=3, padding=3 // 2), 
            nn.Tanh())
        
        self.last_part = nn.Sequential(
            nn.Conv2D(32, num_channels * scale_factor ** 2, kernel_size=3, padding=3 // 2),
            nn.PixelShuffle(scale_factor)
            )
        

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2D):
                if m.in_channels == 32:
                    param_init.normal_init(m.weight.data, mean=0.0, std=0.001)
                    param_init.constant_init(m.bias.data,value=0)
                else:
                    param_init.normal_init(m.weight.data,mean=0.0,std=math.sqrt(2 / (m.out_channels * m.weight.data[0][0].numel())))
                    param_init.constant_init(m.bias.data,value=0)

    def forward(self, x):
        x = self.first_part(x)
        x = self.last_part(x)
        return x
