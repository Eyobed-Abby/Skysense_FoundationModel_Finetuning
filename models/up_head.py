import torch.nn as nn
from mmcv.cnn.utils.weight_init import (kaiming_init, trunc_normal_)


class UPHead(nn.Module):

    def __init__(self, in_dim, out_dim, up_scale):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels=in_dim,
                      out_channels=up_scale**2 * out_dim,
                      kernel_size=1),
            nn.PixelShuffle(up_scale),
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            kaiming_init(m, mode='fan_in', bias=0.)

    def forward(self, x):
        x = self.decoder(x)
        return x