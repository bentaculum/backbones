import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_, DropPath

class DownsamplingBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding=0, normalize=True):
        """Simple convolutional block with no non-linearities followed by a LayerNorm
        Args:
            in_channels (int): number of input channels
            out_channels (int): number of desired output channels
            kernel_size (int): convolutional kernel size
            stride (int): stride of the convolution
            padding (int, optional): padding added to all four sides of the input. Defaults to 0.
            normalize (bool, optional): whether to apply LayerNorm. Defaults to True.
        """
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding)
        self.normalize = normalize
        self.lnorm = nn.LayerNorm(out_channels, eps=1e-5) if self.normalize else None

    def forward(self, inp):
        x = self.conv(inp)
        if self.normalize:
            # LayerNorm requires channel to be last dim
            x = x.permute(0,2,3,1) # (N, C, H, W) -> (N, H, W, C)
            x = self.lnorm(x)
            x = x.permute(0,3,1,2) # (N, C, H, W) -> (N, H, W, C)
        return x

class ConvNeXtBlock(nn.Module):
    """ConvNeXt Block as described in the paper "A ConvNet for the 2020s" by Liu et al. (CVPR '22).
       Implementation from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py
    Args:
        in_channels (int): _description_
        drop_path (float, optional): _description_. Defaults to 0..
        layer_scale_init_value (float, optional): _description_. Defaults to 1e-6.
    """
    def __init__(self, in_channels, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(in_channels=in_channels,
                                out_channels=in_channels,
                                kernel_size=7,
                                padding=3,
                                groups=in_channels)
        self.pwconv1 = nn.Linear(in_channels, 4*in_channels)
        self.pwconv2 = nn.Linear(4*in_channels, in_channels)
        self.lnorm = nn.LayerNorm(in_channels, eps=1e-5)
        self.activation = nn.GELU()
        self.gamma = nn.Parameter(
            layer_scale_init_value*torch.ones(in_channels), requires_grad=True,
        ) if drop_path > 0. else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, inp):
        x_pr = self.dwconv(inp)
        x_pr = x_pr.permute(0,2,3,1) # (N, C, H, W) -> (N, H, W, C)
        x_pr = self.lnorm(x_pr)
        x_pr = self.activation(self.pwconv1(x_pr))
        x_pr = self.pwconv2(x_pr)
        if self.gamma is not None:
            x_pr = self.gamma * x_pr
        x_pr = x_pr.permute(0,3,1,2) # (N, H, W, C) -> (N, C, H, W)
        x_pr = inp + self.drop_path(x_pr)
        return x_pr

class ConvNeXt2d(nn.Module):
    """ConvNeXt backbone based in the architecture described in "A ConvNet for the 2020s" by Liu et al. (CVPR '22).
    Implementation adapted from https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py to allow
    constant simpler definitions, constant downsampling factors as well as outputs of feature maps at full resolution.
    Args:
        in_channels (int, optional): Expected number of input channels. Defaults to 1.
        levels (int, optional): Number of different resolutions levels/stages. Defaults to 3.
        downsample_factor (int, optional): Downsampling factor applied at the beginning of every stage. Defaults to 2.
        block_depth (int, optional): Number of ConvNeXt blocks per stage. Defaults to 3.
        n_channels_per_stage (list, optional): Number of output channels at every stage. Defaults to [4, 16, 32].
        drop_path_rate (float, optional): Stochastic depth rate. Defaults to 0.
        layer_scale_init_value (float, optional): Init value for Layer Scale. Defaults to 1e-6.
    """
    def __init__(self, in_channels=1, levels=3, downsample_factor=2, block_depth=3,
                 n_channels_per_stage=[4, 16, 32], drop_path_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.levels = levels
        self.downsample_factor = downsample_factor
        self.depths = self.levels*[block_depth] if isinstance(block_depth, int) else block_depth
        self.n_channels_per_stage = n_channels_per_stage
        assert self.levels == len(self.n_channels_per_stage), "Given levels and number of channels per stage parameters suggest different number of stages"

        self.downsample_layers = nn.ModuleList([
            DownsamplingBlock(in_channels=self.in_channels,
                              out_channels=self.n_channels_per_stage[0],
                              kernel_size=3,
                              stride=1,
                              padding="same",
                              normalize=True)
        ])
        self.downsample_layers += nn.ModuleList([
            DownsamplingBlock(in_channels=self.n_channels_per_stage[s],
                              out_channels=self.n_channels_per_stage[s+1],
                              kernel_size=self.downsample_factor,
                              stride=self.downsample_factor,
                              normalize=bool(s!=(self.levels-2)))
              for s in range(0, self.levels-1)
        ])
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for s in range(self.levels):
            stage = nn.Sequential(*[
                ConvNeXtBlock(in_channels=n_channels_per_stage[s],
                              drop_path=dp_rates[s+j],
                              layer_scale_init_value=layer_scale_init_value)
            for j in range(self.depths[s])])
            self.stages.append(stage)
            cur += self.depths[s]
        self.apply(self._init_weights)

    def _init_weights(self, layer):
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            trunc_normal_(layer.weight, std=.02)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        for s in range(self.levels):
            x = self.downsample_layers[s](x)
            x = self.stages[s](x)
        return x
