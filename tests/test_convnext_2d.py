import pytest
import numpy as np
import torch
from torch.nn import init
from backbones import ConvNeXt2d


@pytest.mark.parametrize("downsample_factor", [2, 4])
@pytest.mark.parametrize("n_channels_per_stage", [(4,16,64), (2,4,8,16,32)])
@pytest.mark.parametrize("in_channels", [1, 3])
def test_convnext_2d_valid_output_size(downsample_factor, n_channels_per_stage, in_channels):
    levels = len(n_channels_per_stage)
    m = ConvNeXt2d(
        in_channels=in_channels,
        levels=levels,
        downsample_factor=downsample_factor,
        n_channels_per_stage=n_channels_per_stage,
    )
    sizes = (256, 342, 512)

    for s in sizes:
        x = torch.ones(2, in_channels, s, s)
        with torch.no_grad():
            out = m(x)
        expected_out_dims = 2*[s//(downsample_factor**(levels-1))]
        assert out.shape == (2, n_channels_per_stage[-1], *expected_out_dims), f"{out.shape=}"
