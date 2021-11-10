import pytest
import numpy as np
import torch
from torch.nn import init
from backbones import Unet2d


@pytest.mark.parametrize("downsample_factors", [[(4, 4)], [(3, 3), (2, 2)]])
@pytest.mark.parametrize("constant_upsample", [False, True])
def test_unet_2d_valid_output_ones(downsample_factors, constant_upsample):

    m = Unet2d(
        in_channels=3,
        initial_fmaps=4,
        fmap_inc_factor=2,
        downsample_factors=downsample_factors,
        out_channels=3,
        constant_upsample=constant_upsample,
        batch_norm=False,
        padding=0,
    )
    sizes = m.valid_input_sizes_seq(100)

    for s in sizes:
        x = torch.ones(2, 3, s, s)
        for l in m.modules():
            if isinstance(l, torch.nn.Conv2d):
                init.constant_(l.weight,
                               1 / (l.in_channels * np.prod(l.kernel_size)))
                init.zeros_(l.bias)

            if isinstance(l, torch.nn.ConvTranspose2d):
                init.constant_(l.weight, 1 / l.in_channels)
                init.zeros_(l.bias)

        out = m(x)

        assert torch.all(torch.isclose(out, torch.tensor(1.0))), f"{out=}"


@pytest.mark.parametrize("downsample_factors", [[(4, 4)], [(3, 3), (2, 2)]])
@pytest.mark.parametrize("constant_upsample", [False, True])
def test_unet_2d_padding(downsample_factors, constant_upsample):

    m = Unet2d(
        in_channels=3,
        initial_fmaps=4,
        fmap_inc_factor=2,
        downsample_factors=downsample_factors,
        out_channels=3,
        kernel_sizes=((3, 3), (3, 3)),
        constant_upsample=constant_upsample,
        batch_norm=False,
        padding=1,
        padding_mode='replicate',
    )
    sizes = m.valid_input_sizes_seq(100)
    for s in sizes:
        x = torch.ones(2, 3, s, s)
        for l in m.modules():
            if isinstance(l, torch.nn.Conv2d):
                init.constant_(l.weight,
                               1 / (l.in_channels * np.prod(l.kernel_size)))
                init.zeros_(l.bias)

            if isinstance(l, torch.nn.ConvTranspose2d):
                init.constant_(l.weight, 1 / l.in_channels)
                init.zeros_(l.bias)

        out = m(x)

        assert torch.all(torch.isclose(out, torch.tensor(1.0))), f"{out=}"
        assert out.shape[-2:] == x.shape[-2:]
