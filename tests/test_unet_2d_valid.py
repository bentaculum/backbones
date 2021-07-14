import pytest
import numpy as np
import torch
from torch.nn import init
from backbones import Unet2dValid


@pytest.mark.parametrize("downsample_factors", [[(2, 4)], [(3, 3), (2, 2)]])
@pytest.mark.parametrize("constant_upsample", [False, True])
def test_unet_2d_valid_output_ones(downsample_factors, constant_upsample):

    m = Unet2dValid(
        in_channels=3,
        initial_fmaps=4,
        fmap_inc_factor=2,
        downsample_factors=downsample_factors,
        out_channels=3,
        activation='ReLU',
        constant_upsample=constant_upsample,
        batch_norm=False,
    )
    sizes = m.valid_input_sizes_seq(100)
    x = torch.ones(1, 3, sizes[-2], sizes[-1])

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
