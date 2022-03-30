import pytest
import numpy as np
import torch
from torch.nn import init
from backbones import Resnet2d


@pytest.mark.parametrize("downsample_factors", [(2, 2), (2, 1, 3)])
@pytest.mark.parametrize("batch_norm", (False, True))
def test_unet_2d_valid_output_ones(downsample_factors, batch_norm):

    m = Resnet2d(
        in_channels=3,
        initial_fmaps=4,
        downsample_factors=downsample_factors,
        out_features=5,
        batch_norm=batch_norm,
    )
    sizes = (12, 42, 79)

    for s in sizes:
        x = torch.ones(2, 3, s, s)
        for l in m.modules():
            if isinstance(l, torch.nn.Conv2d):
                init.constant_(l.weight, 1 / (l.in_channels * np.prod(l.kernel_size)))
                init.zeros_(l.bias)

            if isinstance(l, torch.nn.Linear):
                init.constant_(l.weight, 1 / l.in_features)
                init.zeros_(l.bias)

        out = m(x)

        # 0s with batch norm,
        # 2**n_blocks without batch norm, due to addition of shortcut features
        assert torch.all(
            torch.isclose(
                out,
                torch.tensor(2.0 ** len(downsample_factors) * float(not (batch_norm))),
            )
        ), f"{out=}"
