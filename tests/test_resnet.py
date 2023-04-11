import logging
import pytest
import numpy as np
import torch
from torch.nn import init
from backbones import Resnet

logger = logging.getLogger(__name__)


@pytest.mark.parametrize("downsample_factors,kernel_sizes", [
    (((2, 2, 2),), (((3, 3, 3),),)),
    (((1, 2, 2), (3, 3, 3)), (((1, 3, 3), (3, 3, 3)), ((1, 3, 3),),)),
])
@ pytest.mark.parametrize("batch_norm", (False, True))
def test_resnet_valid_output_ones(downsample_factors, kernel_sizes, batch_norm):

    m = Resnet(
        in_channels=3,
        initial_fmaps=4,
        downsample_factors=downsample_factors,
        out_features=5,
        kernel_sizes=kernel_sizes,
        batch_norm=batch_norm,
        padding=1,
    )
    sizes = (12, 42, 79)

    for s in sizes:
        x = torch.ones(2, 3, s, s, s)
        for l in m.modules():
            if isinstance(l, torch.nn.Conv3d):
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


@pytest.mark.parametrize("downsample_factors,kernel_sizes", [
    (((2, 2, 2),), (((3, 3, 3),),)),
    (((1, 2, 2), (3, 3, 3)), (((1, 3, 3), (3, 3, 3)), ((1, 3, 3),),)),
])
def test_resnet_fc(downsample_factors, kernel_sizes):
    m = Resnet(
        in_channels=3,
        initial_fmaps=4,
        downsample_factors=downsample_factors,
        out_features=5,
        kernel_sizes=kernel_sizes,
        batch_norm=False,
        fully_convolutional=True,
    )
    sizes = (12, 42, 79)

    for s in sizes:
        x = torch.ones(2, 3, s, s, s)
        for l in m.modules():
            if isinstance(l, torch.nn.Conv3d):
                init.constant_(l.weight, 1 / (l.in_channels * np.prod(l.kernel_size)))
                init.zeros_(l.bias)

        out = m(x)

        # 2**n_blocks without batch norm, due to addition of shortcut features
        assert torch.all(
            torch.isclose(
                out,
                torch.tensor(2.0 ** len(downsample_factors)),
            )
        ), f"{out=}"


if __name__ == "__main__":
    downsample_factors = ((1, 2, 2), (2, 2, 2))
    kernel_sizes = (((1, 3, 3),), ((3, 3, 3),),)
    batch_norm = True
    test_resnet_valid_output_ones(downsample_factors, kernel_sizes, batch_norm)
    # test_resnet_fc(downsample_factors, kernel_sizes)
