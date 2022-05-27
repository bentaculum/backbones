import pytest
import numpy as np
import torch
from torch.nn import init
from backbones import Unet2d


@pytest.mark.parametrize("downsample_factors", [[(4, 4)], [(3, 3), (2, 2)]])
@pytest.mark.parametrize("constant_upsample", [False, True])
@pytest.mark.parametrize("crop_input", [False, True])
def test_unet_2d_valid_output_ones(downsample_factors, constant_upsample, crop_input):

    m = Unet2d(
        in_channels=3,
        initial_fmaps=4,
        fmap_inc_factor=2,
        downsample_factors=downsample_factors,
        out_channels=3,
        constant_upsample=constant_upsample,
        batch_norm=False,
        padding=0,
        pad_input=False,
        crop_input=crop_input,
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
@pytest.mark.parametrize("batch_norm", [False, True])
@pytest.mark.parametrize("group_norm", [False, 1, 2, 4])
@pytest.mark.parametrize("pad_input", [False, True])
def test_unet_2d_padding(
        downsample_factors,
        constant_upsample,
        batch_norm,
        group_norm,
        pad_input,
):

    if group_norm and batch_norm:
        return

    m = Unet2d(
        in_channels=3,
        initial_fmaps=4,
        fmap_inc_factor=2,
        downsample_factors=downsample_factors,
        out_channels=3,
        kernel_sizes=((3, 3), (3, 3)),
        constant_upsample=constant_upsample,
        batch_norm=batch_norm,
        group_norm=group_norm,
        padding=1,
        padding_mode='replicate',
        pad_input=pad_input,
    )

    if pad_input:
        sizes = [(a, b) for a, b in zip(range(80, 90, 1), range(90, 100, 1))]
    else:
        sizes = [
            (a, b) for a, b in zip(
                m.valid_input_sizes_seq(100), m.valid_input_sizes_seq(100)[1:])]

    for s in sizes:
        x = torch.ones(2, 3, s[0], s[1])
        for l in m.modules():
            if isinstance(l, torch.nn.Conv2d):
                init.constant_(l.weight,
                               1 / (l.in_channels * np.prod(l.kernel_size)))
                init.zeros_(l.bias)

            if isinstance(l, torch.nn.ConvTranspose2d):
                init.constant_(l.weight, 1 / l.in_channels)
                init.zeros_(l.bias)

            if isinstance(l, torch.nn.BatchNorm2d) or isinstance(
                    l, torch.nn.GroupNorm):
                init.zeros_(l.weight)
                init.ones_(l.bias)

        out = m(x)

        assert torch.all(torch.isclose(out, torch.tensor(1.0))), f"{out=}"
        assert out.shape[-2:] == x.shape[-2:]
