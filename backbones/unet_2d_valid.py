import logging
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """ConvBlock.

    Args:

        in_channels (``int``):

            The number of input channels for the first convolution in the
            block.

        out_channels (``int``):

            The number of output channels for all convolutions in the
            block.

        kernel_sizes (``tuple`` of ``tuple``):

            The number of tuples determines the number of
            convolutional layers in each ConvBlock. If not given, each
            ConvBlock will consist of two 3x3 convolutions.

        activation (``str``):

            Which activation to use after a convolution. Accepts the name
            of any torch activation function (e.g., ``ReLU`` for
            ``torch.nn.ReLU``).

        batch_norm (``bool``):

            If set to ``True``, apply 2d batch normalization after each
            convolution.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes,
        activation,
        batch_norm=False,
    ):

        super().__init__()

        layers = []

        for k in kernel_sizes:
            layers.append(nn.Conv2d(in_channels, out_channels, k))
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(getattr(nn, activation)())

            in_channels = out_channels

        self.conv_block = nn.Sequential(*layers)

    def forward(self, x):

        return self.conv_block(x)


class Unet2dValid(nn.Module):
    """Unet for 2d inputs with valid padding.

    Input tensors are expected to be of shape ``(B, C, H, W)``.
    This model includes a 1x1Conv-head to return the desired
    number of out_channels.

    Args:

        in_channels:

            The number of input channels.

        inital_fmaps:

            The number of feature maps in the first layer. This is also the
            number of output feature maps. Stored in the ``channels``
            dimension.

        fmap_inc_factor:

            By how much to multiply the number of feature maps between
            layers. If layer 0 has ``k`` feature maps, layer ``l`` will
            have ``k*fmap_inc_factor**l``.

        downsample_factors:

            Tuple of tuples ``(x, y)`` to use to down- and up-sample the
            feature maps between layers.

        out_channels:

            The number of output_channels of the head.

        kernel_sizes (optional):

            Tuple of tuples. The number of tuples determines the number of
            convolutional layers in each ConvBlock. If not given, each
            ConvBlock will consist of two 3x3 convolutions.

        activation (optional):

            Which activation to use after a convolution. Accepts the name
            of any tensorflow activation function (e.g., ``ReLU`` for
            ``torch.nn.ReLU``).

        batch_norm (optional):

            If set to ``True``, apply 2d batch normalization after each
            convolution in the ConvBlocks.
    """

    def __init__(
        self,
        in_channels,
        initial_fmaps,
        fmap_inc_factor,
        downsample_factors,
        out_channels,
        kernel_sizes=((3, 3), (3, 3)),
        activation='LeakyReLU',
        constant_upsample=True,
        batch_norm=False,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.initial_fmaps = initial_fmaps

        if not isinstance(fmap_inc_factor, int):
            raise ValueError(
                "Feature map increase factor has to be integer.")
        self.fmap_inc_factor = fmap_inc_factor

        for d in downsample_factors:
            if not isinstance(d, tuple):
                raise ValueError(
                    "Downsample factors have to be a list of tuples.")
        self.downsample_factors = downsample_factors
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.activation = activation
        self.constant_upsample = constant_upsample
        self.batch_norm = batch_norm

        self.levels = len(downsample_factors) + 1

        # left convolutional passes
        self.l_conv = nn.ModuleList([
            ConvBlock(
                in_channels if level == 0 else initial_fmaps *
                fmap_inc_factor**(level - 1),
                initial_fmaps * fmap_inc_factor**level,
                kernel_sizes,
                activation,
                batch_norm
            )
            for level in range(self.levels)
        ])

        # left downsample layers
        self.l_down = nn.ModuleList([
            nn.MaxPool2d(
                kernel_size=downsample_factors[level],
                stride=downsample_factors[level]
            ) for level in range(self.levels - 1)
        ])

        # right upsample layers
        if constant_upsample:
            self.r_up = nn.ModuleList([
                nn.Upsample(
                    scale_factor=downsample_factors[level],
                    mode='nearest'
                )
                for level in range(self.levels - 1)
            ])
        else:
            self.r_up = nn.ModuleList([
                nn.ConvTranspose2d(
                    in_channels=initial_fmaps * fmap_inc_factor**(level + 1),
                    out_channels=initial_fmaps * fmap_inc_factor**(level + 1),
                    kernel_size=downsample_factors[level],
                    stride=downsample_factors[level],
                )
                for level in range(self.levels - 1)
            ])

        # right convolutional passes
        self.r_conv = nn.ModuleList([
            ConvBlock(
                initial_fmaps * fmap_inc_factor**level +
                initial_fmaps * fmap_inc_factor**(level + 1),
                initial_fmaps * fmap_inc_factor**level
                if level != 0 else max(initial_fmaps, out_channels),
                kernel_sizes,
                activation,
                batch_norm
            )
            for level in range(self.levels - 1)
        ])

        # 1x1 conv to map to the desired number of output channels
        self.head = nn.Conv2d(
            max(initial_fmaps, out_channels), out_channels, 1)

        # Initialize all Conv2d with Kaiming init
        def init_kaiming(m):
            if self.activation == 'ReLU':
                nonlinearity = 'relu'
            elif self.activation == 'LeakyReLU':
                nonlinearity = 'leaky_relu'
            else:
                raise ValueError(
                    f"Kaiming init not applicable for activation {self.activation}.")
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    nonlinearity=nonlinearity)
                nn.init.zeros_(m.bias)

        if activation in ('ReLU', 'LeakyReLU'):
            self.apply(init_kaiming)
            logger.debug("Initialize conv weights with Kaiming init.")

    def forward(self, x):

        if not self.is_valid_input_size(x.shape[2:]):
            raise ValueError((
                f"Input size {x.shape[2:]} is not valid for"
                " this Unet instance."
            ))

        if self.batch_norm and x.shape[0] == 1:
            raise ValueError((
                "This Unet performs batch normalization, "
                "therefore inputs with batch size 1 are not allowed."
            ))

        features = self.rec_forward(self.levels - 1, x)
        return self.head(features)

    def rec_forward(self, level, f_in):

        # index of level in layer arrays
        i = self.levels - level - 1

        # convolve
        l_conv = self.l_conv[i](f_in)

        # end of recursion
        if level == 0:

            out = l_conv

        else:

            # down
            l_down = self.l_down[i](l_conv)

            # nested levels
            r_in = self.rec_forward(level - 1, l_down)

            # up
            r_up = self.r_up[i](r_in)

            # center crop l_conv
            l_conv_cropped = self.crop(l_conv, r_up.shape[-2:])

            # concat
            r_concat = torch.cat([l_conv_cropped, r_up], dim=1)

            # convolve
            out = self.r_conv[i](r_concat)

        return out

    def crop(self, x, shape):
        """Center-crop x to match spatial dimensions given by shape."""

        x_target_size = x.size()[:-2] + shape

        offset = tuple(
            (a - b) // 2
            for a, b in zip(x.size(), x_target_size))

        slices = tuple(
            slice(o, o + s)
            for o, s in zip(offset, x_target_size))

        return x[slices]

    def valid_input_sizes_seq(self, n):

        sizes = []
        for i in range(n + 1):
            if self.is_valid_input_size(i):
                sizes.append(i)

        return sizes

    def is_valid_input_size(self, size):

        assert np.all(np.array(self.kernel_sizes) % 2 == 1)

        size = np.array(size, dtype=np.int_)
        ds_factors = [np.array(x, dtype=np.int_)
                      for x in self.downsample_factors]
        kernel_sizes = np.array(self.kernel_sizes, dtype=np.int_)

        def rec(level, s):
            # index of level in layer arrays
            i = self.levels - level - 1
            for k in kernel_sizes:
                s = s - (k - 1)
                if np.any(s < 1):
                    return False
            if level == 0:
                return s
            else:
                # down
                if np.any(s % 2 == 1):
                    return False
                s = s // ds_factors[i]
                s = rec(level - 1, s)
                # up
                s = s * ds_factors[i]
                for k in kernel_sizes:
                    s = s - (k - 1)
                    if np.any(s < 1):
                        return False

            return s

        out = rec(self.levels - 1, size)
        if out is not False:
            out = True
        return out
