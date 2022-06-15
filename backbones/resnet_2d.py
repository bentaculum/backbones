import logging
import numpy as np
from torch import nn

logger = logging.getLogger(__name__)


class ResidualBlock(nn.Module):
    """ResidualBlock.

    Args:

        in_channels (``int``):

            The number of input channels for the first convolution in the
            block.

        out_channels (``int``):

            The number of output channels for all convolutions in the
            block.

        kernel_sizes (``tuple`` of ``int``):

            The number of tuples determines the number of
            convolutional layers in each ConvBlock. If not given, each
            ConvBlock will consist of two 3x3 convolutions.

        downsample_factor (``int``):

            Use as stride in the first convolution.

        activation (``torch.nn.Module``):

            Which activation to use after a convolution.

        batch_norm (``bool``):

            If set to ``True``, apply 2d batch normalization after each
            convolution.

        group_norm (``int``):

            Number of disjunct groups for group normalization.
            If set to ``False`` group normalization is not applied.

        padding (``int``):

            Padding added to both sides of the input. Defaults to 0.

        padding_mode (``str``):

            `torch.nn.Conv2d` padding modes: `zeros`, `reflect`, `replicate` or
            `circular`.

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes=(3, 3),
        downsample_factor=2,
        activation=nn.LeakyReLU,
        batch_norm=False,
        group_norm=False,
        padding=1,
        padding_mode="replicate",
    ):

        super().__init__()

        layers = []
        rec_in = in_channels
        for i, k in enumerate(kernel_sizes):
            layers.append(
                nn.Conv2d(
                    in_channels=rec_in,
                    out_channels=out_channels,
                    kernel_size=k,
                    padding=padding,
                    padding_mode=padding_mode,
                    stride=downsample_factor if i == 0 else 1,
                )
            )
            if batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            if isinstance(group_norm, int) and group_norm > 0:
                layers.append(
                    nn.GroupNorm(
                        num_groups=group_norm,
                        num_channels=out_channels,
                    )
                )
            if i < len(kernel_sizes) - 1:
                try:
                    layers.append(activation(inplace=True))
                except TypeError:
                    layers.append(activation())

            rec_in = out_channels
        self.conv_block = nn.Sequential(*layers)

        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=downsample_factor,
            )
        ]
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if isinstance(group_norm, int) and group_norm > 0:
            layers.append(
                nn.GroupNorm(
                    num_groups=group_norm,
                    num_channels=out_channels,
                )
            )
        self.shortcut = nn.Sequential(*layers)

        try:
            self.final_activation = activation(inplace=True)
        except TypeError:
            self.final_activation = activation()

    def forward(self, x):

        return self.final_activation(self.conv_block(x) + self.shortcut(x))


class Resnet2d(nn.Module):
    """Configurable Resnet-like CNN.

    Input tensors are expected to be of shape ``(B, C, H, W)``.
    Output tensors are of shape ``(B, out_features)``.

    Args:

        in_channels:

            The number of input channels.

        inital_fmaps:

            The number of feature maps in the first layer. This is also the
            number of output feature maps. Stored in the ``channels``
            dimension.

        downsample_factors:

            Tuple of ints to use to downsample the
            feature in each residual block.

        out_features:

            The number of output features of the head.

        fmap_inc_factor:

            By how much to multiply the number of feature maps between
            blocks. If block 0 has ``k`` feature maps, layer ``l`` will
            have ``k*fmap_inc_factor**l``.

        kernel_sizes (optional):

            Tuple of ints. The number of ints determines the number of
            convolutional layers in each residual block. If not given, each
            block will consist of two 3x3 convolutions.

        activation (``torch.nn.Module``):

            Which activation to use after a convolution.

        batch_norm (optional):

            If set to ``True``, apply 2d batch normalization after each
            convolution in the ConvBlocks.

        group_norm (``int``):

            Number of disjunct groups for group normalization.
            If set to ``False`` group normalization is not applied.

        padding (``int``):

            Padding added to both sides of the convolutions. Defaults to 0.

        padding_mode (``str``):

            `torch.nn.Conv2d` padding modes: `zeros`, `reflect`, `replicate` or
            `circular`.

        fully_convolutional (``bool``):

            If set to ``True``, the head will not use global average pooling and a linear layer.
    """

    def __init__(
        self,
        in_channels,
        initial_fmaps,
        downsample_factors,
        out_features,
        fmap_inc_factor=2,
        kernel_sizes=(3, 3),
        activation=nn.LeakyReLU,
        batch_norm=True,
        group_norm=False,
        padding=1,
        padding_mode="replicate",
        fully_convolutional=False,
    ):

        super().__init__()

        self.in_channels = in_channels
        self.initial_fmaps = initial_fmaps
        self.fmap_ic_factor = fmap_inc_factor
        self.downsample_facors = downsample_factors
        self.out_features = out_features
        self.kernel_sizes = kernel_sizes
        self.activation = activation

        if group_norm and batch_norm:
            raise ValueError("Do not apply multiple normalization approaches.")
        self.batch_norm = batch_norm
        if group_norm > initial_fmaps:
            raise ValueError(f"{group_norm=} bigger {initial_fmaps=}.")
        self.group_norm = group_norm

        if not np.all((np.array(kernel_sizes) - 1) / 2 == padding):
            raise NotImplementedError("Only `same` padding implemented.")
        self.padding = padding
        self.padding_mode = padding_mode
        self.fully_convolutional = fully_convolutional

        self.levels = len(downsample_factors)

        # TODO parametrize input block
        self.input_block = nn.Identity()

        self.blocks = nn.ModuleList(
            [
                ResidualBlock(
                    in_channels=in_channels
                    if level == 0
                    else initial_fmaps * fmap_inc_factor ** (level - 1),
                    out_channels=initial_fmaps * fmap_inc_factor**level,
                    kernel_sizes=kernel_sizes,
                    downsample_factor=downsample_factors[level],
                    activation=activation,
                    batch_norm=batch_norm,
                    group_norm=group_norm,
                    padding=padding,
                    padding_mode=padding_mode,
                )
                for level in range(self.levels)
            ]
        )

        if self.fully_convolutional:
            self.head = nn.Conv2d(
                in_channels=initial_fmaps * fmap_inc_factor ** (self.levels - 1),
                out_channels=out_features,
                kernel_size=1,
                bias=True,
            )
        else:
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),
                nn.Flatten(),
                nn.Linear(
                    in_features=initial_fmaps * fmap_inc_factor ** (self.levels - 1),
                    out_features=out_features,
                ),
            )

        def init_kaiming(m):
            if self.activation == nn.ReLU:
                nonlinearity = "relu"
            elif self.activation == nn.LeakyReLU:
                nonlinearity = "leaky_relu"
            else:
                raise ValueError(
                    f"Kaiming init not applicable for activation {self.activation}."
                )
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
                nn.init.zeros_(m.bias)

        if activation in (nn.ReLU, nn.LeakyReLU):
            self.apply(init_kaiming)
            logger.debug("Initialize conv weights with Kaiming init.")

    def forward(self, x):

        x = self.input_block(x)
        for b in self.blocks:
            x = b(x)
        x = self.head(x)

        return x
