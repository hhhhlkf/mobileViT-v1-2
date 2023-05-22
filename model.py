"""
original code from apple:
https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/mobilevit.py
"""

from typing import Optional, Tuple, Union, Dict, Sequence
import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from transformer import TransformerEncoder, LinearAttnFFN, ConvLayer
from model_config import get_config
import numpy as np


def make_divisible(
        v: Union[float, int],
        divisor: Optional[int] = 8,
        min_value: Optional[Union[float, int]] = None,
) -> Union[float, int]:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class Identity(nn.Module):
    """
    This is a place-holder and returns the same tensor.
    """

    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x

    def profile_module(self, x: Tensor) -> Tuple[Tensor, float, float]:
        return x, 0.0, 0.0


class InvertedResidual(nn.Module):
    """
    This class implements the inverted residual block, as described in `MobileNetv2 <https://arxiv.org/abs/1801.04381>`_ paper

    Args:
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H_{in}, W_{in})`
        out_channels (int): :math:`C_{out}` from an expected output of size :math:`(N, C_{out}, H_{out}, W_{out)`
        stride (int): Use convolutions with a stride. Default: 1
        expand_ratio (Union[int, float]): Expand the input channels by this factor in depth-wise conv
        skip_connection (Optional[bool]): Use skip-connection. Default: True

    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})`

    .. note::
        If `in_channels =! out_channels` and `stride > 1`, we set `skip_connection=False`

    """

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            stride: int,
            expand_ratio: Union[int, float],  # 扩张因子，到底要在隐层将通道数扩张多少倍
            skip_connection: Optional[bool] = True,  # 是否使用跳跃连接
    ) -> None:
        assert stride in [1, 2]
        hidden_dim = make_divisible(int(round(in_channels * expand_ratio)), 8)

        super().__init__()

        block = nn.Sequential()
        if expand_ratio != 1:
            block.add_module(
                name="exp_1x1",
                module=ConvLayer(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1
                ),
            )

        block.add_module(
            name="conv_3x3",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                stride=stride,
                kernel_size=3,
                groups=hidden_dim  # depth-wise convolution
            ),
        )

        block.add_module(
            name="red_1x1",
            module=ConvLayer(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                use_act=False,  # 最后一层不使用激活函数
                use_norm=True,
            ),
        )

        self.block = block
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.exp = expand_ratio
        self.stride = stride
        self.use_res_connect = (
                self.stride == 1 and in_channels == out_channels and skip_connection
        )

    def forward(self, x: Tensor, *args, **kwargs) -> Tensor:
        if self.use_res_connect:  # 如果需要使用残差连接
            return x + self.block(x)
        else:
            return self.block(x)


class MobileViTBlock(nn.Module):
    """
    This class defines the `MobileViT block <https://arxiv.org/abs/2110.02178?context=cs.LG>`_

    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        transformer_dim (int): Input dimension to the transformer unit
        ffn_dim (int): Dimension of the FFN block
        n_transformer_blocks (int): Number of transformer blocks. Default: 2
        head_dim (int): Head dimension in the multi-head attention. Default: 32
        attn_dropout (float): Dropout in multi-head attention. Default: 0.0
        dropout (float): Dropout rate. Default: 0.0
        ffn_dropout (float): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (int): Patch height for unfolding operation. Default: 8
        patch_w (int): Patch width for unfolding operation. Default: 8
        transformer_norm_layer (Optional[str]): Normalization layer in the transformer block. Default: layer_norm
        conv_ksize (int): Kernel size to learn local representations in MobileViT block. Default: 3
        no_fusion (Optional[bool]): Do not combine the input and output feature maps. Default: False
    """

    def __init__(
            self,
            in_channels: int,  # 输入通道数
            transformer_dim: int,  # 输入到transformer的每个token序列长度
            ffn_dim: int,  # feed forward network的维度
            n_transformer_blocks: int = 2,  # transformer block的个数
            head_dim: int = 32,
            attn_dropout: float = 0.0,
            dropout: float = 0.0,
            ffn_dropout: float = 0.0,
            patch_h: int = 8,
            patch_w: int = 8,
            conv_ksize: Optional[int] = 3,  # 卷积核大小
            *args,
            **kwargs
    ) -> None:
        super().__init__()

        conv_3x3_in = ConvLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1
        )
        conv_1x1_in = ConvLayer(
            in_channels=in_channels,
            out_channels=transformer_dim,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False
        )

        conv_1x1_out = ConvLayer(
            in_channels=transformer_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1
        )
        conv_3x3_out = ConvLayer(
            in_channels=2 * in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1
        )

        self.local_rep = nn.Sequential()
        self.local_rep.add_module(name="conv_3x3", module=conv_3x3_in)
        self.local_rep.add_module(name="conv_1x1", module=conv_1x1_in)

        assert transformer_dim % head_dim == 0  # 验证transformer_dim是否可以被head_dim整除
        num_heads = transformer_dim // head_dim

        global_rep = [
            TransformerEncoder(
                embed_dim=transformer_dim,
                ffn_latent_dim=ffn_dim,
                num_heads=num_heads,
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout
            )
            for _ in range(n_transformer_blocks)
        ]
        global_rep.append(nn.LayerNorm(transformer_dim))
        self.global_rep = nn.Sequential(*global_rep)

        self.conv_proj = conv_1x1_out
        self.fusion = conv_3x3_out

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = transformer_dim
        self.n_heads = num_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def unfolding(self, x: Tensor) -> Tuple[Tensor, Dict]:
        patch_w, patch_h = self.patch_w, self.patch_h
        patch_area = patch_w * patch_h
        batch_size, in_channels, orig_h, orig_w = x.shape

        new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)  # 为后文判断是否需要插值做准备
        new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)  # 为后文判断是否需要插值做准备

        interpolate = False
        if new_w != orig_w or new_h != orig_h:
            # Note: Padding can be done, but then it needs to be handled in attention function.
            x = F.interpolate(x, size=(new_h, new_w), mode="bilinear", align_corners=False)
            interpolate = True

        # number of patches along width and height
        num_patch_w = new_w // patch_w  # n_w
        num_patch_h = new_h // patch_h  # n_h
        num_patches = num_patch_h * num_patch_w  # N

        # [B, C, H, W] -> [B * C * n_h, p_h, n_w, p_w]
        x = x.reshape(batch_size * in_channels * num_patch_h, patch_h, num_patch_w, patch_w)
        # [B * C * n_h, p_h, n_w, p_w] -> [B * C * n_h, n_w, p_h, p_w]
        x = x.transpose(1, 2)
        # [B * C * n_h, n_w, p_h, p_w] -> [B, C, N, P] where P = p_h * p_w and N = n_h * n_w
        x = x.reshape(batch_size, in_channels, num_patches, patch_area)
        # [B, C, N, P] -> [B, P, N, C]
        x = x.transpose(1, 3)
        # [B, P, N, C] -> [BP, N, C]
        x = x.reshape(batch_size * patch_area, num_patches, -1)

        info_dict = {
            "orig_size": (orig_h, orig_w),
            "batch_size": batch_size,
            "interpolate": interpolate,
            "total_patches": num_patches,
            "num_patches_w": num_patch_w,
            "num_patches_h": num_patch_h,
        }

        return x, info_dict

    def folding(self, x: Tensor, info_dict: Dict) -> Tensor:
        n_dim = x.dim()
        assert n_dim == 3, "Tensor should be of shape BPxNxC. Got: {}".format(
            x.shape
        )
        # [BP, N, C] --> [B, P, N, C]
        # 将x变成连续的张量，以便进行重塑操作
        x = x.contiguous().view(
            # 重塑x的第一个维度为批量大小
            info_dict["batch_size"],
            # 重塑x的第二个维度为每个图像块的像素数
            self.patch_area,
            # 重塑x的第三个维度为每个批次中的图像块总数
            info_dict["total_patches"],
            # 保持x的最后一个维度不变
            -1
        )

        batch_size, pixels, num_patches, channels = x.size()
        num_patch_h = info_dict["num_patches_h"]
        num_patch_w = info_dict["num_patches_w"]

        # [B, P, N, C] -> [B, C, N, P]
        x = x.transpose(1, 3)
        # [B, C, N, P] -> [B*C*n_h, n_w, p_h, p_w]
        x = x.reshape(batch_size * channels * num_patch_h, num_patch_w, self.patch_h, self.patch_w)
        # [B*C*n_h, n_w, p_h, p_w] -> [B*C*n_h, p_h, n_w, p_w]
        x = x.transpose(1, 2)
        # [B*C*n_h, p_h, n_w, p_w] -> [B, C, H, W]
        x = x.reshape(batch_size, channels, num_patch_h * self.patch_h, num_patch_w * self.patch_w)
        if info_dict["interpolate"]:
            x = F.interpolate(
                x,
                size=info_dict["orig_size"],
                mode="bilinear",
                align_corners=False,
            )
        return x

    def forward(self, x: Tensor) -> Tensor:
        res = x

        fm = self.local_rep(x)  # [4, 64, 28, 28]

        # convert feature map to patches
        patches, info_dict = self.unfolding(fm)  # [16, 196, 64]
        # print(patches.shape)
        # learn global representations
        for transformer_layer in self.global_rep:
            patches = transformer_layer(patches)

        # [B x Patch x Patches x C] -> [B x C x Patches x Patch]
        # Patch 所有的条状Patch的数量
        # Patches 每个条状Patch的长度
        fm = self.folding(x=patches, info_dict=info_dict)

        fm = self.conv_proj(fm)

        fm = self.fusion(torch.cat((res, fm), dim=1))
        return fm


class MobileViTBlockV2(nn.Module):
    """
    This class defines the `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ block

    Args:
        opts: command line arguments
        in_channels (int): :math:`C_{in}` from an expected input of size :math:`(N, C_{in}, H, W)`
        attn_unit_dim (int): Input dimension to the attention unit
        ffn_multiplier (int): Expand the input dimensions by this factor in FFN. Default is 2.
        n_attn_blocks (Optional[int]): Number of attention units. Default: 2
        attn_dropout (Optional[float]): Dropout in multi-head attention. Default: 0.0
        dropout (Optional[float]): Dropout rate. Default: 0.0
        ffn_dropout (Optional[float]): Dropout between FFN layers in transformer. Default: 0.0
        patch_h (Optional[int]): Patch height for unfolding operation. Default: 8
        patch_w (Optional[int]): Patch width for unfolding operation. Default: 8
        conv_ksize (Optional[int]): Kernel size to learn local representations in MobileViT block. Default: 3
        dilation (Optional[int]): Dilation rate in convolutions. Default: 1
        attn_norm_layer (Optional[str]): Normalization layer in the attention block. Default: layer_norm_2d
    """

    def __init__(self,
                 in_channels: int,
                 attn_unit_dim: int,
                 ffn_multiplier: Optional[Union[Sequence[Union[int, float]], int, float]] = 2.0,
                 n_transformer_blocks: Optional[int] = 2,
                 attn_dropout: Optional[float] = 0.0,
                 dropout: Optional[float] = 0.0,
                 ffn_dropout: Optional[float] = 0.0,
                 patch_h: Optional[int] = 8,
                 patch_w: Optional[int] = 8,
                 conv_ksize: Optional[int] = 3,
                 *args,
                 **kwargs) -> None:
        super(MobileViTBlockV2, self).__init__()
        cnn_out_dim = attn_unit_dim
        conv_3x3_in = ConvLayer(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=conv_ksize,
            stride=1,
            use_norm=True,
            use_act=True,
            groups=in_channels,
        )
        conv_1x1_in = ConvLayer(
            in_channels=in_channels,
            out_channels=cnn_out_dim,
            kernel_size=1,
            stride=1,
            use_norm=False,
            use_act=False,
        )
        self.local_rep = nn.Sequential(conv_3x3_in, conv_1x1_in)
        self.global_rep, attn_unit_dim = self._build_attn_layer(
            d_model=attn_unit_dim,
            ffn_mult=ffn_multiplier,
            n_layers=n_transformer_blocks,
            attn_dropout=attn_dropout,
            dropout=dropout,
            ffn_dropout=ffn_dropout,
        )
        self.conv_proj = ConvLayer(
            in_channels=cnn_out_dim,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            use_norm=True,
            use_act=False,
        )

        self.patch_h = patch_h
        self.patch_w = patch_w
        self.patch_area = self.patch_w * self.patch_h

        self.cnn_in_dim = in_channels
        self.cnn_out_dim = cnn_out_dim
        self.transformer_in_dim = attn_unit_dim
        self.dropout = dropout
        self.attn_dropout = attn_dropout
        self.ffn_dropout = ffn_dropout
        self.n_blocks = n_transformer_blocks
        self.conv_ksize = conv_ksize

    def _build_attn_layer(self,
                          d_model: int,
                          ffn_mult: Union[Sequence, int, float],
                          n_layers: int,
                          attn_dropout: float,
                          dropout: float,
                          ffn_dropout: float,
                          attn_norm_layer: str = "layer_norm_2d",
                          *args,
                          **kwargs) -> Tuple[nn.Module, int]:
        if isinstance(ffn_mult, Sequence) and len(ffn_mult) == 2:
            ffn_dims = (
                    np.linspace(ffn_mult[0], ffn_mult[1], n_layers, dtype=float) * d_model
            )
        elif isinstance(ffn_mult, Sequence) and len(ffn_mult) == 1:
            ffn_dims = [ffn_mult[0] * d_model] * n_layers
        elif isinstance(ffn_mult, (int, float)):
            ffn_dims = [ffn_mult * d_model] * n_layers
        else:
            raise NotImplementedError

        ffn_dims = [int((d // 16) * 16) for d in ffn_dims]

        global_rep = [
            LinearAttnFFN(
                embed_dim=d_model,
                ffn_latent_dim=ffn_dims[block_idx],
                attn_dropout=attn_dropout,
                dropout=dropout,
                ffn_dropout=ffn_dropout,
            )
            for block_idx in range(n_layers)
        ]
        global_rep.append(nn.GroupNorm(1, d_model))
        return nn.Sequential(*global_rep), d_model

    def forward(
            self, x: Union[Tensor, Tuple[Tensor]], *args, **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if isinstance(x, Tuple) and len(x) == 2:
            # for spatio-temporal data (e.g., videos)
            return self.forward_temporal(x=x[0], x_prev=x[1])
        elif isinstance(x, Tensor):
            # for image data
            return self.forward_spatial(x)
        else:
            raise NotImplementedError

    def forward_spatial(self, x: Tensor, *args, **kwargs) -> Tensor:
        x = self.resize_input_if_needed(x)
        # learn global representations on all patches
        fm = self.local_rep(x)
        patches, output_size = self.unfolding_pytorch(fm)
        # print(f"original x.shape = {patches.shape}")
        patches = self.global_rep(patches)
        # [B x Patch x Patches x C] --> [B x C x Patches x Patch]
        fm = self.folding_pytorch(patches=patches, output_size=output_size)
        fm = self.conv_proj(fm)
        return fm

    def forward_temporal(
            self, x: Tensor, x_prev: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        x = self.resize_input_if_needed(x)

        fm = self.local_rep(x)
        patches, output_size = self.unfolding_pytorch(fm)
        for global_layer in self.global_rep:
            if isinstance(global_layer, LinearAttnFFN):
                patches = global_layer(x=patches, x_prev=x_prev)
            else:
                patches = global_layer(patches)
        fm = self.folding_pytorch(patches=patches, output_size=output_size)
        fm = self.conv_proj(fm)

        return fm, patches

    def resize_input_if_needed(self, x: Tensor) -> Tensor:
        # print(f"original x.shape = {x.shape}")
        batch_size, in_channels, orig_h, orig_w = x.shape
        if orig_h % self.patch_h != 0 or orig_w % self.patch_w != 0:
            new_h = int(math.ceil(orig_h / self.patch_h) * self.patch_h)
            new_w = int(math.ceil(orig_w / self.patch_w) * self.patch_w)
            x = F.interpolate(
                x, size=(new_h, new_w), mode="bilinear", align_corners=True
            )
        # print(f"changed x.shape = {x.shape}")
        return x

    def unfolding_pytorch(self, feature_map: Tensor) -> Tuple[Tensor, Tuple[int, int]]:

        batch_size, in_channels, img_h, img_w = feature_map.shape

        # [B, C, H, W] --> [B, C, P, N]
        patches = F.unfold(
            feature_map,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )
        patches = patches.reshape(
            batch_size, in_channels, self.patch_h * self.patch_w, -1
        )

        return patches, (img_h, img_w)

    def folding_pytorch(self, patches: Tensor, output_size: Tuple[int, int]) -> Tensor:
        batch_size, in_dim, patch_size, n_patches = patches.shape

        # [B, C, P, N]
        patches = patches.reshape(batch_size, in_dim * patch_size, n_patches)

        feature_map = F.fold(
            patches,
            output_size=output_size,
            kernel_size=(self.patch_h, self.patch_w),
            stride=(self.patch_h, self.patch_w),
        )

        return feature_map


class MobileViT(nn.Module):
    """
    This class implements the `MobileViT architecture <https://arxiv.org/abs/2110.02178?context=cs.LG>`_
    """

    def __init__(self, model_cfg: Dict, num_classes: int = 1000):
        super().__init__()

        image_channels = 3
        out_channels = 16

        self.conv_1 = ConvLayer(
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2
        )

        self.layer_1, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer1"])
        self.layer_2, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer2"])
        self.layer_3, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer3"])
        self.layer_4, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer4"])
        self.layer_5, out_channels = self._make_layer(input_channel=out_channels, cfg=model_cfg["layer5"])

        exp_channels = min(model_cfg["last_layer_exp_factor"] * out_channels, 960)
        self.conv_1x1_exp = ConvLayer(
            in_channels=out_channels,
            out_channels=exp_channels,
            kernel_size=1
        )

        self.classifier = nn.Sequential()  # 有可能会被冻结，来进行网络微调
        self.classifier.add_module(name="global_pool", module=nn.AdaptiveAvgPool2d(1))
        self.classifier.add_module(name="flatten", module=nn.Flatten())
        if 0.0 < model_cfg["cls_dropout"] < 1.0:
            self.classifier.add_module(name="dropout", module=nn.Dropout(p=model_cfg["cls_dropout"]))
        self.classifier.add_module(name="fc", module=nn.Linear(in_features=exp_channels, out_features=num_classes))

        # weight init
        self.apply(self.init_parameters)

    def _make_layer(self, input_channel, cfg: Dict) -> Tuple[nn.Sequential, int]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(input_channel=input_channel, cfg=cfg)
        else:
            return self._make_mobilenet_layer(input_channel=input_channel, cfg=cfg)

    @staticmethod
    def _make_mobilenet_layer(input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio
            )
            block.append(layer)
            input_channel = output_channels

        return nn.Sequential(*block), input_channel

    @staticmethod
    def _make_mit_layer(input_channel: int, cfg: Dict) -> [nn.Sequential, int]:
        stride = cfg.get("stride", 1)
        block = []

        if stride == 2:
            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4)
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")

        transformer_dim = cfg["transformer_channels"]
        ffn_dim = cfg.get("ffn_dim")
        num_heads = cfg.get("num_heads", 4)
        head_dim = transformer_dim // num_heads

        if transformer_dim % head_dim != 0:
            raise ValueError("Transformer input dimension should be divisible by head dimension. "
                             "Got {} and {}.".format(transformer_dim, head_dim))

        block.append(MobileViTBlock(
            in_channels=input_channel,
            transformer_dim=transformer_dim,
            ffn_dim=ffn_dim,
            n_transformer_blocks=cfg.get("transformer_blocks", 1),
            patch_h=cfg.get("patch_h", 2),
            patch_w=cfg.get("patch_w", 2),
            dropout=cfg.get("dropout", 0.1),
            ffn_dropout=cfg.get("ffn_dropout", 0.0),
            attn_dropout=cfg.get("attn_dropout", 0.1),
            head_dim=head_dim,
            conv_ksize=3
        ))

        return nn.Sequential(*block), input_channel

    @staticmethod
    def init_parameters(m):
        if isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)

        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.conv_1x1_exp(x)
        x = self.classifier(x)
        return x


class MobileViTV2(nn.Module):
    """
    This class defines the `MobileViTv2 <https://arxiv.org/abs/2206.02680>`_ architecture
    """

    def __init__(self, model_cfg: Dict, num_classes: int = 1000):
        super().__init__()
        image_channels = model_cfg["layer0"]["img_channels"]
        out_channels = model_cfg["layer0"]["out_channels"]

        self.model_conf_dict = dict()
        self.conv_1 = ConvLayer(
            in_channels=image_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=2,
            use_norm=True,
            use_act=True,
        )

        self.model_conf_dict["conv1"] = {"in": image_channels, "out": out_channels}
        in_channels = out_channels
        self.layer_1, out_channels = self._make_layer(
            input_channel=in_channels, cfg=model_cfg["layer1"]
        )
        self.model_conf_dict["layer1"] = {"in": in_channels, "out": out_channels}
        in_channels = out_channels
        self.layer_2, out_channels = self._make_layer(
            input_channel=in_channels, cfg=model_cfg["layer2"]
        )

        self.model_conf_dict["layer2"] = {"in": in_channels, "out": out_channels}
        in_channels = out_channels
        self.layer_3, out_channels = self._make_layer(
            input_channel=in_channels, cfg=model_cfg["layer3"]
        )
        self.model_conf_dict["layer3"] = {"in": in_channels, "out": out_channels}
        in_channels = out_channels
        self.layer_4, out_channels = self._make_layer(
            input_channel=in_channels, cfg=model_cfg["layer4"]
        )
        in_channels = out_channels
        self.layer_5, out_channels = self._make_layer(
            input_channel=in_channels, cfg=model_cfg["layer5"] # 有可能会被冻结，来进行网络微调
        )
        self.model_conf_dict["layer5"] = {"in": in_channels, "out": out_channels}
        self.conv_1x1_exp = Identity()

        self.model_conf_dict["exp_before_cls"] = {
            "in": out_channels,
            "out": out_channels,
        }

        self.classifier = nn.Sequential()  # 有可能会被冻结，来进行网络微调
        self.classifier.add_module(name="global_pool", module=nn.AdaptiveAvgPool2d(1))
        self.classifier.add_module(name="flatten", module=nn.Flatten())
        self.classifier.add_module(name="fc", module=nn.Linear(in_features=out_channels, out_features=num_classes))

        self.apply(self.init_parameters)

    def _make_layer(self, input_channel, cfg: Dict, dilate: Optional[bool] = False) -> Optional[
        tuple[nn.Sequential, int]]:
        block_type = cfg.get("block_type", "mobilevit")
        if block_type.lower() == "mobilevit":
            return self._make_mit_layer(
                input_channel=input_channel, cfg=cfg
            )
        else:
            return self._make_mobilenet_layer(
                input_channel=input_channel, cfg=cfg
            )

    @staticmethod
    def _make_mobilenet_layer(
            input_channel: int, cfg: Dict) -> Tuple[nn.Sequential, int]:
        output_channels = cfg.get("out_channels")
        num_blocks = cfg.get("num_blocks", 2)
        expand_ratio = cfg.get("expand_ratio", 4)
        block = []

        for i in range(num_blocks):
            stride = cfg.get("stride", 1) if i == 0 else 1

            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=output_channels,
                stride=stride,
                expand_ratio=expand_ratio
            )
            block.append(layer)
            input_channel = output_channels

        return nn.Sequential(*block), input_channel

    @staticmethod
    def _make_mit_layer(input_channel: int, cfg: Dict) -> [nn.Sequential, int]:
        block = []
        stride = cfg.get("stride", 1)
        if stride == 2:
            layer = InvertedResidual(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                stride=stride,
                expand_ratio=cfg.get("mv_expand_ratio", 4),
            )

            block.append(layer)
            input_channel = cfg.get("out_channels")
        attn_unit_dim = cfg["attn_unit_dim"]
        ffn_multiplier = cfg.get("ffn_multiplier")

        block.append(
            MobileViTBlockV2(
                in_channels=input_channel,
                out_channels=cfg.get("out_channels"),
                attn_unit_dim=attn_unit_dim,
                ffn_multiplier=ffn_multiplier,
                n_transformer_blocks=cfg.get("attn_blocks", 1),
                patch_h=cfg.get("patch_h", 2),
                patch_w=cfg.get("patch_w", 2),
                dropout=cfg.get("dropout", 0.1),
                attn_dropout=cfg.get("attn_dropout", 0.1),
                ff_dropout=cfg.get("ff_dropout", 0.1),
                conv_ksize=3,
            )
        )

        return nn.Sequential(*block), input_channel
        pass

    @staticmethod
    def init_parameters(m):
        if isinstance(m, nn.Conv2d):
            if m.weight is not None:
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.GroupNorm, nn.BatchNorm2d, nn.LayerNorm)):
            if m.weight is not None:
                nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.Linear,)):
            if m.weight is not None:
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        else:
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.classifier(x)
        return x


def mobile_vit_xx_small(num_classes: int = 1000):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xxs.pt
    config = get_config("xx_small")
    m = MobileViT(config, num_classes=num_classes)
    return m


def mobile_vit_x_small(num_classes: int = 1000):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_xs.pt
    config = get_config("x_small")
    m = MobileViT(config, num_classes=num_classes)
    return m


def mobile_vit_small(num_classes: int = 1000):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit_s.pt
    config = get_config("small")
    m = MobileViT(config, num_classes=num_classes)
    return m


def mobile_vit2_xx_small(num_classes: int = 1000):
    # pretrain weight link
    # https://docs-assets.developer.apple.com/ml-research/models/cvnets/classification/mobilevit2_xxs.pt
    config = get_config("2xx_small")
    m = MobileViTV2(config, num_classes=num_classes)
    return m
