import torch
import torch.nn as nn
import numpy as np
import math
from typing import Literal


def truncated_normal_(tensor, mean=0, std=0.02):
    with torch.no_grad():
        size = tensor.size()
        tmp = tensor.new_empty(size + (4,)).normal_().cuda()
        valid = (tmp < 2) & (tmp > -2)
        ind = valid.max(-1, keepdim=True)[1]
        tensor.data.copy_(tmp.gather(-1, ind.cuda()).squeeze(-1))
        tensor.data.mul_(std).add_(mean)
        return tensor


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x

    keep_prob = torch.tensor(1 - drop_prob).cuda()
    shape = (x.size()[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype).cuda()
    random_tensor = torch.floor(random_tensor)
    output = torch.div(x, keep_prob) * random_tensor
    return output


class ConvBNLayer(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, bias_attr=False, groups=1, act=nn.GELU
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=bias_attr,
        )
        self.norm = nn.BatchNorm2d(out_channels)
        self.act = act()

    def forward(self, inputs):
        out = self.conv(inputs)
        out = self.norm(out)
        out = self.act(out)
        return out


class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvMixer(nn.Module):
    def __init__(self, dim, num_heads=8, HW=(8, 25), local_k=(3, 3)):
        super().__init__()
        self.HW = HW
        self.dim = dim
        self.local_mixer = nn.Conv2d(dim, dim, local_k, 1, (local_k[0] // 2, local_k[1] // 2), groups=num_heads)

    def forward(self, x):
        h = self.HW[0]
        w = self.HW[1]
        x = x.permute(0, 2, 1).reshape([-1, self.dim, h, w])
        x = self.local_mixer(x)
        x = torch.flatten(x, 2).permute(0, 2, 1)
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        mixer="Global",
        HW=(8, 25),
        local_k=[7, 11],
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.HW = HW
        if HW is not None:
            H = HW[0]
            W = HW[1]
            self.N = H * W
            self.C = dim
        if mixer == "Local" and HW is not None:
            hk = local_k[0]
            wk = local_k[1]
            mask = torch.ones([H * W, H + hk - 1, W + wk - 1], dtype=torch.float32).cuda()
            for h in range(0, H):
                for w in range(0, W):
                    mask[h * W + w, h : h + hk, w : w + wk] = 0.0
            mask_torch = torch.flatten(mask[:, hk // 2 : H + hk // 2, wk // 2 : W + wk // 2], 1)
            mask_inf = torch.full([H * W, H * W], -np.inf, dtype=torch.float32).cuda()
            mask = torch.where(mask_torch < 1, mask_torch, mask_inf)
            self.mask = mask.unsqueeze(0)
            self.mask = self.mask.unsqueeze(0)
            # print(self.mask.size())

        self.mixer = mixer

    def forward(self, x):
        if self.HW is not None:
            N = self.N
            C = self.C
        else:
            _, N, C = x.size()
        qkv = self.qkv(x)
        qkv = qkv.reshape((-1, N, 3, self.num_heads, C // self.num_heads))
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0] * self.scale, qkv[1], qkv[2]
        attn = q.matmul(k.permute(0, 1, 3, 2))
        if self.mixer == "Local":
            attn += self.mask
        attn = nn.functional.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        x = (attn.matmul(v)).permute(0, 2, 1, 3).reshape((-1, N, C))
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        mixer="Global",
        local_mixer=[7, 11],
        HW=[8, 25],
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0,
        attn_drop=0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer="nn.LayerNorm",
        epsilon=1e-6,
        prenorm=True,
    ):
        super().__init__()
        if isinstance(norm_layer, str):
            self.norm1 = eval(norm_layer)(dim, eps=epsilon)
        else:
            self.norm1 = norm_layer(dim)
        if mixer == "Global" or mixer == "Local":
            self.mixer = Attention(
                dim,
                num_heads=num_heads,
                mixer=mixer,
                HW=HW,
                local_k=local_mixer,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                attn_drop=attn_drop,
                proj_drop=drop,
            )
        elif mixer == "Conv":
            self.mixer = ConvMixer(dim, num_heads=num_heads, HW=HW, local_k=local_mixer)
        else:
            raise TypeError("The mixer must be one of [Global, Local, Conv]")

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else Identity()
        if isinstance(norm_layer, str):
            self.norm2 = eval(norm_layer)(dim, eps=epsilon)
        else:
            self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_ratio = mlp_ratio
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.prenorm = prenorm

    def forward(self, x):
        if self.prenorm:
            x = self.norm1(x + self.drop_path(self.mixer(x)))
            x = self.norm2(x + self.drop_path(self.mlp(x)))
        else:
            x = x + self.drop_path(self.mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):

    def __init__(self, img_size=[32, 100], in_channels=3, embed_dim=768, sub_num=2):
        super().__init__()
        num_patches = (img_size[1] // (2**sub_num)) * (img_size[0] // (2**sub_num))
        self.img_size = img_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.norm = None
        if sub_num == 2:
            self.proj = nn.Sequential(
                ConvBNLayer(
                    in_channels=in_channels,
                    out_channels=embed_dim // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    act=nn.GELU,
                    bias_attr=False,
                ),
                ConvBNLayer(
                    in_channels=embed_dim // 2,
                    out_channels=embed_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    act=nn.GELU,
                    bias_attr=False,
                ),
            )
        if sub_num == 3:
            self.proj = nn.Sequential(
                ConvBNLayer(
                    in_channels=in_channels,
                    out_channels=embed_dim // 4,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    act=nn.GELU,
                    bias_attr=False,
                ),
                ConvBNLayer(
                    in_channels=embed_dim // 4,
                    out_channels=embed_dim // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    act=nn.GELU,
                    bias_attr=False,
                ),
                ConvBNLayer(
                    in_channels=embed_dim // 2,
                    out_channels=embed_dim,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    act=nn.GELU,
                    bias_attr=False,
                ),
            )

    def forward(self, x):
        B, C, H, W = x.size()
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).permute(0, 2, 1)
        return x


class SubSample(nn.Module):
    def __init__(self, in_channels, out_channels, types="Pool", stride=(2, 1), sub_norm="nn.LayerNorm", act=None):
        super().__init__()
        self.types = types
        if types == "Pool":
            self.avgpool = nn.AvgPool2d(kernel_size=(3, 5), stride=stride, padding=(1, 2))
            self.maxpool = nn.MaxPool2d(kernel_size=(3, 5), stride=stride, padding=(1, 2))
            self.proj = nn.Linear(in_channels, out_channels)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.norm = eval(sub_norm)(out_channels)
        if act is not None:
            self.act = act()
        else:
            self.act = None

    def forward(self, x):
        if self.types == "Pool":
            x1 = self.avgpool(x)
            x2 = self.maxpool(x)
            x = (x1 + x2) * 0.5
            out = self.proj(x.flatten(2).permute(0, 2, 1))
        else:
            x = self.conv(x)
            out = x.flatten(2).permute(0, 2, 1)
        out = self.norm(out)
        if self.act is not None:
            out = self.act(out)

        return out


class SVTRNet(nn.Module):
    def __init__(
        self,
        img_size=[32, 128],
        in_channels=3,
        embed_dim=[64, 128, 256],
        depth=[3, 6, 3],
        num_heads=[2, 4, 8],
        mixer=["Local"] * 6 + ["Global"] * 6,  # Local atten, Global atten, Conv
        local_mixer=[[7, 11], [7, 11], [7, 11]],
        patch_merging="Conv",  # Conv, Pool, None
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        last_drop=0.1,
        attn_drop_rate=0.0,
        drop_path_rate=0.1,
        norm_layer="nn.LayerNorm",
        sub_norm="nn.LayerNorm",
        epsilon=1e-6,
        out_channels=192,
        out_char_num=25,
        block_unit="Block",
        act="nn.GELU",
        last_stage=True,
        sub_num=2,
        prenorm=True,
        use_lenhead=False,
        **kwargs,
    ):
        super().__init__()
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.out_channels = out_channels
        self.prenorm = prenorm
        patch_merging = None if patch_merging != "Conv" and patch_merging != "Pool" else patch_merging
        self.patch_embed = PatchEmbed(
            img_size=img_size, in_channels=in_channels, embed_dim=embed_dim[0], sub_num=sub_num
        )
        num_patches = self.patch_embed.num_patches
        self.HW = [img_size[0] // (2**sub_num), img_size[1] // (2**sub_num)]
        # self.pos_embed = self.create_parameter(
        #     shape=[1, num_patches, embed_dim[0]], default_initializer=zeros_)
        # self.add_parameter("pos_embed", self.pos_embed)
        self.pos_embed = nn.Parameter(
            torch.zeros([1, num_patches, embed_dim[0]], dtype=torch.float32), requires_grad=True
        )
        self.pos_drop = nn.Dropout(p=drop_rate)
        Block_unit = eval(block_unit)

        dpr = np.linspace(0, drop_path_rate, sum(depth))

        self.blocks1 = nn.ModuleList(
            [
                Block_unit(
                    dim=embed_dim[0],
                    num_heads=num_heads[0],
                    mixer=mixer[0 : depth[0]][i],
                    HW=self.HW,
                    local_mixer=local_mixer[0],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=eval(act),
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[0 : depth[0]][i],
                    norm_layer=norm_layer,
                    epsilon=epsilon,
                    prenorm=prenorm,
                )
                for i in range(depth[0])
            ]
        )
        if patch_merging is not None:
            self.sub_sample1 = SubSample(
                embed_dim[0], embed_dim[1], sub_norm=sub_norm, stride=[2, 1], types=patch_merging
            )
            HW = [self.HW[0] // 2, self.HW[1]]
        else:
            HW = self.HW
        self.patch_merging = patch_merging
        self.blocks2 = nn.ModuleList(
            [
                Block_unit(
                    dim=embed_dim[1],
                    num_heads=num_heads[1],
                    mixer=mixer[depth[0] : depth[0] + depth[1]][i],
                    HW=HW,
                    local_mixer=local_mixer[1],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=eval(act),
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[depth[0] : depth[0] + depth[1]][i],
                    norm_layer=norm_layer,
                    epsilon=epsilon,
                    prenorm=prenorm,
                )
                for i in range(depth[1])
            ]
        )
        if patch_merging is not None:
            self.sub_sample2 = SubSample(
                embed_dim[1], embed_dim[2], sub_norm=sub_norm, stride=[2, 1], types=patch_merging
            )
            HW = [self.HW[0] // 4, self.HW[1]]
        else:
            HW = self.HW
        self.blocks3 = nn.ModuleList(
            [
                Block_unit(
                    dim=embed_dim[2],
                    num_heads=num_heads[2],
                    mixer=mixer[depth[0] + depth[1] :][i],
                    HW=HW,
                    local_mixer=local_mixer[2],
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=eval(act),
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[depth[0] + depth[1] :][i],
                    norm_layer=norm_layer,
                    epsilon=epsilon,
                    prenorm=prenorm,
                )
                for i in range(depth[2])
            ]
        )
        self.last_stage = last_stage
        if last_stage:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, out_char_num))
            self.last_conv = nn.Conv2d(
                in_channels=embed_dim[2], out_channels=self.out_channels, kernel_size=1, stride=1, padding=0, bias=False
            )
            self.hardswish = nn.Hardswish()
            self.dropout = nn.Dropout(p=last_drop)
        if not prenorm:
            self.norm = eval(norm_layer)(embed_dim[-1], epsilon=epsilon)
        self.use_lenhead = use_lenhead
        if use_lenhead:
            self.len_conv = nn.Linear(embed_dim[2], self.out_channels)
            self.hardswish_len = nn.Hardswish()
            self.dropout_len = nn.Dropout(p=last_drop)

        truncated_normal_(self.pos_embed)
        self.apply(self._init_weights)
        print("---------model weight inits-----------")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            truncated_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.zeros_(m.bias)
        if isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

    def forward_features(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks1:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample1(x.permute(0, 2, 1).reshape([-1, self.embed_dim[0], self.HW[0], self.HW[1]]))
        for blk in self.blocks2:
            x = blk(x)
        if self.patch_merging is not None:
            x = self.sub_sample2(x.permute(0, 2, 1).reshape([-1, self.embed_dim[1], self.HW[0] // 2, self.HW[1]]))
        for blk in self.blocks3:
            x = blk(x)
        if not self.prenorm:
            x = self.norm(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        if self.use_lenhead:
            len_x = self.len_conv(x.mean(1))
            len_x = self.dropout_len(self.hardswish_len(len_x))
        if self.last_stage:
            if self.patch_merging is not None:
                h = self.HW[0] // 4
            else:
                h = self.HW[0]
            x = self.avg_pool(x.permute(0, 2, 1).reshape([-1, self.embed_dim[2], h, self.HW[1]]))
            x = self.last_conv(x)
            x = self.hardswish(x)
            x = self.dropout(x)
        if self.use_lenhead:
            return x, len_x
        return x


class Im2Seq(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        # self.out_channels = in_channels

    def forward(self, x):
        B, C, H, W = x.size()
        assert H == 1
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)  # (NTC)(batch, width, channels)
        return x


class CTCHead(nn.Module):
    def __init__(
        self,
        in_channels=192,
        out_channels=6624,
        fc_decay=0.0004,
        mid_channels=None,
        return_feats=False,
        **kwargs,
    ):
        super(CTCHead, self).__init__()
        if mid_channels is None:
            self.fc = nn.Linear(in_channels, out_channels)
        else:
            self.fc1 = nn.Linear(in_channels, mid_channels)
            self.fc2 = nn.Linear(mid_channels, out_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels
        self.return_feats = return_feats
        self.apply(self._init_weights)
        print("---------model weight inits-----------")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            stdv = 1.0 / math.sqrt(self.in_channels * 1.0)
            nn.init.uniform_(m.weight, -stdv, stdv)
            nn.init.uniform_(m.bias, -stdv, stdv)

    def forward(self, x):
        if self.mid_channels is None:
            predicts = self.fc(x)
        else:
            x = self.fc1(x)
            predicts = self.fc2(x)
        #   batchsize * T * C ---->  T * batchsize * C
        predicts = predicts.permute(1, 0, 2)
        return predicts
        # predicts = predicts.log_softmax(2).requires_grad_()

        # if self.return_feats:
        #     result = (predicts, x)
        # else:
        #     result = (predicts, None)
        # return result


class SVTRArch(nn.Module):
    def __init__(
        self,
        img_size: list[int],
        embed_dim: list[int],
        depth: list[int],
        num_heads: list[int],
        out_channels: int,
        mixer: list[str],
        num_classes: int,
        out_char_num: int,
    ):
        super(SVTRArch, self).__init__()
        self.backbone = SVTRNet(
            img_size=img_size,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            out_channels=out_channels,
            mixer=mixer,
            out_char_num=out_char_num,
        )
        self.neck = Im2Seq()
        self.head = CTCHead(
            in_channels=out_channels,
            out_channels=num_classes,
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)
        return x


def SVTR_T(num_classes: int):
    return SVTRArch(
        img_size=[32, 128],
        embed_dim=[64, 128, 256],
        depth=[3, 6, 3],
        num_heads=[2, 4, 8],
        out_channels=192,
        mixer=["Local"] * 6 + ["Global"] * 6,
        num_classes=num_classes,
        out_char_num=2 * 25 + 1,
    )


def SVTR_S(num_classes: int):
    return SVTRArch(
        img_size=[32, 128],
        embed_dim=[96, 192, 256],
        depth=[3, 6, 6],
        num_heads=[3, 6, 8],
        out_channels=192,
        mixer=["Local"] * 8 + ["Global"] * 7,
        num_classes=num_classes,
        out_char_num=2 * 25 + 1,
    )


def SVTR_B(num_classes: int):
    return SVTRArch(
        img_size=[32, 128],
        embed_dim=[128, 256, 384],
        depth=[3, 6, 9],
        num_heads=[4, 8, 12],
        out_channels=256,
        mixer=["Local"] * 8 + ["Global"] * 10,
        num_classes=num_classes,
        out_char_num=2 * 25 + 1,
    )


def SVTR_L(num_classes: int):
    return SVTRArch(
        img_size=[32, 128],
        embed_dim=[192, 256, 512],
        depth=[3, 9, 9],
        num_heads=[6, 8, 16],
        out_channels=384,
        mixer=["Local"] * 10 + ["Global"] * 11,
        num_classes=num_classes,
        out_char_num=2 * 25 + 1,
    )


class SVTR:

    def __init__(self, num_classes: int, model_size: str = Literal["T", "S", "B", "L"]):

        self.num_classes = num_classes
        self.model_size = model_size

    def __call__(self):
        return globals()[f"SVTR_{self.model_size}"](self.num_classes)
