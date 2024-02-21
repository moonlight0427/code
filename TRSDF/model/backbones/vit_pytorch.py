""" Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in
'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale' - https://arxiv.org/abs/2010.11929

The official jax code is released and available at https://github.com/google-research/vision_transformer

Status/TODO:
* Models updated to be compatible with official impl. Args added to support backward compat for old PyTorch weights.
* Weights ported from official jax impl for 384x384 base and small models, 16x16 and 32x32 patches.
* Trained (supervised on ImageNet-1k) my custom 'small' patch model to 77.9, 'base' to 79.4 top-1 with this code.
* Hopefully find time and GPUs for SSL or unsupervised pretraining on OpenImages w/ ImageNet fine-tune in future.

Acknowledgments:
* The paper authors for releasing code and weights, thanks!
* I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch ... check it out
for some einops/einsum fun
* Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
* Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020 Ross Wightman
"""
import math
from functools import partial
from itertools import repeat
from collections import OrderedDict

import numpy
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch._six import container_abcs
import collections.abc as container_abcs
from ..generalizemeanpooling import GeneralizedMeanPoolingP, GeneralizedMeanPooling
from .interchange_matrix import interchange_matrix
import matplotlib.pyplot as plt
from torchvision import transforms



# From PyTorch internals
def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
to_2tuple = _ntuple(2)

def complement_idx(idx, dim):
    """
    Compute the complement: set(range(dim)) - set(idx).
    idx is a multi-dimensional tensor, find the complement for its trailing dimension,
    all other dimension is considered batched.
    Args:
        idx: input index, shape: [N, *, K]
        dim: the max index for complement
    """
    a = torch.arange(dim, device=idx.device)
    ndim = idx.ndim
    dims = idx.shape
    n_idx = dims[-1]
    dims = dims[:-1] + (-1, )
    for i in range(1, ndim):
        a = a.unsqueeze(0)
    a = a.expand(*dims)
    masked = torch.scatter(a, -1, idx, 0)
    compl, _ = torch.sort(masked, dim=-1, descending=False)
    compl = compl.permute(-1, *tuple(range(ndim - 1)))
    compl = compl[n_idx:].permute(*(tuple(range(1, ndim)) + (0,)))
    return compl

def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'vit_small_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/vit_small_p16_224-15ec54c9.pth',
    ),
    'vit_base_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_224-80ecf9dd.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vit_base_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p16_384-83fb41ba.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_base_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_base_p32_384-830016f5.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_224-4ee7a4dc.pth',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    'vit_large_patch16_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p16_384-b3be5167.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
    'vit_huge_patch16_224': _cfg(),
    'vit_huge_patch32_384': _cfg(input_size=(3, 384, 384)),
    # hybrid models
    'vit_small_resnet26d_224': _cfg(),
    'vit_small_resnet50d_s3_224': _cfg(),
    'vit_base_resnet26d_224': _cfg(),
    'vit_base_resnet50d_224': _cfg(),
}


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,  keep_rate=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5


        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.keep_rate = keep_rate
        assert 0 < keep_rate <= 1, "keep_rate must > 0 and <= 1, got {0}".format(keep_rate)

    def forward(self, x, keep_rate=None, tokens=None):
        if keep_rate is None:  # 调用传过来的keep_rate为None
            keep_rate = self.keep_rate
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # size = [3, B, self.num_heads, N, C // self.num_heads]
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)  # [B, H, N, N]


        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)


        # left_tokens = N - 1  # N减去class_token  244-1=243
        # if self.keep_rate < 1 and keep_rate < 1 or tokens is not None:  # double check the keep rate  判断正确！self.keep_rate: 0.7, keep_rate: 0.7, token: None
        #     left_tokens = math.ceil(keep_rate * (N - 1))  # math.ceil(x)返回大于等于参数x的最小整数  left_tokens:保留tokens的数量  171！！！！！
        #     if tokens is not None:
        #         left_tokens = tokens
        #     if left_tokens == N - 1:  # 即keep_rate=1
        #         return x, None, None, None, left_tokens
        #     # 当前left_tokens = keep_rate * (N-1)!!一定记住！！
        #     assert left_tokens >= 1
        #     cls_attn = attn[:, :, 0, 1:]  # [B, H, N-1] = [B, H, 242]  # 提取出class_token与除class_token的其他patch之间的注意力分数（即相关性）
        #     cls_attn = cls_attn.mean(dim=1)  # [B, N-1] = [B, 242]  对应原文会对不同head中[cls] token与patch token的attention取一个平均，再用这个结果去保留对应比率的tokens  计算每个token的所有Heads的平均注意力值作为该token的注意力值
        #
        #     _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True, sorted=True)  # [B, left_tokens] 返回前left_tokens个并按从大到小排序的值和对应索引
        #     # cls_idx = torch.zeros(B, 1, dtype=idx.dtype, device=idx.device)
        #     # index = torch.cat([cls_idx, idx + 1], dim=1)
        #     index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C] = [B, 171, C]  增加最后一个维度，并expand为C，保持输出为[B, N, C]
        #
        #     return x, index, idx, cls_attn, left_tokens  # x为原本attention的输出[B, N, C]， index为本文keep_rate后的输出[B, N, C]



        # left_tokens = N - 1
        left_tokens = 242  # N减去class_token  244-1=243
        if self.keep_rate < 1 and keep_rate < 1 or tokens is not None:  # double check the keep rate  判断正确！self.keep_rate: 0.7, keep_rate: 0.7, token: None
            left_tokens = math.ceil(
                keep_rate * (242))  # math.ceil(x)返回大于等于参数x的最小整数  left_tokens:保留tokens的数量  171！！！！！
            if tokens is not None:
                left_tokens = tokens
            if left_tokens == 242:  # 即keep_rate=1
                return x, None, None, None, left_tokens
            # 当前left_tokens = keep_rate * (N-1)!!一定记住！！
            assert left_tokens >= 1



            cls_attn = attn[:, :, 0, -242:]
            # cls_attn = attn[:, :, 0, 1:]  # [B, H, N-1] = [B, H, 242]  # 提取出class_token与除class_token的其他patch之间的注意力分数（即相关性）
            cls_attn = cls_attn.mean(dim=1)  # [B, N-1] = [B, 242]  对应原文会对不同head中[cls] token与patch token的attention取一个平均，再用这个结果去保留对应比率的tokens  计算每个token的所有Heads的平均注意力值作为该token的注意力值

            # 加上attn矩阵列注意力值！！
            patch_attn = attn[:, :, -242:, 0]
            # patch_attn = attn[:, :, 1:, 0]
            patch_attn = patch_attn.mean(dim=1)

            # cls_attn = F.softmax(cls_attn, dim=-1) * F.softmax(patch_attn, dim=-2)
            cls_attn = F.softmax(cls_attn, dim=-1) * F.softmax(patch_attn, dim=-1)

            _, idx = torch.topk(cls_attn, left_tokens, dim=1, largest=True,
                                sorted=True)  # [B, left_tokens] 返回前left_tokens个并按从大到小排序的值和对应索引
            # cls_idx = torch.zeros(B, 1, dtype=idx.dtype, device=idx.device)
            # index = torch.cat([cls_idx, idx + 1], dim=1)
            index = idx.unsqueeze(-1).expand(-1, -1, C)  # [B, left_tokens, C] = [B, 171, C]  增加最后一个维度，并expand为C，保持输出为[B, N, C]

            return x, index, idx, cls_attn, left_tokens  # x为原本attention的输出[B, N, C]， index为本文keep_rate后的输出[B, N, C]



        return x, None, None, None, left_tokens



# transformer Encoder
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, keep_rate=0.,
                 fuse_token=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, keep_rate=keep_rate)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.keep_rate = keep_rate
        self.mlp_hidden_dim = mlp_hidden_dim
        self.fuse_token = fuse_token

    def forward(self, x, keep_rate=None, tokens=None, get_idx=False):
        if keep_rate is None:
            keep_rate = self.keep_rate  # this is for inference, use the default keep rate  0.7！！
        B, N, C = x.shape

        x_class_cat = x[:, 0:1]

        tmp, index, idx, cls_attn, left_tokens = self.attn(self.norm1(x), keep_rate, tokens)  # tmp原transformer输出，index为本文keep_rate后的输出，idx前k个最大attn值得索引，cls_attn为class_token与其他patch的相关性，left_tokens=(N-1)*keep_rate
        x = x + self.drop_path(tmp)  # 残差



        # 加A矩阵生成函数，传入值x

        if index is not None:  # 成立


            # compl = complement_idx(idx, N - 1)  # 取除后10%patch的所有其他patch做加权平均替换到需要被替换的patch(加权平均)
            # matrix_temp = numpy.ones((242, 242))  # 生成这个矩阵为了用torch.cat，并无实际作用
            #
            # matrix_temp = matrix_temp.astype(numpy.float32)
            # matrix_temp = torch.from_numpy(matrix_temp).unsqueeze(0).cuda()
            # for i in range(B):  # 循环后生成[33, 242, 242]的tensor，最后剔除第一个matrix_temp   在循环中，针对每一个matrix进行变换
            #     matrix = interchange_matrix.generate_interchange_matrix(x)
            #     matrix = matrix.astype(numpy.float32)
            #     # 矩阵对角线选取需要被替换的patch
            #     non_topk_attn_idx = compl[i].tolist()  # 获取每张图像的非相关注意力索引值，list形式
            #     topk_attn_idx = idx[i].tolist()
            #
            #     for i in range(len(non_topk_attn_idx)):
            #         matrix[non_topk_attn_idx[i]][non_topk_attn_idx[i]] = 0
            #         for j in range(len(topk_attn_idx)):
            #             matrix[topk_attn_idx[j]][non_topk_attn_idx[i]] = 1 / len(topk_attn_idx)
            #     matrix = torch.from_numpy(matrix).unsqueeze(0).cuda()
            #     matrix_temp = torch.cat((matrix_temp, matrix), dim=0)
            #
            # matrix = matrix_temp[1:, :, :]  # 当前matrix为确定哪些需要被替换哪些不需要被替换 [b,242,242]
            # matrix = matrix.to(torch.float32)
            #
            # cls_token = x[:, 0:1, :]  # [b, 1, c]
            # x_non_token = x[:, 1:, :]  # [b, 242, c]
            # x_non_token = x_non_token.transpose(1, 2)  # [b, c, 242]
            #
            # x_non_token = torch.bmm(x_non_token, matrix)
            # x_non_token = x_non_token.transpose(1, 2)  # [b, 242, c]
            #
            # x = torch.cat((cls_token, x_non_token), dim=1)





            compl = complement_idx(idx, 242)  # 取除后10%patch的所有其他patch做加权平均替换到需要被替换的patch(加权平均),做残差===========================69.2
            matrix_temp = numpy.ones((242, 242))  # 生成这个矩阵为了用torch.cat，并无实际作用

            matrix_temp = matrix_temp.astype(numpy.float32)
            matrix_temp = torch.from_numpy(matrix_temp).unsqueeze(0).cuda()
            for i in range(B):  # 循环后生成[33, 242, 242]的tensor，最后剔除第一个matrix_temp   在循环中，针对每一个matrix进行变换
                matrix = interchange_matrix.generate_interchange_matrix(x)
                matrix = matrix.astype(numpy.float32)
                # 矩阵对角线选取需要被替换的patch
                non_topk_attn_idx = compl[i].tolist()  # 获取每张图像的非相关注意力索引值，list形式
                topk_attn_idx = idx[i].tolist()

                for i in range(len(non_topk_attn_idx)):
                    matrix[non_topk_attn_idx[i]][non_topk_attn_idx[i]] = 0
                    for j in range(len(topk_attn_idx)):
                        matrix[topk_attn_idx[j]][non_topk_attn_idx[i]] = 1 / len(topk_attn_idx)
                matrix = torch.from_numpy(matrix).unsqueeze(0).cuda()
                matrix_temp = torch.cat((matrix_temp, matrix), dim=0)

            matrix = matrix_temp[1:, :, :]  # 当前matrix为确定哪些需要被替换哪些不需要被替换 [b,242,242]
            matrix = matrix.to(torch.float32)

            cls_token = x[:, 0:-242, :]  # [b, 1, c]
            x_non_token = x[:, -242:, :]  # [b, 242, c]
            x_non_token = x_non_token.transpose(1, 2)  # [b, c, 242]

            x_non_token = torch.bmm(x_non_token, matrix)
            x_non_token = x_non_token.transpose(1, 2)  # [b, 242, c]

            x_ = torch.cat((cls_token, x_non_token), dim=1)



            # compl = complement_idx(idx, 242)  # 取所有patch加权平均为全局特征替换到需要被替换的patch(加权平均),做残差
            # matrix_temp = numpy.ones((242, 242))  # 生成这个矩阵为了用torch.cat，并无实际作用
            #
            # matrix_temp = matrix_temp.astype(numpy.float32)
            # matrix_temp = torch.from_numpy(matrix_temp).unsqueeze(0).cuda()
            # for i in range(B):  # 循环后生成[33, 242, 242]的tensor，最后剔除第一个matrix_temp   在循环中，针对每一个matrix进行变换
            #     matrix = interchange_matrix.generate_interchange_matrix(x)
            #     matrix = matrix.astype(numpy.float32)
            #     # 矩阵对角线选取需要被替换的patch
            #     non_topk_attn_idx = compl[i].tolist()  # 获取每张图像的非相关注意力索引值，list形式
            #     topk_attn_idx = idx[i].tolist()
            #
            #     for i in range(len(non_topk_attn_idx)):
            #         matrix[non_topk_attn_idx[i]][non_topk_attn_idx[i]] = 0
            #         for j in range(len(topk_attn_idx)):
            #             matrix[topk_attn_idx[j]][non_topk_attn_idx[i]] = 1 / 242
            #     matrix = torch.from_numpy(matrix).unsqueeze(0).cuda()
            #     matrix_temp = torch.cat((matrix_temp, matrix), dim=0)
            #
            # matrix = matrix_temp[1:, :, :]  # 当前matrix为确定哪些需要被替换哪些不需要被替换 [b,242,242]
            # matrix = matrix.to(torch.float32)
            #
            # cls_token = x[:, 0:-242, :]  # [b, 1, c]
            # x_non_token = x[:, -242:, :]  # [b, 242, c]
            # x_non_token = x_non_token.transpose(1, 2)  # [b, c, 242]
            #
            # x_non_token = torch.bmm(x_non_token, matrix)
            # x_non_token = x_non_token.transpose(1, 2)  # [b, 242, c]
            #
            # x_ = torch.cat((cls_token, x_non_token), dim=1)



            x = x + self.drop_path(self.mlp(self.norm2(x_)))  # [B, 173, C]

            x = torch.cat((x_class_cat, x), dim=1)

            # return x, non_topk_attn_idx  # 可视化时打开
            return x










            # B, N, C = x.shape
            # non_cls = x[:, 1:]  # 除class_token以外的tokens  [B, 242, 768]
            # x_others = torch.gather(non_cls, dim=1, index=index)  # [B, left_tokens, C] = [B, 171, C]  取出相关注意力特征
            #
            # if self.fuse_token:  # self.fuse_token为True,表示fusion非相关注意力特征   融合fusion
            #     compl = complement_idx(idx, N - 1)  # [B, N-1-left_tokens] = [B, 72]  取除class_token和相关注意力特征以外的非相关注意力特征
            #
            #     # non_topk = torch.gather(non_cls, dim=1, index=compl.unsqueeze(-1).expand(-1, -1, C))  # [B, N-1-left_tokens, C] = [B, 72, C] 非topk注意力特征
            #     #
            #     # non_topk_attn = torch.gather(cls_attn, dim=1, index=compl)  # [B, N-1-left_tokens] = [B, 72]
            #     # extra_token = torch.sum(non_topk * non_topk_attn.unsqueeze(-1), dim=1, keepdim=True)  # [B, 1, C]  # 取出非相关融合注意力特征
            #     # x = torch.cat([x[:, 0:1], x_others, extra_token], dim=1)  # class_token和相关注意力和融合后的非相关注意力拼接
            # else:
            #     x = torch.cat([x[:, 0:1], x_others], dim=1)

        x = x + self.drop_path(self.mlp(self.norm2(x)))  # [B, 173, C]
        x = torch.cat((x_class_cat, x), dim=1)
        # return x, None  # 可视化时打开
        return x

# 没用！！！！！！不重叠patch块
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


# 原文中生成重叠的patch块
class PatchEmbed_overlap(nn.Module):
    """ Image to Patch Embedding with overlapping patches
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=20, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)  # to_2tuple作用是将输入变为元组，2表示新元组长度
        patch_size = to_2tuple(patch_size)
        stride_size_tuple = to_2tuple(stride_size)
        self.num_x = (img_size[1] - patch_size[1]) // stride_size_tuple[1] + 1  # 一个方向上的patch数
        self.num_y = (img_size[0] - patch_size[0]) // stride_size_tuple[0] + 1
        print('using stride: {}, and patch number is num_y{} * num_x{}'.format(stride_size, self.num_y, self.num_x))
        num_patches = self.num_x * self.num_y  # 重叠的patch块数 11*22=242
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride_size)  # linear projection线性投影，文中F，将patch映射为embed_dim
        for m in self.modules():
            if isinstance(m, nn.Conv2d):

                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.InstanceNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        B, C, H, W = x.shape  # [64, 3, 256, 128]

        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)  # [64, 768, 22, 11]
        # x = x[:, :, : 11, :]  # 取一半patch数（上半部分）不跑实验3要记得注释掉！

        x = x.flatten(2).transpose(1, 2)  # [64, 242, 768]
        return x


class TransReID(nn.Module):
    """ Transformer-based Object Re-Identification
    """
    def __init__(self, img_size=224, patch_size=16, stride_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, representation_size=None, distilled=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., camera=0, view=0,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, local_feature=False, sie_xishu =1.0,
                 act_layer=None, weight_init='', keep_rate=(1, ), fuse_token=False):
        super().__init__()

        if len(keep_rate) == 1:  # keep_rate为元组
            keep_rate = keep_rate * depth
        self.keep_rate = keep_rate
        self.depth = depth
        self.first_shrink_idx = depth
        for i, s in enumerate(keep_rate):  # i为索引，s为值
            if s < 1:
                self.first_shrink_idx = i  # self.shrink = 0应该是
                break

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU


        self.local_feature = local_feature  # 传的是cfg.MODEL.JPM,默认为True
        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            self.patch_embed = PatchEmbed_overlap(
                img_size=img_size, patch_size=patch_size, stride_size=stride_size, in_chans=in_chans,
                embed_dim=embed_dim)

        num_patches = self.patch_embed.num_patches  # 重叠的patch块数量 11*22=242

        # num_patches = 121  # 做裁剪一半patch数的小实验时候改的

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # [1, 1, 768]
        # nn.Parameter:含义是 将一个固定不可训练的tensor转换成可以训练的类型  torch.zero为生成参数给定size的tensor(全为0)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))  # [1, 243, 768]

        self.LPDE = nn.Parameter(torch.ones(1, num_patches, embed_dim))  # 对应原文构造一个与输入补丁序列大小相同的可学习张量LPDE [1, 242, 768] 不确定是1还是64！！ 实验8!!!
        self.cam_num = camera
        self.view_num = view
        self.sie_xishu = sie_xishu
        # Initialize SIE Embedding
        if camera > 1 and view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera * view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('camera number is : {} and viewpoint number is : {}'.format(camera, view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif camera > 1:
            self.sie_embed = nn.Parameter(torch.zeros(camera, 1, embed_dim))  # [8, 1, 768]
            trunc_normal_(self.sie_embed, std=.02)  # 对self.sie_embed用截断正态分布
            print('camera number is : {}'.format(camera))
            print('using SIE_Lambda is : {}'.format(sie_xishu))
        elif view > 1:
            self.sie_embed = nn.Parameter(torch.zeros(view, 1, embed_dim))
            trunc_normal_(self.sie_embed, std=.02)
            print('viewpoint number is : {}'.format(view))
            print('using SIE_Lambda is : {}'.format(sie_xishu))

        print('using drop_out rate is : {}'.format(drop_rate))
        print('using attn_drop_out rate is : {}'.format(attn_drop_rate))
        print('using drop_path rate is : {}'.format(drop_path_rate))

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                keep_rate=keep_rate[i], fuse_token=fuse_token)
            for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:  # distilled：浓缩，压缩
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.fc = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        # self.init_weights(weight_init)  # 不知道加不加

        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        # trunc_normal_(self.dist_token, std=.02)

        self.apply(self._init_weights)

        self.conv1x1 = nn.Conv2d(in_channels=3072, out_channels=768, kernel_size=1, stride=1)  # 实验四用到！

        self.conv1x1_ = nn.Linear(3072, 768)

        self.linear_3072 = nn.Linear(3072, 768)  # 实验5,7用到！

        self.linear_1536 = nn.Linear(1536, 768)


        self.linear_219 = nn.Linear(219, 181)
        self.linear_199 = nn.Linear(199, 181)

        self.adaptivemaxpooling = nn.AdaptiveMaxPool1d(181)
        self.adaptiveavgpooling = nn.AdaptiveAvgPool1d(181)
        self.adaptiveavgpooling_2 = nn.AdaptiveAvgPool1d(243)
        self.generalizedmeanpoolingp = GeneralizedMeanPoolingP(norm=4)  # norm值可变，一般为2-10
        self.generalizedmeanpooling = GeneralizedMeanPooling(norm=2)
        # self.dropout = nn.Dropout

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}

    def get_classifier(self):
        if self.dist_token is None:
            return self.head
        else:
            return self.head, self.head_dist

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.fc = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        if self.num_tokens == 2:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, camera_id, view_id, keep_rate=None, tokens=None, get_idx=False):
        B = x.shape[0]  # batchsize = 64

        if not isinstance(keep_rate, (tuple, list)):
            keep_rate = (keep_rate, ) * self.depth
        if not isinstance(tokens, (tuple, list)):
            tokens = (tokens, ) * self.depth
        assert len(keep_rate) == self.depth
        assert len(tokens) == self.depth

        x = self.patch_embed(x)  # [64, 242, 768]

        # self.LPDE = self.LPDE.expand(B, -1, -1)  # 可以试一下这句话，根据class_token想到的
        # x = x * self.LPDE  # LPDE和输入补丁序列之间的Hadamard乘积  实验8！！！！！！

        # x = x[:, 0:121]  # 效果不好，rank1--48.6

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [64, 1, 768]  # stole cls_tokens impl from Phil Wang, thanks
        if self.dist_token is None:
            x = torch.cat((cls_tokens, x), dim=1)  # 感觉是执行这句
        else:
            x = torch.cat((cls_tokens, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)

        if self.cam_num > 0 and self.view_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id * self.view_num + view_id]
        elif self.cam_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[camera_id]
            # x = x + self.pos_embed
        elif self.view_num > 0:
            x = x + self.pos_embed + self.sie_xishu * self.sie_embed[view_id]
        else:
            x = x + self.pos_embed

        x = self.pos_drop(x)  # [B, 244, 768]


        # left_tokens = []
        # idxs = []
        # for i, blk in enumerate(self.blocks[:-1]):
        #     x, left_token, idx = blk(x, keep_rate[i], tokens[i], get_idx)
        #     left_tokens.append(left_token)
        #     if idx is not None:
        #         idxs.append(idx)
        # # x = self.norm(x)
        # if self.dist_token is None:
        #     return self.pre_logits(x), left_tokens, idxs
        # else:
        #     return x, x[:, 1], idxs







        # B = x.shape[0]  ###  实验四！！！！！！！！！  AdaptiveAvgPool拼接
        # W = int(((x.shape[1] - 1) / 2.0) ** 0.5)  # W：11
        # H = int((x.shape[1] - 1) / W)  # H：22
        #
        # flag = 0
        # # if self.local_feature:
        # for i, blk in enumerate(self.blocks[:-1]):
        #     # x = blk(x)
        #     x = blk(x, keep_rate[i], tokens[i], get_idx)  # x:[32, 243, 768], orth_proto:[32, 12, 242]
        #     if flag == 3:
        #         # x [32, 220, 768]
        #         x1 = x[:, 1:]  # [32， 768， 22， 11]  # 对应原文除class_token与最后一个block输出做组合  2th block输出
        #     elif flag == 6:
        #         x3 = x[:, 1:]  # [32， 768， 22， 11]  4th block输出
        #     elif flag == 9:
        #         x9 = x[:, 1:]  # [32， 768， 22， 11]  10th blocks输出
        #     flag += 1
        #     # return x  # 先通过L-1个transfermer Layer得到原文中隐藏特征Z(l-1)
        #
        # x1 = x1.transpose(1, 2)
        # x1 = self.adaptiveavgpooling(x1).transpose(1, 2)
        #
        # x3 = x3.transpose(1, 2)
        # x3 = self.adaptiveavgpooling(x3).transpose(1, 2)
        #
        # x_token = x[:, 0:1]
        # x_no_token = x[:, 1:]  # 11th block输出
        # x_combine = torch.cat((x1, x3, x9, x_no_token), 2)
        # x = self.conv1x1_(x_combine)
        # # x = x.permute(0, 2, 3, 1).reshape(B, H * W, 768)
        # x = torch.cat((x_token, x), 1)
        # return x







        B = x.shape[0]                                             ###  实验四！！！！！！！！！
        W = int(((x.shape[1] - 1) / 2.0) ** 0.5)  # W：11
        H = int((x.shape[1] - 1) / W)  # H：22

        flag = 0
        # if self.local_feature:
        for blk in self.blocks[:-1]:
            x = blk(x)
            # x, index_visual = blk(x)  # x:[32, 243, 768], orth_proto:[32, 12, 242]  # 做可视化时放开
            if flag == 1:
                x1 = x[:, -242:].transpose(1, 2).reshape(B, -1, H, W)  # [32， 768， 22， 11]  # 对应原文除class_token与最后一个block输出做组合  2th block输出

            elif flag == 3:
                x3 = x[:, -242:].transpose(1, 2).reshape(B, -1, H, W)  # [32， 768， 22， 11]  4th block输出
                # index_visual1 = index_visual  # 做可视化时放开
            # elif flag ==6:
            #     index_visual1 = index_visual
            elif flag == 9:
                x9 = x[:, -242:].transpose(1, 2).reshape(B, -1, H, W)  # [32， 768， 22， 11]  10th blocks输出
                # index_visual1 = index_visual
            flag += 1
            # return x  # 先通过L-1个transfermer Layer得到原文中隐藏特征Z(l-1)

        x_token = x[:, 0:-242]
        x_no_token = x[:, -242:].transpose(1, 2).reshape(B, -1, H, W)  # 11th block输出

        x_combine = torch.cat((x1, x3, x9, x_no_token), 1)
        x = self.conv1x1(x_combine)
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, 768)
        x = torch.cat((x_token, x), 1)
        # return x, index_visual1  # 做可视化时放开
        return x





        # for i, blk in enumerate(self.blocks[:-1]):                               # 原文
        #     x = blk(x)
        # return x


    def forward(self, x, cam_label=None, view_label=None, keep_rate=None, tokens=None, get_idx=True):
        # print(x.shape,cam_label.shape,view_label.shape)
        # x, index_visual = self.forward_features(x, cam_label, view_label, keep_rate, tokens, get_idx)  # 通过transreid的输出[64, 243, 768]
        x = self.forward_features(x, cam_label, view_label, keep_rate, tokens, get_idx)
        # if self.head_dist is not None:
        #     x, x_dist = self.fc(x[0]), self.head_dist(x[1])  # x must be a tuple
        #     if self.training and not torch.jit.is_scripting():
        #         # during inference, return the average of both classifier predictions
        #         return x, x_dist
        #     else:
        #         return (x + x_dist) / 2
        # else:
        #     x = self.head(x)
        # if get_idx:
        #     return x, idxs
        # return x, index_visual   # 做可视化时放开
        return x



    def load_param(self, model_path):
        param_dict = torch.load(model_path, map_location='cpu')
        if 'model' in param_dict:
            param_dict = param_dict['model']
        if 'state_dict' in param_dict:
            param_dict = param_dict['state_dict']
        for k, v in param_dict.items():
            if 'head' in k or 'dist' in k:
                continue
            if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
                # For old models that I trained prior to conv based patchification
                O, I, H, W = self.patch_embed.proj.weight.shape
                v = v.reshape(O, -1, H, W)
            elif k == 'pos_embed' and v.shape != self.pos_embed.shape:
                # To resize pos embedding when using model at different size from pretrained weights
                if 'distilled' in model_path:
                    print('distill need to choose right cls token in the pth')
                    v = torch.cat([v[:, 0:1], v[:, 2:]], dim=1)
                v = resize_pos_embed(v, self.pos_embed, self.patch_embed.num_y, self.patch_embed.num_x)
            try:
                self.state_dict()[k].copy_(v)
            except:
                print('===========================ERROR=========================')
                print('shape do not match in k :{}: param_dict{} vs self.state_dict(){}'.format(k, v.shape, self.state_dict()[k].shape))


def resize_pos_embed(posemb, posemb_new, hight, width):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    ntok_new = posemb_new.shape[1]

    posemb_token, posemb_grid = posemb[:, :1], posemb[0, 1:]
    ntok_new -= 1

    gs_old = int(math.sqrt(len(posemb_grid)))
    print('Resized position embedding from size:{} to size: {} with height:{} width: {}'.format(posemb.shape, posemb_new.shape, hight, width))
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(hight, width), mode='bilinear')
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, hight * width, -1)
    posemb = torch.cat([posemb_token, posemb_grid], dim=1)
    return posemb


def vit_base_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_rate=0.0, attn_drop_rate=0.0, drop_path_rate=0.1, camera=0, view=0, local_feature=False, sie_xishu=1.5, **kwargs):
    model = TransReID(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
        camera=camera, view=view, drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  sie_xishu=sie_xishu, local_feature=local_feature, **kwargs)

    return model

def vit_small_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_rate=0., attn_drop_rate=0.,drop_path_rate=0.1, camera=0, view=0, local_feature=False, sie_xishu=1.5, **kwargs):
    kwargs.setdefault('qk_scale', 768 ** -0.5)
    model = TransReID(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=768, depth=8, num_heads=8,  mlp_ratio=3., qkv_bias=False, drop_path_rate=drop_path_rate,\
        camera=camera, view=view,  drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  sie_xishu=sie_xishu, local_feature=local_feature, **kwargs)

    return model

def deit_small_patch16_224_TransReID(img_size=(256, 128), stride_size=16, drop_path_rate=0.1, drop_rate=0.0, attn_drop_rate=0.0, camera=0, view=0, local_feature=False, sie_xishu=1.5, **kwargs):
    model = TransReID(
        img_size=img_size, patch_size=16, stride_size=stride_size, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
        drop_path_rate=drop_path_rate, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, camera=camera, view=view, sie_xishu=sie_xishu, local_feature=local_feature,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)

    return model


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        print("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)

# if __name__ == '__main__':
#     import torch
#     import torch.nn as nn
#     model = TransReID(img_size=[256, 128], sie_xishu=3.0,
#                                                     local_feature=True, camera=8, view=1,
#                                                     stride_size=[11, 11],
#                                                     drop_path_rate=0.1, distilled=False,
#                                                     fuse_token=True,
#                                                     keep_rate=(1, 1, 1, 0.9, 1, 1, 0.9, 1, 1, 0.9, 1, 1))
#     x = torch.rand([32, 3, 256, 128])
#     y = torch.rand([32])
#     z = torch.rand([32])
#     model(x,y,z)