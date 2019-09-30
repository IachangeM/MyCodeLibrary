# -*- coding: utf-8 -*-

"""
如果模型输入的Size尺寸不固定，有几种方法可以解决：
    方案1：对输入进行resize，统一到同一大小。
    方案2：取消全连接层，对最后的卷积层Global Average Pooling（GAP）。
    方案3：在第一个全连接层前，加入SPP layer。本代码。

Spatial Pyramid Pooling 空间金字塔池化。
论文题目：Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
论文地址：https://arxiv.org/abs/1406.4729
代码实现：https://github.com/yueruchen/sppnet-pytorch
本代码来自:https://github.com/yueruchen/sppnet-pytorch/blob/master/spp_layer.py

【注意】
    在做金字塔卷积的时候，padding未必一定是0，要按照padding填充之后的Size计算Output...
    计算公式如下(在height为例)：
        kernel_size = input_height_size / height_out   -> 向上取整!!     height_out是池化后输出的一个特征图的height_out
        stride = kernel_size
        padding = (kernel_size * height_out - input_height_size) / 2  -> 向下取整!!!

演算，SPPLayer输入7*7*512或者5*5*512之后的输出都是(4*4+2*2+1）*512 = 21*512  : 4、2、1是将输入分成3个尺寸的块 然后融合。
SPM其实在传统的机器学习特征提取中很常用，主要思路就是对于一副图像分成若干尺度的一些块，比如一幅图像分成1份，4份，8份等。
然后对于每一块提取特征然后融合在一起，这样就可以兼容多个尺度的特征（所以从本质上讲，这种创新还是源于传统的机器学习：尺度金字塔+特征融合）。
SPPNet首次将这种思想应用在CNN中，对于卷积层特征我们首先对他分成不同的尺寸，然后每个尺寸提取一个固定维度的特征，最后拼接这些特征成一个固定维度
---------------------
作者：沈子恒
来源：CSDN
原文：https://blog.csdn.net/shenziheng1/article/details/82504615
版权声明：本文为博主原创文章，转载请附上博文链接！

"""


import math
import torch
import torch.nn as nn


def spatial_pyramid_pool(previous_conv, num_sample, previous_conv_size, out_pool_size):
    """
    previous_conv: a tensor vector of previous convolution layer
    num_sample: an int number of image in the batch
    previous_conv_size: an int vector [height, width] of the matrix features size of previous convolution layer
    out_pool_size: a int vector of expected output size of max pooling layer

    returns: a tensor vector with shape [1 x n] is the concentration of multi-level pooling
    """
    # print(previous_conv.size())
    spp = None
    for i in range(len(out_pool_size)):
        # print(previous_conv_size)
        h_wid = int(math.ceil(previous_conv_size[0] / out_pool_size[i]))
        w_wid = int(math.ceil(previous_conv_size[1] / out_pool_size[i]))
        h_pad = (h_wid * out_pool_size[i] - previous_conv_size[0] + 1) / 2
        w_pad = (w_wid * out_pool_size[i] - previous_conv_size[1] + 1) / 2
        maxpool = nn.MaxPool2d((h_wid, w_wid), stride=(h_wid, w_wid), padding=(h_pad, w_pad))
        x = maxpool(previous_conv)
        if i == 0:
            spp = x.view(num_sample, -1)
            # print("spp size:",spp.size())
        else:
            # print("size:",spp.size())
            spp = torch.cat((spp, x.view(num_sample, -1)), 1)
    return spp


if __name__ == '__main__':
    pass

