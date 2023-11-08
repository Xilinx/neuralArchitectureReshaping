# basemodel.py -*- Python -*-
#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2023 Advanced Micro Devices, Inc.

"""
QABAS Design Space
"""

import numpy as np
from bonito.nn import Permute, layers
import torch
import nni.retiarii.nn.pytorch as nn
from torch.nn.functional import log_softmax, ctc_loss
from fast_ctc_decode import beam_search, viterbi_search
from nni.retiarii.nn.pytorch import LayerChoice
from collections import OrderedDict
from nni.retiarii.serializer import basic_unit
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.quant import Int8Bias as BiasQuant
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8Bias
from brevitas.core.scaling import ScalingImplType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.stats import StatsOp

class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        if 'qscore' not in config:
            self.qbias = 0.0
            self.qscale = 1.0
        else:
            self.qbias = config['qscore']['bias']
            self.qscale = config['qscore']['scale']

        self.config = config
        self.stride = config['block'][0]['stride'][0]
        self.alphabet = config['labels']['labels']
        self.features = 48
        self.encoder = Encoder(config)
        self.decoder = Decoder(48, len(self.alphabet))

    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded)

    def decode(self, x, beamsize=5, threshold=1e-3, qscores=False, return_path=False):
        x = x.exp().cpu().detach().numpy().astype(np.float32)
        if beamsize == 1 or qscores:
            seq, path  = viterbi_search(x, self.alphabet, qscores, self.qscale, self.qbias)
        else:
            seq, path = beam_search(x, self.alphabet, beamsize, threshold)
        if return_path: return seq, path
        return seq

    def ctc_label_smoothing_loss(self, log_probs, targets, lengths, weights=None):
        T, N, C = log_probs.shape
        weights = weights or torch.cat([torch.tensor([0.4]), (0.1 / (C - 1)) * torch.ones(C - 1)])
        log_probs_lengths = torch.full(size=(N, ), fill_value=T, dtype=torch.int64)
        loss = ctc_loss(log_probs.to(torch.float32), targets, log_probs_lengths, lengths, reduction='mean')
        label_smoothing_loss = -((log_probs * weights.to(log_probs.device)).mean())
        return {'loss': loss + label_smoothing_loss, 'ctc_loss': loss, 'label_smooth_loss': label_smoothing_loss}

class Encoder(nn.Module):
    """
    Builds the model encoder
    """
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config

        features = self.config['input']['features']
        activation = layers[self.config['encoder']['activation']]()
        encoder_layers = []

        choice_keys=[]

        encoder_layers.append(
                Block(
                     activation,
                    
                )
            )
        # for layer in self.config['block']:
        #     print("Flter:",layer['filters'])
        #     encoder_layers.append(
        #         Block(
        #             features, layer['filters'], activation,
        #             repeat=layer['repeat'], kernel_size=layer['kernel'],
        #             stride=layer['stride'], dilation=layer['dilation'],
        #             dropout=layer['dropout'], residual=layer['residual'],
        #             separable=layer['separable']
        #         )
        #     )

        #     features = layer['filters']
            

        self.encoder = nn.Sequential(*encoder_layers)

    def forward(self, x):
        
        return self.encoder(x)


OPS = {
    'Zero': lambda in_C, out_C, stride: ZeroLayer(stride=stride),
    'Identity': lambda in_C, out_C, stride: nn.Identity(in_C, out_C),
    '2x2_MBConv1': lambda in_C, out_C, act, stride: Block_repeat(in_C, out_C, act, 0.05, kernel_size=2,stride=stride, expand_ratio=1),
    '3x3_MBConv1': lambda in_C, out_C, act, stride: Block_repeat(in_C, out_C, act, 0.05, kernel_size=3,stride=stride, expand_ratio=1),
    '3x3_MBConv2': lambda in_C, out_C, act, stride: Block_repeat(in_C, out_C, act, 0.05, kernel_size=3,stride=stride, expand_ratio=2),
    '3x3_MBConv6': lambda in_C, out_C, act, stride: Block_repeat(in_C, out_C, act, 0.05, kernel_size=3,stride=stride, expand_ratio=6),
    '5x5_MBConv1': lambda in_C, out_C, act, stride: Block_repeat(in_C, out_C, act, 0.05, kernel_size=5,stride=stride, expand_ratio=1),
    '5x5_MBConv3': lambda in_C, out_C, act, stride: Block_repeat(in_C, out_C, act, 0.05, kernel_size=5,stride=stride, expand_ratio=3),
    '6x6_MBConv1': lambda in_C, out_C, act, stride: Block_repeat(in_C, out_C, act, 0.05, kernel_size=6,stride=stride, expand_ratio=1),
    '7x7_MBConv1': lambda in_C, out_C, act, stride: Block_repeat(in_C, out_C, act, 0.05, kernel_size=7,stride=stride, expand_ratio=1),
    '7x7_MBConv3': lambda in_C, out_C, act, stride: Block_repeat(in_C, out_C, act, 0.05, kernel_size=7,stride=stride, expand_ratio=3),
    '9x9_MBConv1': lambda in_C, out_C, act, stride: Block_repeat(in_C, out_C, act, 0.05, kernel_size=9,stride=stride, expand_ratio=1),
    '9x9_MBConv3': lambda in_C, out_C, act, stride: Block_repeat(in_C, out_C, act, 0.05, kernel_size=9,stride=stride, expand_ratio=3),
    '15x15_MBConv1': lambda in_C, out_C, act, stride: Block_repeat(in_C, out_C, act, 0.05, kernel_size=15,stride=stride, expand_ratio=1),
    '25x25_MBConv1': lambda in_C, out_C, act, stride: Block_repeat(in_C, out_C, act, 0.05, kernel_size=25,stride=stride, expand_ratio=1),
    '25x25_MBConv3': lambda in_C, out_C, act, stride: Block_repeat(in_C, out_C, act, 0.05, kernel_size=25,stride=stride, expand_ratio=3),
    '31x31_MBConv1': lambda in_C, out_C, act, stride: Block_repeat(in_C, out_C, act, 0.05, kernel_size=31,stride=stride, expand_ratio=1),
    '55x55_MBConv1': lambda in_C, out_C, act, stride: Block_repeat(in_C, out_C, act, 0.05, kernel_size=55,stride=stride, expand_ratio=1),
    '75x75_MBConv1': lambda in_C, out_C, act, stride: Block_repeat(in_C, out_C, act, 0.05, kernel_size=75,stride=stride, expand_ratio=1),
    '75x75_MBConv3': lambda in_C, out_C, act, stride: Block_repeat(in_C, out_C, act, 0.05, kernel_size=75,stride=stride, expand_ratio=3),
    '115x115_MBConv1': lambda in_C, out_C, act, stride: Block_repeat(in_C, out_C, act, 0.05, kernel_size=115,stride=stride, expand_ratio=1),
    '123x123_MBConv1': lambda in_C, out_C, act, stride: Block_repeat(in_C, out_C, act, 0.05, kernel_size=123,stride=stride, expand_ratio=1)
}
# in_channels, out_channels,activation, dropout, kernel_size=1, stride=1, dilation=1, padding=0, bias=False, separable=False
class Block(nn.Module):
    """
    TCSConv, Batch Normalisation, Activation, Dropout
    """
    def __init__(self, activation, in_channels=1, out_channels=344,  repeat=5, kernel_size=1, stride=1, dilation=1, dropout=0.0, residual=False, separable=False):

        super(Block, self).__init__()

        self.use_res = residual
        self.conv = nn.ModuleList()

        _in_channels = in_channels
        # padding = self.get_padding(kernel_size[0], stride[0], dilation[0])
        blocks=[]
        conv_blocks=[]

        first_block=Block_non_repeat(_in_channels, out_channels,activation,stride=3, kernel_size=9)
        _in_channels=out_channels
        blocks = [first_block]
        

        width_stage=[424]
        n_cell_stages=[6]


        ###########################B1###################################
        width_stage=[424]
        n_cell_stages=[4]
        # width_stage=[424]
        # n_cell_stages=[5]
        width_mult=1
        for i in range(len(width_stage)):
            width_stage[i] = self.make_divisible(width_stage[i] * width_mult, 8)
        # print(width_stage)
        
        for width,n_cell in zip(width_stage,n_cell_stages):
            # print(width)
            for i in range(n_cell):
                # print(i)
                op_candidates=[
                    OPS['3x3_MBConv1'](_in_channels, width, activation, stride), 
                    OPS['5x5_MBConv1'](_in_channels, width, activation, stride),
                    OPS['7x7_MBConv1'](_in_channels, width, activation, stride),
                    OPS['9x9_MBConv1'](_in_channels, width, activation, stride),
                    OPS['25x25_MBConv1'](_in_channels, width, activation, stride),
                    OPS['31x31_MBConv1'](_in_channels, width, activation, stride),
                    OPS['55x55_MBConv1'](_in_channels, width, activation, stride),
                    OPS['75x75_MBConv1'](_in_channels, width, activation, stride),
                    OPS['115x115_MBConv1'](_in_channels, width, activation, stride),
                    OPS['123x123_MBConv1'](_in_channels, width, activation, stride)]
                # op_candidates += [("identity1",nn.Identity(_in_channels,width))]
                # op_candidates += [("Zero",OPS['Zero'](_in_channels, width, stride))]
                if(i!=0):
                    op_candidates += [OPS['Zero'](_in_channels, width, stride)]
                    # op_candidates += [nn.Identity(_in_channels,width)]
                    conv_op =LayerChoice(op_candidates,label="B_{}_{}".format(width,i))
                    #  op_candidates += [("identity1",nn.Identity(_in_channels,width))]
                # shortcut=Block_non_repeat(_in_channels, width,activation)
                else:
                    conv_op =LayerChoice(op_candidates,label="B_{}_{}".format(width,i))
                
                if(i!=0)  and _in_channels != width: ## upscaling
                    shortcut=Block_non_repeat(_in_channels, width,activation)   
                elif(i!=0)  and _in_channels == width:
                    shortcut=IdentityLayer(_in_channels, _in_channels)
                else:
                    shortcut=None
                residual_block = InvertedResidualBlock(conv_op, shortcut, op_candidates)
                
                blocks.append(residual_block)
                _in_channels = width
        
        
        ###########################B2###################################
        width_stage=[464]
        n_cell_stages=[4]
        width_mult=1
        for i in range(len(width_stage)):
            width_stage[i] = self.make_divisible(width_stage[i] * width_mult, 8)
        # print(width_stage)
        
        for width,n_cell in zip(width_stage,n_cell_stages):
            # print(width)
            for i in range(n_cell):
                # print(i)
                op_candidates=[
                    OPS['3x3_MBConv1'](_in_channels, width, activation, stride), 
                    OPS['5x5_MBConv1'](_in_channels, width, activation, stride),
                    OPS['7x7_MBConv1'](_in_channels, width, activation, stride),
                    OPS['9x9_MBConv1'](_in_channels, width, activation, stride),
                    OPS['25x25_MBConv1'](_in_channels, width, activation, stride),
                    OPS['31x31_MBConv1'](_in_channels, width, activation, stride),
                    OPS['55x55_MBConv1'](_in_channels, width, activation, stride),
                    OPS['75x75_MBConv1'](_in_channels, width, activation, stride),
                    OPS['115x115_MBConv1'](_in_channels, width, activation, stride),
                    OPS['123x123_MBConv1'](_in_channels, width, activation, stride)]
                # op_candidates += [("identity1",nn.Identity(_in_channels,width))]
                # op_candidates += [("Zero",OPS['Zero'](_in_channels, width, stride))]
                
                if(i!=0):
                    op_candidates += [OPS['Zero'](_in_channels, width, stride)]
                    # op_candidates += [nn.Identity(_in_channels,width)]
                    conv_op =LayerChoice(op_candidates,label="B_{}_{}".format(width,i))
                    #  op_candidates += [("identity1",nn.Identity(_in_channels,width))]
                # shortcut=Block_non_repeat(_in_channels, width,activation)
                else:
                    conv_op =LayerChoice(op_candidates,label="B_{}_{}".format(width,i))
                
                if(i!=0)  and _in_channels != width: ## upscaling
                    shortcut=Block_non_repeat(_in_channels, width,activation)   
                elif(i!=0)  and _in_channels == width:
                    shortcut=IdentityLayer(_in_channels, _in_channels)
                else:
                    shortcut=None
                residual_block = InvertedResidualBlock(conv_op, shortcut, op_candidates)
                
                blocks.append(residual_block)
                _in_channels = width
        ########B3################
        width_stage=[456]
        n_cell_stages=[4]
        width_mult=1
        for i in range(len(width_stage)):
            width_stage[i] = self.make_divisible(width_stage[i] * width_mult, 8)        
        for width,n_cell in zip(width_stage,n_cell_stages):
            # print(width)
            for i in range(n_cell):
                # print(i)
                op_candidates=[OPS['3x3_MBConv1'](_in_channels, width, activation, stride), 
                    OPS['5x5_MBConv1'](_in_channels, width, activation, stride),
                    OPS['7x7_MBConv1'](_in_channels, width, activation, stride),
                    OPS['9x9_MBConv1'](_in_channels, width, activation, stride),
                    OPS['25x25_MBConv1'](_in_channels, width, activation, stride),
                    OPS['31x31_MBConv1'](_in_channels, width, activation, stride),
                    OPS['55x55_MBConv1'](_in_channels, width, activation, stride),
                    OPS['75x75_MBConv1'](_in_channels, width, activation, stride),
                    OPS['115x115_MBConv1'](_in_channels, width, activation, stride),
                    OPS['123x123_MBConv1'](_in_channels, width, activation, stride)]
                # op_candidates += [("identity1",nn.Identity(_in_channels,width))]
                # op_candidates += [("Zero",OPS['Zero'](_in_channels, width, stride))]
                
                if(i!=0):
                    op_candidates += [OPS['Zero'](_in_channels, width, stride)]
                    # op_candidates += [nn.Identity(_in_channels,width)]
                    conv_op =LayerChoice(op_candidates,label="B_{}_{}".format(width,i))
                    #  op_candidates += [("identity1",nn.Identity(_in_channels,width))]
                # shortcut=Block_non_repeat(_in_channels, width,activation)
                else:
                    conv_op =LayerChoice(op_candidates,label="B_{}_{}".format(width,i))
                
                if(i!=0)  and _in_channels != width: ## upscaling
                    shortcut=Block_non_repeat(_in_channels, width,activation)   
                elif(i!=0)  and _in_channels == width:
                    shortcut=IdentityLayer(_in_channels, _in_channels)
                else:
                    shortcut=None
                residual_block = InvertedResidualBlock(conv_op, shortcut, op_candidates)
                
                blocks.append(residual_block)
                _in_channels = width


        ########B4################
        width_stage=[440]
        n_cell_stages=[4]
        width_mult=1
        for i in range(len(width_stage)):
            width_stage[i] = self.make_divisible(width_stage[i] * width_mult, 8)        
        for width,n_cell in zip(width_stage,n_cell_stages):
            # print(width)
            for i in range(n_cell):
                # print(i)
                op_candidates=[
                    OPS['3x3_MBConv1'](_in_channels, width, activation, stride), 
                    OPS['5x5_MBConv1'](_in_channels, width, activation, stride),
                    OPS['7x7_MBConv1'](_in_channels, width, activation, stride),
                    OPS['9x9_MBConv1'](_in_channels, width, activation, stride),
                    OPS['25x25_MBConv1'](_in_channels, width, activation, stride),
                    OPS['31x31_MBConv1'](_in_channels, width, activation, stride),
                    OPS['55x55_MBConv1'](_in_channels, width, activation, stride),
                    OPS['75x75_MBConv1'](_in_channels, width, activation, stride),
                    OPS['115x115_MBConv1'](_in_channels, width, activation, stride),
                    OPS['123x123_MBConv1'](_in_channels, width, activation, stride)]
                # op_candidates += [("identity1",nn.Identity(_in_channels,width))]
                # op_candidates += [("Zero",OPS['Zero'](_in_channels, width, stride))]
                
                if(i!=0):
                    op_candidates += [OPS['Zero'](_in_channels, width, stride)]
                    # op_candidates += [nn.Identity(_in_channels,width)]
                    conv_op =LayerChoice(op_candidates,label="B_{}_{}".format(width,i))
                    #  op_candidates += [("identity1",nn.Identity(_in_channels,width))]
                # shortcut=Block_non_repeat(_in_channels, width,activation)
                else:
                    conv_op =LayerChoice(op_candidates,label="B_{}_{}".format(width,i))
                
                if(i!=0)  and _in_channels != width: ## upscaling
                    shortcut=Block_non_repeat(_in_channels, width,activation)   
                elif(i!=0)  and _in_channels == width:
                    shortcut=IdentityLayer(_in_channels, _in_channels)
                else:
                    shortcut=None
                residual_block = InvertedResidualBlock(conv_op, shortcut, op_candidates)
                
                blocks.append(residual_block)
                _in_channels = width

        ########B5################
        width_stage=[280]
        n_cell_stages=[4]
        width_mult=1
        for i in range(len(width_stage)):
            width_stage[i] = self.make_divisible(width_stage[i] * width_mult, 8)        
        for width,n_cell in zip(width_stage,n_cell_stages):
            # print(width)
            for i in range(n_cell):
                # print(i)
                op_candidates=[OPS['3x3_MBConv1'](_in_channels, width, activation, stride), 
                    OPS['5x5_MBConv1'](_in_channels, width, activation, stride),
                    OPS['7x7_MBConv1'](_in_channels, width, activation, stride),
                    OPS['9x9_MBConv1'](_in_channels, width, activation, stride),
                    OPS['25x25_MBConv1'](_in_channels, width, activation, stride),
                    OPS['31x31_MBConv1'](_in_channels, width, activation, stride),
                    OPS['55x55_MBConv1'](_in_channels, width, activation, stride),
                    OPS['75x75_MBConv1'](_in_channels, width, activation, stride),
                    OPS['115x115_MBConv1'](_in_channels, width, activation, stride),
                    OPS['123x123_MBConv1'](_in_channels, width, activation, stride)]
                # op_candidates += [("identity1",nn.Identity(_in_channels,width))]
                # op_candidates += [("Zero",OPS['Zero'](_in_channels, width, stride))]
                
                if(i!=0):
                    op_candidates += [OPS['Zero'](_in_channels, width, stride)]
                    # op_candidates += [nn.Identity(_in_channels,width)]
                    conv_op =LayerChoice(op_candidates,label="B_{}_{}".format(width,i))
                    #  op_candidates += [("identity1",nn.Identity(_in_channels,width))]
                # shortcut=Block_non_repeat(_in_channels, width,activation)
                else:
                    conv_op =LayerChoice(op_candidates,label="B_{}_{}".format(width,i))
                
                if(i!=0)  and _in_channels != width: ## upscaling
                    shortcut=Block_non_repeat(_in_channels, width,activation)   
                elif(i!=0)  and _in_channels == width:
                    shortcut=IdentityLayer(_in_channels, _in_channels)
                else:
                    shortcut=None
                residual_block = InvertedResidualBlock(conv_op, shortcut, op_candidates)
                
                blocks.append(residual_block)
                _in_channels = width
        second_last_block=Block_repeat(_in_channels, 384,activation,0.05, kernel_size=67) #separable block
        blocks.append(second_last_block)
        last_block=Block_non_repeat(384,48,activation, kernel_size=15)
        blocks.append(last_block)
        self.blocks = nn.ModuleList(blocks)

    def get_activation(self, activation, dropout):
        # print(activation)
        # print(Dropout)
        return activation, nn.Dropout(p=dropout)

    def make_divisible(self, v, divisor, min_val=None):
        """
        This function is taken from the original tf repo.
        It ensures that all layers have a channel number that is divisible by 8
        It can be seen here:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        """
        if min_val is None:
            min_val = divisor
        new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    def forward(self, x):
        _x = x
        # print("input",_x.size())
        for layer in self.blocks:
            # print(_x.size())
            _x = layer(_x)
            
        return (_x) 


class Block_repeat(nn.Module):
    def __init__(self, in_channels, out_channels,activation, dropout, kernel_size=1, stride=1,expand_ratio=6,  dilation=1, padding=0, bias=False, separable=False):
        super(Block_repeat, self).__init__()
        # padding = self.get_padding(kernel_size, stride[0], dilation)
        padding = self.get_same_padding(kernel_size)
        # print("padding:::",padding)
        stride=1
        
        # feature_dim = self.make_divisible(in_channels/2, 8)
        
        feature_dim=round(in_channels)

        self.dwconv = nn.Conv1d(
        feature_dim, feature_dim, kernel_size=kernel_size, stride=stride,
        # padding=padding, dilation=dilation, bias=bias, groups=in_channels//8
        padding=padding, dilation=dilation, bias=bias, groups=feature_dim//8
        )

        self.conv = nn.Conv1d(
        feature_dim, out_channels, kernel_size=1, stride=1,
        dilation=dilation, bias=bias, padding=0
        )      

        self.bn= nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        self.act=  nn.ReLU(inplace=True)
        self.drop= nn.Dropout(p=dropout)

    def make_divisible(self, v, divisor, min_val=None):
        """
        This function is taken from the original tf repo.
        It ensures that all layers have a channel number that is divisible by 8
        It can be seen here:
        https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        """
        if min_val is None:
            min_val = divisor
        new_v = max(min_val, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v
    def forward(self, x):
        x=self.dwconv(x)
        x=self.conv(x)
        x=self.bn(x)
        x=self.act(x)
        return self.drop(x)
    @staticmethod
    def is_zero_layer():
        return False
    def get_padding(self, kernel_size, stride, dilation):
        if stride > 1 and dilation > 1:
            raise ValueError("Dilation and stride can not both be greater than 1")
        return (kernel_size // 2) * dilation

    def get_same_padding(self,kernel_size):
        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
            p1 = get_same_padding(kernel_size[0])
            p2 = get_same_padding(kernel_size[1])
            return p1, p2
        assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
        assert kernel_size % 2 > 0, 'kernel size should be odd number'
        return kernel_size // 2


class Block_non_repeat(nn.Module):
    def __init__(self, in_channels, out_channels,activation, dropout=0.05, kernel_size=1, stride=1,expand_ratio=6,  dilation=1, padding=0, bias=False, separable=False):
        super(Block_non_repeat, self).__init__()
        # padding = self.get_padding(kernel_size, stride[0], dilation)
        padding = self.get_same_padding(kernel_size)
       
        # feature_dim = round(in_channels * expand_ratio)
        
        feature_dim=in_channels
        self.conv = nn.Conv1d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, bias=bias
            )
        self.bn= nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        self.act=  nn.ReLU(inplace=True)
        self.drop= nn.Dropout(p=dropout)

        # self.blockconv = nn.Sequential(OrderedDict([
        #     ('conv', nn.Conv1d(
        #         in_channels, out_channels, kernel_size=kernel_size,
        #         stride=stride, padding=padding, dilation=dilation, bias=bias
        #     )),
        #     ('bn', nn.BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)),
        #     ('act', nn.ReLU(inplace=True)),
        #     ('dropout',nn.Dropout(p=dropout))
        # ]))
        # ,
            # self.get_activation(activation, dropout)
    def forward(self, x):
        # x = self.blockconv(x)
        # return x
        x=self.conv(x)
        x=self.bn(x)
        x=self.act(x)
        return self.drop(x)
    @staticmethod
    def is_zero_layer():
        return False
    def get_padding(self, kernel_size, stride, dilation):
        if stride > 1 and dilation > 1:
            raise ValueError("Dilation and stride can not both be greater than 1")
        return (kernel_size // 2) * dilation

    def get_same_padding(self,kernel_size):
        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2, 'invalid kernel size: %s' % kernel_size
            p1 = get_same_padding(kernel_size[0])
            p2 = get_same_padding(kernel_size[1])
            return p1, p2
        assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
        assert kernel_size % 2 > 0, 'kernel size should be odd number'
        return kernel_size // 2
class Decoder(nn.Module):
    """
    Decoder
    """
    def __init__(self, features, classes):
        super(Decoder, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(features, classes, kernel_size=1, bias=True),
            Permute([2, 0, 1])
        )

    def forward(self, x):
        return log_softmax(self.layers(x), dim=-1)


class InvertedResidualBlock(nn.Module):
    
    def __init__(self, inverted_conv, shortcut, op_candidates_list):
        super(InvertedResidualBlock, self).__init__()

        self.inverted_conv = inverted_conv
        self.op_candidates_list = op_candidates_list
        self.zero_layer_module = ZeroLayerModule(shortcut)

    def forward(self, x):
        out = self.inverted_conv(x)
        return self.zero_layer_module(x, out)

@basic_unit
class ZeroLayerModule(nn.Module):
    def __init__(self, shortcut):
        super().__init__()
        self.shortcut = shortcut
        
    def forward(self, x, out):
        if torch.sum(torch.abs(out)).item() == 0:
            if x.size() == out.size():
                # is zero layer
                return x
        if self.shortcut is None:
            return out
        return out + self.shortcut(x)


class Base2DLayer(nn.Module):
    
    def __init__(self, in_channels, out_channels,act_func,
                 use_bn=True,  dropout_rate=0, ops_order='weight_bn_act'):
        super(Base2DLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.use_bn = use_bn
        self.act_func = act_func
        self.dropout_rate = dropout_rate
        self.ops_order = ops_order

        """ modules """
        modules = {}
        # batch norm
        if self.use_bn:
            if self.bn_before_weight:
                modules['bn'] = nn.BatchNorm1d(in_channels)
            else:
                modules['bn'] = nn.BatchNorm1d(out_channels)
        else:
            modules['bn'] = None
        # activation
        modules['act'] = act_func
        # dropout
        if self.dropout_rate > 0:
            modules['dropout'] = nn.Dropout(self.dropout_rate)
        else:
            modules['dropout'] = None
        # weight
        modules['weight'] = self.weight_op()

        # add modules
        for op in self.ops_list:
            if modules[op] is None:
                continue
            elif op == 'weight':
                if modules['dropout'] is not None:
                    self.add_module('dropout', modules['dropout'])
                for key in modules['weight']:
                    self.add_module(key, modules['weight'][key])
            else:
                self.add_module(op, modules[op])
        self.sequence = nn.Sequential(self._modules)

    @property
    def ops_list(self):
        return self.ops_order.split('_')

    @property
    def bn_before_weight(self):
        for op in self.ops_list:
            if op == 'bn':
                return True
            elif op == 'weight':
                return False
        raise ValueError(f'Invalid ops_order: {self.ops_order}')

    def weight_op(self):
        raise NotImplementedError

    def forward(self, x):
        x = self.sequence(x)
        return x

    @staticmethod
    def is_zero_layer():
        return False


class IdentityLayer(Base2DLayer):

    def __init__(self, in_channels, out_channels,
                 use_bn=False, act_func=None, dropout_rate=0, ops_order='weight_bn_act'):
        super(IdentityLayer, self).__init__(in_channels, out_channels,act_func, use_bn,  dropout_rate, ops_order)

    def weight_op(self):
        return None



class ZeroLayer(nn.Module):

    def __init__(self, stride):
        super(ZeroLayer, self).__init__()
        self.stride = stride

    def forward(self, x):
        '''n, c, h, w = x.size()
        h //= self.stride
        w //= self.stride
        device = x.get_device() if x.is_cuda else torch.device('cpu')
        # noinspection PyUnresolvedReferences
        padding = torch.zeros(n, c, h, w, device=device, requires_grad=False)
        return padding'''
        return x * 0

    @staticmethod
    def is_zero_layer():
        return True
