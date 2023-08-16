# rubiconskiptrim.py -*- Python -*-
#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2023 Advanced Micro Devices, Inc.

"""
RUBICON Model Quantized Model After QABAS For SkipTrim
"""
import numpy as np
from bonito.nn import Permute, layers
import torch
from torch.nn.functional import log_softmax, ctc_loss
from torch.nn import Module, ModuleList, Sequential, Conv1d, BatchNorm1d, Dropout
from fast_ctc_decode import beam_search, viterbi_search
import brevitas.nn as qnn
from brevitas.core.quant import QuantType
from brevitas.quant import Int8Bias as BiasQuant
from brevitas.quant.scaled_int import Int8ActPerTensorFloat
from brevitas.quant.scaled_int import Int8Bias
from brevitas.core.scaling import ScalingImplType
from brevitas.core.restrict_val import RestrictValueType
from brevitas.core.stats import StatsOp


from brevitas.quant import Int8WeightPerTensorFixedPoint as WeightQuant
from brevitas.quant import Int8ActPerTensorFixedPoint as ActQuant
from brevitas.quant import Int8BiasPerTensorFixedPointInternalScaling as BiasQuant

class RubiconSkipTrim(Module):
    def __init__(self, config):
        super(RubiconSkipTrim, self).__init__()
        if 'qscore' not in config:
            self.qbias = 0.0
            self.qscale = 1.0
        else:
            self.qbias = config['qscore']['bias']
            self.qscale = config['qscore']['scale']

        self.config = config
        self.stride = config['block'][0]['stride'][0]
        self.alphabet = config['labels']['labels']
        self.features =48
        self.encoder = Encoder(config)
        self.decoder = Decoder(48, len(self.alphabet))
        self.quant_identity=qnn.QuantIdentity(
             bit_width=16, return_quant_tensor=True)

    def forward(self, x):
        encoded = self.encoder(self.quant_identity(x))
        return self.decoder(encoded)

    def decode(self, x, beamsize=5, threshold=1e-3, qscores=False, return_path=False):
        x = x.exp().cpu().numpy().astype(np.float32)
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


class Encoder(Module):
    """
    Builds the model encoder
    """
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.config = config
        features = self.config['input']['features']
        activation = layers[self.config['encoder']['activation']]()
        encoder_layers = []
        layer_list=[]
        b1_b=[]
        b2_b=[]
        b3_b=[]
        b4_b=[]
        b5_b=[]
        c1_b=Block( 1, 208, activation,
                        repeat=1, kernel_size=9,
                        stride=3, dilation=1,
                        dropout=0.05, residual=False,
                        separable=False,
                        quant=16,quant_act=16)
        b10_b=Block_repeat_b1( 208, 280, activation, 
                repeat=2, kernel_size=75,
                stride=1, dilation=1,
                dropout=0.05, residual=True,
                separable=True,
                quant=16,quant_act=16)
        b11_b=Block_repeat_b1( 280, 280, activation, 
                    repeat=1, kernel_size=31,
                    stride=1, dilation=1,
                    dropout=0.05, residual=True,
                    separable=True,
                    quant=8,quant_act=8)
        b14_b=Block_repeat_b1( 280, 296, activation,
                    repeat=2, kernel_size=31,
                    stride=1, dilation=1,
                    dropout=0.05, residual=True,
                    separable=True,
                    quant=8,quant_act=8)
        b23_b=Block_repeat_b2( 296, 232, activation,
                    repeat=1, kernel_size=123,
                    stride=1, dilation=1,
                    dropout=0.05, residual=True,
                    separable=True,
                    quant=16,quant_act=8)

        b31_b=Block_repeat_b3( 232,232, activation,
                    repeat=2, kernel_size=55,
                    stride=1, dilation=1,
                    dropout=0.05, residual=True,
                    separable=True,
                    quant=16,quant_act=8)
        b34_b=Block_repeat_b3( 232,232, activation,
                    repeat=2, kernel_size=5,
                    stride=1, dilation=1,
                    dropout=0.05, residual=True,
                    separable=True,
                    quant=16,quant_act=8)
        b35_b=Block_repeat_b3( 232,224, activation,
                    repeat=2, kernel_size=3,
                    stride=1, dilation=1,
                    dropout=0.05, residual=True,
                    separable=True,
                    quant=8,quant_act=8)
        b40_b=Block_repeat_b4( 224,224, activation,
                    repeat=2, kernel_size=9,
                    stride=1, dilation=1,
                    dropout=0.05, residual=True,
                    separable=True,
                    quant=16,quant_act=8)

        b41_b=Block_repeat_b4( 224,224, activation,
                    repeat=2, kernel_size=115,
                    stride=1, dilation=1,
                    dropout=0.05, residual=True,
                    separable=True,
                    quant=16,quant_act=8)
        b43_b=Block_repeat_b4( 224,224, activation,
                    repeat=2, kernel_size=55,
                    stride=1, dilation=1,
                    dropout=0.05, residual=True,
                    separable=True,
                    quant=16,quant_act=8)
        b44_b=Block_repeat_b4( 224,144, activation,
                    repeat=2, kernel_size=25,
                    stride=1, dilation=1,
                    dropout=0.05, residual=True,
                    separable=True,
                    quant=16,quant_act=8)

        b51_b=Block_repeat_b5( 144,144 ,activation,
                    repeat=2, kernel_size=9,
                    stride=1, dilation=1,
                    dropout=0.05, residual=True,
                    separable=True,
                    quant=16,quant_act=8)

        b52_b=Block_repeat_b5( 144,144 ,activation,
                    repeat=1, kernel_size=25,
                    stride=1, dilation=1,
                    dropout=0.05, residual=True,
                    separable=True,
                    quant=8,quant_act=8)

        b54_b=Block_repeat_b5( 144,144 ,activation,
                    repeat=2, kernel_size=7,
                    stride=1, dilation=1,
                    dropout=0.05, residual=True,
                    separable=True,
                    quant=8,quant_act=8)
        b55_b=Block_repeat_b5( 144,144 ,activation,
                    repeat=1, kernel_size=123,
                    stride=1, dilation=1,
                    dropout=0.05, residual=True,
                    separable=True,
                    quant=8,quant_act=4)
        c2_b=Block( 144,152, activation,
                    repeat=1, kernel_size=9,
                    stride=1, dilation=1,
                    dropout=0.05, residual=False,
                    separable=True,
                    quant=8,quant_act=4)
        c3_b=Block( 152,48, activation,
                    repeat=1, kernel_size=9,
                    stride=1, dilation=1,
                    dropout=0.05, residual=False,
                    separable=False,
                    quant=8,quant_act=4)

        encoder_layers.append(c1_b)
        encoder_layers.append(b10_b)
        encoder_layers.append(b11_b)
        encoder_layers.append(b14_b)
        encoder_layers.append(b23_b)
        encoder_layers.append(b31_b)
        encoder_layers.append(b34_b)
        encoder_layers.append(b35_b)
        encoder_layers.append(b40_b)
        encoder_layers.append(b41_b)
        encoder_layers.append(b43_b)
        encoder_layers.append(b44_b)
        encoder_layers.append(b51_b)
        encoder_layers.append(b52_b)
        encoder_layers.append(b54_b)
        encoder_layers.append(b55_b)
        encoder_layers.append(c2_b)
        encoder_layers.append(c3_b)
        self.encoder = Sequential(*encoder_layers)

    def forward(self, x):
        return self.encoder(x)


class TCSConv1d_quant(Module):
    """
    Time-Channel Separable 1D Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, separable=False,quant=8,quant_act=8):

        super(TCSConv1d_quant, self).__init__()
        self.separable = separable

        if separable:
            self.depthwise = qnn.QuantConv1d(
                in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                padding=padding, dilation=dilation, bias=bias, groups=in_channels//8, 
                bias_quant=None,
                return_quant_tensor=False,
                weight_bit_width=quant
            )

            self.pointwise = qnn.QuantConv1d(
        in_channels, out_channels, kernel_size=1, stride=1,
        dilation=dilation, bias=bias, padding=0,
        bias_quant=None,
                return_quant_tensor=False,
                weight_bit_width=quant
        )      
            self.quant_identity=qnn.QuantIdentity(bit_width=quant, return_quant_tensor=True)
        else:
            self.conv = qnn.QuantConv1d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, bias=bias,
                bias_quant=None,
                return_quant_tensor=False,
                weight_bit_width=quant
            )
        

    def forward(self, x):
        if self.separable:
            x = self.depthwise(x)
            x = self.quant_identity(x)
            x = self.pointwise(x)
        else:
            x = self.conv(x)
        return x

class TCSConv1d(Module):
    """
    Time-Channel Separable 1D Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, separable=False):

        super(TCSConv1d, self).__init__()
        self.separable = separable

        if separable:
            self.depthwise = Conv1d(
                in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                # padding=padding, dilation=dilation, bias=bias, groups=in_channels//8
                padding=padding, dilation=dilation, bias=bias, groups=in_channels//8
            )

            self.pointwise = Conv1d(
                in_channels, out_channels, kernel_size=1, stride=1,
                dilation=dilation, bias=bias, padding=0
            )
        else:
            self.conv = Conv1d(
                in_channels, out_channels, kernel_size=kernel_size,
                stride=stride, padding=padding, dilation=dilation, bias=bias
            )

    def forward(self, x):
        if self.separable:
            x = self.depthwise(x)
            x = self.pointwise(x)
        else:
            x = self.conv(x)
        return x

class Block_repeat(Module):
    """
    TCSConv, Batch Normalisation, Activation, Dropout
    """
    def __init__(self, in_channels, out_channels, activation, repeat=5, kernel_size=1, stride=1, dilation=1, dropout=0.0, residual=False, separable=False,quant=8,quant_act=8):

        super(Block_repeat, self).__init__()

        self.use_res = residual
        self.conv = ModuleList()

        _in_channels = in_channels
        padding = self.get_padding(kernel_size, stride, dilation)

        # add the first n - 1 convolutions + activation
        for _ in range(repeat - 1):
            self.conv.extend(
                self.get_tcs_quant(
                    _in_channels, out_channels, kernel_size=kernel_size,
                    stride=stride, dilation=dilation,
                    padding=padding, separable=separable,
                    quant=quant,quant_act=quant_act
                )
            )

            self.conv.extend(self.get_activation_quant(activation, dropout,quant_act=quant_act))
            _in_channels = out_channels

        # add the last conv and batch norm
        self.conv.extend(
            self.get_tcs_quant(
                _in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride, dilation=dilation,
                padding=padding, separable=separable,
                    quant=quant,quant_act=quant_act
            )
        )
        self.conv.append(qnn.QuantIdentity(bit_width=quant))
        # add the residual connection
        if self.use_res:
            self.residual = Sequential(*self.get_tcs_quant(in_channels, out_channels,
                    quant=quant,quant_act=quant_act),qnn.QuantIdentity(bit_width=quant))

        # add the activation and dropout
        self.activation = Sequential(*self.get_activation_quant(activation, dropout,quant_act=quant_act))

    def get_activation(self, activation, dropout):
        return activation, Dropout(p=dropout)

    def get_activation_quant(self, activation, dropout,quant_act=8):
        return qnn.QuantReLU(inplace=True,return_quant_tensor=False,
                bit_width=quant_act), Dropout(p=dropout)
    def get_padding(self, kernel_size, stride, dilation):
        if stride > 1 and dilation > 1:
            raise ValueError("Dilation and stride can not both be greater than 1")
        return (kernel_size // 2) * dilation

    def get_tcs(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False, separable=False):
        return [
            TCSConv1d(
                in_channels, out_channels, kernel_size,
                stride=stride, dilation=dilation, padding=padding,
                bias=bias, separable=separable
            ),
            BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        ]

    def get_tcs_quant(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False, separable=False,quant=8,quant_act=8):
        return [
            TCSConv1d_quant(
                in_channels, out_channels, kernel_size,
                stride=stride, dilation=dilation, padding=padding,
                bias=bias, separable=separable,
                    quant=quant,quant_act=quant_act
            ),
            BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        ]


    def forward(self, x):
        _x = x
        for layer in self.conv:
            _x = layer(_x)
        if self.use_res:
            _x = _x + self.residual(x)
        return self.activation(_x)
class Block(Module):
    """
    TCSConv, Batch Normalisation, Activation, Dropout
    """
    def __init__(self, in_channels, out_channels, activation, repeat=5, kernel_size=1, stride=1, dilation=1, dropout=0.0, residual=False, separable=False,
                    quant=16,quant_act=16):

        super(Block, self).__init__()

        self.use_res = residual
        self.conv = ModuleList()

        _in_channels = in_channels
        padding = self.get_padding(kernel_size, stride, dilation)

        # add the first n - 1 convolutions + activation
        for _ in range(repeat - 1):
            self.conv.extend(
                self.get_tcs_quant(
                    _in_channels, out_channels, kernel_size=kernel_size,
                    stride=stride, dilation=dilation,
                    padding=padding, separable=separable,
                    quant=quant,quant_act=quant_act
                )
            )

            self.conv.extend(self.get_activation_quant(activation, dropout,quant_act=quant_act))
            _in_channels = out_channels

        # add the last conv and batch norm
        self.conv.extend(
            self.get_tcs_quant(
                _in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride, dilation=dilation,
                padding=padding, separable=separable,
                    quant=quant,quant_act=quant_act
            )
        )

        # add the residual connection
        if self.use_res:
            self.residual = Sequential(*self.get_tcs_quant(in_channels, out_channels,
                    quant=quant,quant_act=quant_act))

        # add the activation and dropout
        self.activation = Sequential(*self.get_activation_quant(activation, dropout,quant_act=quant_act))

    def get_activation_quant(self, activation, dropout,quant_act=8):
        return qnn.QuantReLU(inplace=True,return_quant_tensor=False,
                bit_width=quant_act), Dropout(p=dropout)

    def get_activation(self, activation, dropout):
        return activation, Dropout(p=dropout)

    def get_padding(self, kernel_size, stride, dilation):
        if stride > 1 and dilation > 1:
            raise ValueError("Dilation and stride can not both be greater than 1")
        return (kernel_size // 2) * dilation
    def get_tcs_quant(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False, separable=False,quant=8,quant_act=8):
        return [
            TCSConv1d_quant(
                in_channels, out_channels, kernel_size,
                stride=stride, dilation=dilation, padding=padding,
                bias=bias, separable=separable,
                    quant=quant,quant_act=quant_act
            ),
            BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        ]

    def get_tcs(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False, separable=False):
        return [
            TCSConv1d(
                in_channels, out_channels, kernel_size,
                stride=stride, dilation=dilation, padding=padding,
                bias=bias, separable=separable
            ),
            BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        ]

    def forward(self, x):
        _x = x
        for layer in self.conv:
            _x = layer(_x)
        if self.use_res:
            _x = _x + self.residual(x)
        return self.activation(_x)


class Decoder(Module):
    """
    Decoder
    """
    def __init__(self, features, classes):
        super(Decoder, self).__init__()
        self.layers = Sequential(
            qnn.QuantConv1d(features, classes, kernel_size=1, bias=True,
                bias_quant=None,
                return_quant_tensor=False,
                weight_bit_width=8),
            Permute([2, 0, 1])
        )

    def forward(self, x):
        return log_softmax(self.layers(x), dim=-1)

class Block_repeat_b1(Module):
    """
    TCSConv, Batch Normalisation, Activation, Dropout
    """
    def __init__(self, in_channels, out_channels, activation, repeat=5, kernel_size=1, stride=1, dilation=1, dropout=0.0, residual=False, separable=False,quant=8,quant_act=8):

        super(Block_repeat_b1, self).__init__()

        self.use_res = residual
        self.conv = ModuleList()
        self.switch_b1=False
        _in_channels = in_channels
        padding = self.get_padding(kernel_size, stride, dilation)

        # add the first n - 1 convolutions + activation
        for _ in range(repeat - 1):
            self.conv.extend(
                self.get_tcs_quant(
                    _in_channels, out_channels, kernel_size=kernel_size,
                    stride=stride, dilation=dilation,
                    padding=padding, separable=separable,
                    quant=quant,quant_act=quant_act
                )
            )

            self.conv.extend(self.get_activation_quant(activation, dropout,quant_act=quant_act))
            _in_channels = out_channels

        # add the last conv and batch norm
        self.conv.extend(
            self.get_tcs_quant(
                _in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride, dilation=dilation,
                padding=padding, separable=separable,
                    quant=quant,quant_act=quant_act
            )
        )

        self.conv.append(qnn.QuantIdentity(bit_width=quant))
        # add the residual connection
        if self.use_res:
            self.residual = Sequential(*self.get_tcs_quant(in_channels, out_channels,
                    quant=quant,quant_act=quant_act),qnn.QuantIdentity(bit_width=quant))

        # add the activation and dropout
        self.activation = Sequential(*self.get_activation_quant(activation, dropout,quant_act=quant_act))

    def set_switch(self, set_switch):
        self.switch_b1=set_switch
    def get_activation(self, activation, dropout):
        return activation, Dropout(p=dropout)

    def get_activation_quant(self, activation, dropout,quant_act=8):
        return qnn.QuantReLU(inplace=True,return_quant_tensor=False,
                bit_width=quant_act), Dropout(p=dropout)
    def get_padding(self, kernel_size, stride, dilation):
        if stride > 1 and dilation > 1:
            raise ValueError("Dilation and stride can not both be greater than 1")
        return (kernel_size // 2) * dilation

    def get_tcs(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False, separable=False):
        return [
            TCSConv1d(
                in_channels, out_channels, kernel_size,
                stride=stride, dilation=dilation, padding=padding,
                bias=bias, separable=separable
            ),
            BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        ]

    def get_tcs_quant(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False, separable=False,quant=8,quant_act=8):
        return [
            TCSConv1d_quant(
                in_channels, out_channels, kernel_size,
                stride=stride, dilation=dilation, padding=padding,
                bias=bias, separable=separable,
                    quant=quant,quant_act=quant_act
            ),
            BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        ]


    def forward(self, x):
        _x = x
        for layer in self.conv:
            _x = layer(_x)
        
        # print("B1 skip active")
        
        # print(self.switch_b1)
        if not self.switch_b1:
            _x = _x + self.residual(x)
        # else:
        #     print("B1 is OFF")
        return self.activation(_x)


class Block_repeat_b2(Module):
    """
    TCSConv, Batch Normalisation, Activation, Dropout
    """
    def __init__(self, in_channels, out_channels, activation, repeat=5, kernel_size=1, stride=1, dilation=1, dropout=0.0, residual=False, separable=False,quant=8,quant_act=8):

        super(Block_repeat_b2, self).__init__()

        self.use_res = residual
        self.conv = ModuleList()
        self.switch_b2=False
        _in_channels = in_channels
        padding = self.get_padding(kernel_size, stride, dilation)

        # add the first n - 1 convolutions + activation
        for _ in range(repeat - 1):
            self.conv.extend(
                self.get_tcs_quant(
                    _in_channels, out_channels, kernel_size=kernel_size,
                    stride=stride, dilation=dilation,
                    padding=padding, separable=separable,
                    quant=quant,quant_act=quant_act
                )
            )

            self.conv.extend(self.get_activation_quant(activation, dropout,quant_act=quant_act))
            _in_channels = out_channels

        # add the last conv and batch norm
        self.conv.extend(
            self.get_tcs_quant(
                _in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride, dilation=dilation,
                padding=padding, separable=separable,
                    quant=quant,quant_act=quant_act
            )
        )

        self.conv.append(qnn.QuantIdentity(bit_width=quant))
        # add the residual connection
        if self.use_res:
            self.residual = Sequential(*self.get_tcs_quant(in_channels, out_channels,
                    quant=quant,quant_act=quant_act),qnn.QuantIdentity(bit_width=quant))

        # add the activation and dropout
        self.activation = Sequential(*self.get_activation_quant(activation, dropout,quant_act=quant_act))

    def set_switch(self, set_switch):
        self.switch_b2=set_switch
    def get_activation(self, activation, dropout):
        return activation, Dropout(p=dropout)

    def get_activation_quant(self, activation, dropout,quant_act=8):
        return qnn.QuantReLU(inplace=True,return_quant_tensor=False,
                bit_width=quant_act), Dropout(p=dropout)
    def get_padding(self, kernel_size, stride, dilation):
        if stride > 1 and dilation > 1:
            raise ValueError("Dilation and stride can not both be greater than 1")
        return (kernel_size // 2) * dilation

    def get_tcs(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False, separable=False):
        return [
            TCSConv1d(
                in_channels, out_channels, kernel_size,
                stride=stride, dilation=dilation, padding=padding,
                bias=bias, separable=separable
            ),
            BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        ]

    def get_tcs_quant(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False, separable=False,quant=8,quant_act=8):
        return [
            TCSConv1d_quant(
                in_channels, out_channels, kernel_size,
                stride=stride, dilation=dilation, padding=padding,
                bias=bias, separable=separable,
                    quant=quant,quant_act=quant_act
            ),
            BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        ]


    def forward(self, x):
        _x = x
        for layer in self.conv:
            _x = layer(_x)
        
        if not self.switch_b2:
            _x = _x + self.residual(x)
        # else:
        #     print("B2 is OFF")
        return self.activation(_x)

class Block_repeat_b3(Module):
    """
    TCSConv, Batch Normalisation, Activation, Dropout
    """
    def __init__(self, in_channels, out_channels, activation, repeat=5, kernel_size=1, stride=1, dilation=1, dropout=0.0, residual=False, separable=False,quant=8,quant_act=8):

        super(Block_repeat_b3, self).__init__()

        self.use_res = residual
        self.conv = ModuleList()
        self.switch_b3=False
        _in_channels = in_channels
        padding = self.get_padding(kernel_size, stride, dilation)

        # add the first n - 1 convolutions + activation
        for _ in range(repeat - 1):
            self.conv.extend(
                self.get_tcs_quant(
                    _in_channels, out_channels, kernel_size=kernel_size,
                    stride=stride, dilation=dilation,
                    padding=padding, separable=separable,
                    quant=quant,quant_act=quant_act
                )
            )

            self.conv.extend(self.get_activation_quant(activation, dropout,quant_act=quant_act))
            _in_channels = out_channels

        # add the last conv and batch norm
        self.conv.extend(
            self.get_tcs_quant(
                _in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride, dilation=dilation,
                padding=padding, separable=separable,
                    quant=quant,quant_act=quant_act
            )
        )

        self.conv.append(qnn.QuantIdentity(bit_width=quant))
        # add the residual connection
        if self.use_res:
            self.residual = Sequential(*self.get_tcs_quant(in_channels, out_channels,
                    quant=quant,quant_act=quant_act),qnn.QuantIdentity(bit_width=quant))

        # add the activation and dropout
        self.activation = Sequential(*self.get_activation_quant(activation, dropout,quant_act=quant_act))

    def set_switch(self, set_switch):
        self.switch_b3=set_switch
    def get_activation(self, activation, dropout):
        return activation, Dropout(p=dropout)

    def get_activation_quant(self, activation, dropout,quant_act=8):
        return qnn.QuantReLU(inplace=True,return_quant_tensor=False,
                bit_width=quant_act), Dropout(p=dropout)
    def get_padding(self, kernel_size, stride, dilation):
        if stride > 1 and dilation > 1:
            raise ValueError("Dilation and stride can not both be greater than 1")
        return (kernel_size // 2) * dilation

    def get_tcs(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False, separable=False):
        return [
            TCSConv1d(
                in_channels, out_channels, kernel_size,
                stride=stride, dilation=dilation, padding=padding,
                bias=bias, separable=separable
            ),
            BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        ]

    def get_tcs_quant(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False, separable=False,quant=8,quant_act=8):
        return [
            TCSConv1d_quant(
                in_channels, out_channels, kernel_size,
                stride=stride, dilation=dilation, padding=padding,
                bias=bias, separable=separable,
                    quant=quant,quant_act=quant_act
            ),
            BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        ]


    def forward(self, x):
        _x = x
        for layer in self.conv:
            _x = layer(_x)
        
        if not self.switch_b3:
            _x = _x + self.residual(x)
        # else:
            # print("B3 is OFF")
        return self.activation(_x)
class Block_repeat_b4(Module):
    """
    TCSConv, Batch Normalisation, Activation, Dropout
    """
    def __init__(self, in_channels, out_channels, activation, repeat=5, kernel_size=1, stride=1, dilation=1, dropout=0.0, residual=False, separable=False,quant=8,quant_act=8):

        super(Block_repeat_b4, self).__init__()

        self.use_res = residual
        self.conv = ModuleList()
        self.switch_b4=False
        _in_channels = in_channels
        padding = self.get_padding(kernel_size, stride, dilation)

        # add the first n - 1 convolutions + activation
        for _ in range(repeat - 1):
            self.conv.extend(
                self.get_tcs_quant(
                    _in_channels, out_channels, kernel_size=kernel_size,
                    stride=stride, dilation=dilation,
                    padding=padding, separable=separable,
                    quant=quant,quant_act=quant_act
                )
            )

            self.conv.extend(self.get_activation_quant(activation, dropout,quant_act=quant_act))
            _in_channels = out_channels

        # add the last conv and batch norm
        self.conv.extend(
            self.get_tcs_quant(
                _in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride, dilation=dilation,
                padding=padding, separable=separable,
                    quant=quant,quant_act=quant_act
            )
        )

        self.conv.append(qnn.QuantIdentity(bit_width=quant))
        # add the residual connection
        if self.use_res:
            self.residual = Sequential(*self.get_tcs_quant(in_channels, out_channels,
                    quant=quant,quant_act=quant_act),qnn.QuantIdentity(bit_width=quant))

        # add the activation and dropout
        self.activation = Sequential(*self.get_activation_quant(activation, dropout,quant_act=quant_act))

    def set_switch(self, set_switch):
        self.switch_b4=set_switch
    def get_activation(self, activation, dropout):
        return activation, Dropout(p=dropout)

    def get_activation_quant(self, activation, dropout,quant_act=8):
        return qnn.QuantReLU(inplace=True,return_quant_tensor=False,
                bit_width=quant_act), Dropout(p=dropout)
    def get_padding(self, kernel_size, stride, dilation):
        if stride > 1 and dilation > 1:
            raise ValueError("Dilation and stride can not both be greater than 1")
        return (kernel_size // 2) * dilation

    def get_tcs(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False, separable=False):
        return [
            TCSConv1d(
                in_channels, out_channels, kernel_size,
                stride=stride, dilation=dilation, padding=padding,
                bias=bias, separable=separable
            ),
            BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        ]

    def get_tcs_quant(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False, separable=False,quant=8,quant_act=8):
        return [
            TCSConv1d_quant(
                in_channels, out_channels, kernel_size,
                stride=stride, dilation=dilation, padding=padding,
                bias=bias, separable=separable,
                    quant=quant,quant_act=quant_act
            ),
            BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        ]


    def forward(self, x):
        _x = x
        for layer in self.conv:
            _x = layer(_x)
        
        if not self.switch_b4:
            _x = _x + self.residual(x)
        # else:
        #     print("B4 is OFF")
        return self.activation(_x)
class Block_repeat_b5(Module):
    """
    TCSConv, Batch Normalisation, Activation, Dropout
    """
    def __init__(self, in_channels, out_channels, activation, repeat=5, kernel_size=1, stride=1, dilation=1, dropout=0.0, residual=False, separable=False,quant=8,quant_act=8):

        super(Block_repeat_b5, self).__init__()

        self.use_res = residual
        self.conv = ModuleList()
        self.switch_b5=False
        _in_channels = in_channels
        padding = self.get_padding(kernel_size, stride, dilation)

        # add the first n - 1 convolutions + activation
        for _ in range(repeat - 1):
            self.conv.extend(
                self.get_tcs_quant(
                    _in_channels, out_channels, kernel_size=kernel_size,
                    stride=stride, dilation=dilation,
                    padding=padding, separable=separable,
                    quant=quant,quant_act=quant_act
                )
            )

            self.conv.extend(self.get_activation_quant(activation, dropout,quant_act=quant_act))
            _in_channels = out_channels

        # add the last conv and batch norm
        self.conv.extend(
            self.get_tcs_quant(
                _in_channels, out_channels,
                kernel_size=kernel_size,
                stride=stride, dilation=dilation,
                padding=padding, separable=separable,
                    quant=quant,quant_act=quant_act
            )
        )

        self.conv.append(qnn.QuantIdentity(bit_width=quant))
        # add the residual connection
        if self.use_res:
            self.residual = Sequential(*self.get_tcs_quant(in_channels, out_channels,
                    quant=quant,quant_act=quant_act),qnn.QuantIdentity(bit_width=quant))

        # add the activation and dropout
        self.activation = Sequential(*self.get_activation_quant(activation, dropout,quant_act=quant_act))

    def set_switch(self, set_switch):
        self.switch_b5=set_switch
    def get_activation(self, activation, dropout):
        return activation, Dropout(p=dropout)

    def get_activation_quant(self, activation, dropout,quant_act=8):
        return qnn.QuantReLU(inplace=True,return_quant_tensor=False,
                bit_width=quant_act), Dropout(p=dropout)
    def get_padding(self, kernel_size, stride, dilation):
        if stride > 1 and dilation > 1:
            raise ValueError("Dilation and stride can not both be greater than 1")
        return (kernel_size // 2) * dilation

    def get_tcs(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False, separable=False):
        return [
            TCSConv1d(
                in_channels, out_channels, kernel_size,
                stride=stride, dilation=dilation, padding=padding,
                bias=bias, separable=separable
            ),
            BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        ]

    def get_tcs_quant(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, padding=0, bias=False, separable=False,quant=8,quant_act=8):
        return [
            TCSConv1d_quant(
                in_channels, out_channels, kernel_size,
                stride=stride, dilation=dilation, padding=padding,
                bias=bias, separable=separable,
                    quant=quant,quant_act=quant_act
            ),
            BatchNorm1d(out_channels, eps=1e-3, momentum=0.1)
        ]


    def forward(self, x):
        _x = x
        for layer in self.conv:
            _x = layer(_x)
        
        if not self.switch_b5:
            _x = _x + self.residual(x)
        # else:
        #     print("B5 is OFF")
        return self.activation(_x)