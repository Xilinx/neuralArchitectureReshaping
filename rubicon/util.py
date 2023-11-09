# util.py -*- Python -*-
#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2023 Advanced Micro Devices, Inc.

"""
ONT Bonito Basecall
"""

import os
import re
import sys
import random
from glob import glob
from itertools import groupby
from operator import itemgetter
from importlib import import_module
from collections import deque, defaultdict, OrderedDict
from torch.utils.data import DataLoader
from torch.nn.utils.prune import _validate_structured_pruning, BasePruningMethod
import toml
import torch
import koi.lstm
import parasail
from torch.nn.utils import prune
import numpy as np
from torch.cuda import get_device_capability
from torch import nn
try:
    from claragenomics.bindings import cuda
    from claragenomics.bindings.cudapoa import CudaPoaBatch
except ImportError:
    pass
import logging
_logger = logging.getLogger(__name__)

__dir__ = os.path.dirname(os.path.realpath(__file__))
__data__ = os.path.join(__dir__, "data")
__models__ = os.path.join(__dir__, "models")
__configs__ = os.path.join(__dir__, "models/configs")
__dataorg__ = os.path.join(__dir__, "data/organism")
split_cigar = re.compile(r"(?P<len>\d+)(?P<op>\D+)")
default_data = os.path.join(__data__, "dna_r9.4.1")
default_config = os.path.join(__configs__, "dna_r9.4.1@v3.1.toml")



class GaganPrune(BasePruningMethod):
    """Prune (currently unpruned) units in a tensor at random positions, but same for all channels.

    Args:
        name (str): parameter name within ``module`` on which pruning
            will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.

    """
    
    PRUNUNG_TYPE = 'unstructured'

    def __init__(self, amount):
        # Check range of validity of pruning amount
        # _validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):
        # Check that the amount of units to prune is not > than the number of
        # parameters in t channels

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        # we assume the shape of t is: OFM, IFM, kernel size
        new_mask = torch.ones_like(t) # OFM x IFM x Kernel
        
        new_mask[t==0] = 0 # set mask for zero weights to zero
          
        mask[new_mask == 0] = 0 # apply the new mask

        return mask

    @classmethod
    def apply(cls, module, name, amount):
        r"""Adds the forward pre-hook that enables pruning on the fly and
        the reparametrization of a tensor in terms of the original tensor
        and the pruning mask.
        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
        """
        return super(GaganPrune, cls).apply(module, name, amount=amount)

def gagan_prune(module, name, amount=0):
    r"""Prunes tensor corresponding to parameter called ``name`` in ``module``
    by removing the specified ``amount`` of (currently unpruned) units
    selected at random but similar between channels.
    Modifies module in place (and also return the modified module) by:
    1) adding a named buffer called ``name+'_mask'`` corresponding to the
    binary mask applied to the parameter `name` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
    original (unpruned) parameter is stored in a new parameter named
    ``name+'_orig'``.
    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
                will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        ogs (int): OFM group size
    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module
    Examples:
        >>> m = vail_random_unstructured(nn.Linear(2, 3), 'weight', amount=1, group=8)
        >>> torch.sum(m.weight_mask == 0)
        tensor(1)
    """
    GaganPrune.apply(module, name, amount)
    return module

def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):

    num_zeros = 0
    num_elements = 0
    sum=0
    if use_mask == True:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight == True:
                # print("param_name={}".format(param_name))
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
                # print("num_zeros={}, num_elements={}".format(num_zeros,num_elements))
            if "bias" in param_name and bias == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity
def measure_global_sparsity(model,
                            weight=True,
                            bias=False,
                            conv1d_use_mask=False):

    num_zeros = 0
    num_elements = 0
    sum=0
    table = PrettyTable(["Layer_Sum", "Sparsity"])
    for module_name, module in model.encoder.named_modules():
        
        if isinstance(module, qnn.QuantConv1d):
            sum+=((module.quant_weight().value != 0.) * module.quant_weight().bit_width).sum() / 8
            layer_sum = ((module.quant_weight().value != 0.) * module.quant_weight().bit_width).sum() / 8
            layer_sparsity = 1. - (module.quant_weight().value != 0.).int().sum() / module.quant_weight().value.numel()
            # print(f"{layer_sum} {layer_sparsity}")
            table.add_row([f"{layer_sum}", f"{layer_sparsity}"])
            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=conv1d_use_mask)
            
            num_zeros += module_num_zeros
            num_elements += module_num_elements

    for module_name, module in model.decoder.named_modules():

        if isinstance(module, qnn.QuantConv1d):
            sum+=((module.quant_weight().value != 0.) * module.quant_weight().bit_width).sum() / 8
            layer_sum = ((module.quant_weight().value != 0.) * module.quant_weight().bit_width).sum() / 8
            layer_sparsity = 1. - (module.quant_weight().value != 0.).int().sum() / module.quant_weight().value.numel()
            # print(f"{layer_sum};{layer_sparsity}")
            table.add_row([f"{layer_sum}", f"{layer_sparsity}"])
            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=conv1d_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements
    sparsity = num_zeros / num_elements
    print(table)
    print("model size=",f"{sum}")
    return num_zeros, num_elements, sparsity

def prune_model_structured(model, layer_type, proportion,l1_scoring,prune_dim):
    for name, module in model.encoder.named_modules():
        # print("Name={}, Module={}, Layer type={}".format(name,module, layer_type))
        if isinstance(module, layer_type):
            # print("pruning")
            if(l1_scoring):
                # print("****BEFORE****")
                # print(list(module.named_parameters()))
                
                prune.ln_structured(module, 'weight', proportion, n=1, dim=prune_dim,importance_scores=module.quant_weight().value) #dim=0 output channels
                print("****AFTER****")
                # print(list(module.named_buffers()))
                # print(list(module.named_parameters()))

            else:
                prune.random_structured(module, 'weight', proportion, dim=prune_dim)
                # prune.ln_structured(module, 'weight', proportion, n=1, dim=prune_dim,importance_scores=module.quant_weight().value)          
            # prune.remove(module, 'weight')

    for name, module in model.decoder.named_modules():
        if isinstance(module, layer_type):
            if(l1_scoring):
              
                prune.ln_structured(module, 'weight', proportion, n=1, dim=prune_dim,importance_scores=module.quant_weight().value) #dim=0 output channels
                
                
            else:
                
                prune.random_structured(module, 'weight', proportion, dim=prune_dim)
            # prune.remove(module, 'weight')   
    return model
def prune_model_gagan(model, layer_type):
    for name, module in model.encoder.named_modules():
        # print("Name={}, Module={}, Layer type={}".format(name,module, layer_type))
        if isinstance(module, layer_type):
            gagan_prune(module, 'weight')
                # prune.ln_structured(module, 'weight', proportion, n=1, dim=prune_dim,importance_scores=module.quant_weight().value)          
            # prune.remove(module, 'weight')

    for name, module in model.decoder.named_modules():
        if isinstance(module, layer_type):
            gagan_prune(module, 'weight')
    return model
def remove_prune_mask(model, layer_type):
    for name, module in model.encoder.named_modules():
        # print("Name={},  Layer type={}".format(name, layer_type))
        if isinstance(module, layer_type):
           prune.remove(module, 'weight')

    for name, module in model.decoder.named_modules():
        if isinstance(module, layer_type):
            prune.remove(module, 'weight')   
    return model
def load_model_prune_for_kd(dirname, device, weights=None, half=None, quant=False, chunksize=None, batchsize=None, overlap=None, quantize=False, use_koi=False,load_model_type=None, no_prune=False):
    """
    Load a model from disk
    """
    print("[util] loading pre-trained from", dirname)
    if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models__, dirname)):
        dirname = os.path.join(__models__, dirname)

    if not weights: # take the latest checkpoint
        weight_files = glob(os.path.join(dirname, "weights_*.tar"))
        if not weight_files:
            raise FileNotFoundError("no model weights found in '%s'" % dirname)
        weights = max([int(re.sub(".*_([0-9]+).tar", "\\1", w)) for w in weight_files])
        print("[util] Using previous weights:",weights)
    
    device = torch.device(device)
    config = toml.load(os.path.join(dirname, 'config.toml'))
    config = toml.load("models/configs/config.toml")
    weights = os.path.join(dirname, 'weights_%s.tar' % weights)
    
    if quant:
            module_check=qnn.QuantConv1d
    else: 
            module_check=nn.Conv1d

    if (load_model_type=="rubiconqabas"):
            _logger.info("Training Rubicon QABAS model")
            model = load_symbol(config, 'RubiconQabas')(config)  
    elif (load_model_type=="bonito"):
            _logger.info("ONT Bonito model")
            model = load_symbol(config, 'Model')(config) 
    elif (load_model_type=="bonitostaticquant"):
            _logger.info("ONT Bonito model for static quantization")
            model = load_symbol(config, 'BonitoStaticQuant')(config)  
   

    if use_koi:
        model.encoder = koi.lstm.update_graph(
            model.encoder, batchsize=batchsize, chunksize=chunksize // model.stride, quantize=quantize
        )
    # print("***************before PRUNING****************")
    # print(dict(model.named_parameters()).keys())
    state_dict = torch.load(weights, map_location=device)
    if(no_prune):
        import copy
        model.load_state_dict(state_dict)
        final_model=copy.deepcopy(model)
    else:
        prune_model=prune_model_structured(model,  module_check, 0,True,0)
        prune_model.load_state_dict(state_dict)
        prune_model=remove_prune_mask(prune_model, module_check)
        final_model=prune_model_gagan(prune_model, module_check)

        num_zeros, num_elements, sparsity = measure_global_sparsity(
                    final_model,module_check,
                    weight=True,
                    bias=False,
                    conv1d_use_mask=True)
        print("Global Sparsity:")
        print("{:.2f}".format(sparsity))

    # if half is None:
    #     half = half_supported()

    # if half: model = model.half()
    final_model.eval()
    final_model.to(device)
    print("[SUCCESFULLY LOADED THE PRUNED MODEL FOR KD]")
    return final_model

def load_model_prune_for_basecall(dirname, device, weights=None, half=None, chunksize=None, batchsize=None, remove_mask=False,overlap=None, quantize=False, use_koi=False,load_model_type=None):
    """
    Load a model from disk
    """
    # print("[util] loading pre-trained from", dirname)
    if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models__, dirname)):
        dirname = os.path.join(__models__, dirname)

    if not weights: # take the latest checkpoint
        weight_files = glob(os.path.join(dirname, "weights_*.tar"))
        if not weight_files:
            raise FileNotFoundError("no model weights found in '%s'" % dirname)
        weights = max([int(re.sub(".*_([0-9]+).tar", "\\1", w)) for w in weight_files])
        # print("[util] Using previous weights:",weights)
    # print("Loading weights::{}".format(weights))
    device = torch.device(device)
    config = toml.load(os.path.join(dirname, 'config.toml'))
    # config = toml.load("models/configs/config.toml")

    # if remove_mask:
    #     weights = os.path.join(dirname, 'WITHOUT_MASK_WEIGHT_%s.tar' % weights)
    # else:
    weights = os.path.join(dirname, 'weights_%s.tar' % weights)

    if (load_model_type=="rubiconqabas"):
            _logger.info("Rubicon QABAS model")
            model = load_symbol(config, 'RubiconQabas')(config)  
    elif (load_model_type=="bonito"):
            _logger.info("ONT Bonito model")
            model = load_symbol(config, 'Bonito')(config) 
    elif (load_model_type=="bonitostaticquant"):
            _logger.info("ONT Bonito model for static quantization")
            model = load_symbol(config, 'BonitoStaticQuant')(config)  
    elif (load_model_type=="rubiconnoskipfp"):
            _logger.info("rubiconnoskipfp")
            model = load_symbol(config, 'RubiconNoSkipFP')(config)  
    state_dict = torch.load(weights, map_location=device)
    if (load_model_type in ["bonito"]) or (remove_mask):
        import copy
        model.load_state_dict(state_dict)
        prune_model=copy.deepcopy(model)

    else:
        prune_model=prune_model_structured(model,  qnn.QuantConv1d, 0,True,0)
        prune_model.load_state_dict(state_dict)
        prune_model=remove_prune_mask(prune_model, qnn.QuantConv1d)
    prune_model.eval()
    prune_model.to(device)
    _logger.info("Succesfully loaded the model")
    return prune_model
def load_model_kd_basecall(dirname, device, weights=None, half=None, quant=False, chunksize=None, batchsize=None, overlap=None, quantize=False, use_koi=False,load_model_type=None, no_prune=False):
    """
    Load a model from disk
    """
    print("[util] loading pre-trained from", dirname)
    if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models__, dirname)):
        dirname = os.path.join(__models__, dirname)

    if not weights: # take the latest checkpoint
        weight_files = glob(os.path.join(dirname, "weights_*.tar"))
        if not weight_files:
            raise FileNotFoundError("no model weights found in '%s'" % dirname)
        weights = max([int(re.sub(".*_([0-9]+).tar", "\\1", w)) for w in weight_files])
        print("[util] Using previous weights:",weights)
    
    device = torch.device(device)
    config = toml.load(os.path.join(dirname, 'config.toml'))

    weights = os.path.join(dirname, 'weights_%s.tar' % weights)
    
    if quant:
            module_check=qnn.QuantConv1d
    else: 
            module_check=nn.Conv1d

    if (load_model_type=="rubiconqabas"):
            _logger.info("Training Rubicon QABAS model")
            model = load_symbol(config, 'RubiconQabas')(config)  
    elif (load_model_type=="bonito"):
            _logger.info("ONT Bonito model")
            model = load_symbol(config, 'Model')(config) 
    elif (load_model_type=="bonitostaticquant"):
            _logger.info("ONT Bonito model for static quantization")
            model = load_symbol(config, 'BonitoStaticQuant')(config)  
   
    elif (load_model_type in ["rubiconskiptrim","rubicallmp"]):
            _logger.info("SkipTrim model")
            model = load_symbol(config, 'RubiconSkipTrim')(config)  

    if use_koi:
        model.encoder = koi.lstm.update_graph(
            model.encoder, batchsize=batchsize, chunksize=chunksize // model.stride, quantize=quantize
        )

    state_dict = torch.load(weights, map_location=device)
    if(no_prune):
        import copy
        model.load_state_dict(state_dict)
        final_model=copy.deepcopy(model)
    else:
        prune_model=prune_model_structured(model,  module_check, 0,True,0)        
        prune_model.load_state_dict(state_dict)
        prune_model=remove_prune_mask(prune_model, module_check)
       
    prune_model.eval()
    prune_model.to(device)

    return prune_model
def load_model_prune(dirname, device, weights=None, half=None, chunksize=None, batchsize=None, overlap=None, quantize=False, use_koi=False,load_model_type=None, load_model_channel=None):
    """
    Load a model from disk
    """
    print("[util] loading pre-trained from", dirname)
    if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models__, dirname)):
        dirname = os.path.join(__models__, dirname)

    if not weights: # take the latest checkpoint
        weight_files = glob(os.path.join(dirname, "weights_*.tar"))
        if not weight_files:
            raise FileNotFoundError("no model weights found in '%s'" % dirname)
        weights = max([int(re.sub(".*_([0-9]+).tar", "\\1", w)) for w in weight_files])
        print("[util] Using previous weights:",weights)
    # weights=1
    device = torch.device(device)
    config = toml.load(os.path.join(dirname, 'config.toml'))
    config = toml.load("/scratch/adaptive/users/gagandee/bonito_nas/bonito/models/configs/dna_r9.4.1@v2.1.toml")
    weights = os.path.join(dirname, 'weights_%s.tar' % weights)

    basecall_params = config.get("basecaller", {})
    # use `value or dict.get(key)` rather than `dict.get(key, value)` to make
    # flags override values in config
    chunksize = basecall_params["chunksize"] = chunksize or basecall_params.get("chunksize", 4000)
    overlap = basecall_params["overlap"] = overlap or basecall_params.get("overlap", 500)
    batchsize = basecall_params["batchsize"] = batchsize or basecall_params.get("batchsize", 64)
    quantize = basecall_params["quantize"] = basecall_params.get("quantize") if quantize is None else quantize
    config["basecaller"] = basecall_params

    # Model = load_symbol(config, "Model")
    # model = Model(config)

    if (load_model_type=="rubiconqabas"):
            _logger.info("Rubicon QABAS model")
            model = load_symbol(config, 'RubiconQabas')(config)  
    elif (load_model_type=="bonito"):
            _logger.info("ONT Bonito model")
            model = load_symbol(config, 'Bonito')(config) 
    elif (load_model_type=="bonitostaticquant"):
            _logger.info("ONT Bonito model for static quantization")
            model = load_symbol(config, 'BonitoStaticQuant')(config)  
    elif (load_model_type=="rubiconnoskipfp"):
            _logger.info("rubiconnoskipfp")
            model = load_symbol(config, 'RubiconNoSkipFP')(config)  

    if use_koi:
        model.encoder = koi.lstm.update_graph(
            model.encoder, batchsize=batchsize, chunksize=chunksize // model.stride, quantize=quantize
        )

    state_dict = torch.load(weights, map_location=device)
    print("*********OLD DICT (len={})**********".format(len(state_dict.keys())))
    print("*********NEW MODEL DICT (len={})**********".format(sum(p.numel() for p in model.parameters())))

    model.load_state_dict(state_dict)

    model.eval()
    model.to(device)
    return model
