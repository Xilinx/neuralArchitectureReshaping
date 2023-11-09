#!/usr/bin/env python3
# prune.py -*- Python -*-
#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2023 Advanced Micro Devices, Inc.

import os
from statistics import mode
import sys
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
from importlib import import_module

from rubicon.data import load_numpy_shuf,load_numpy_full 
from bonito.data import load_script
from bonito.util import load_symbol, init,load_model
from rubicon.util import load_model_prune,load_model_prune_for_basecall


from rubicon.util import __models__, default_config, default_data

from rubicon.training import load_state, Trainer
from rubicon.prunekdtraining import load_state, PruneKDTrainer
import subprocess
from prettytable import PrettyTable
import toml
import torch
from torch import nn
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils import prune
import brevitas.nn as qnn

import shutil
import logging
_logger = logging.getLogger(__name__)

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

def measure_global_sparsity(model, module_check,
                            weight=True,
                            bias=False,
                            conv1d_use_mask=False):

    num_zeros = 0
    num_elements = 0
    sum=0
    table = PrettyTable(["Layer_Sum", "Sparsity"])
    _logger.info("Checking for pruning module:%s"%module_check)
    for module_name, module in model.encoder.named_modules():
        # print("checking for pruning module:::",module)
        if(module_check==qnn.QuantConv1d):
            if isinstance(module, module_check):
                    sum+=((module.quant_weight().value != 0.) * module.quant_weight().bit_width).sum() / 8
                    layer_sum = ((module.quant_weight().value != 0.) * module.quant_weight().bit_width).sum() / 8
                    layer_sparsity = 1. - (module.quant_weight().value != 0.).int().sum() / module.quant_weight().value.numel()
                    table.add_row([f"{layer_sum}", f"{layer_sparsity}"])
                    module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                        module, weight=weight, bias=bias, use_mask=conv1d_use_mask)
                    num_zeros += module_num_zeros
                    num_elements += module_num_elements

        else: 
            if isinstance(module, module_check):

                module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                    module, weight=weight, bias=bias, use_mask=conv1d_use_mask)
                num_zeros += module_num_zeros
                num_elements += module_num_elements

    for module_name, module in model.decoder.named_modules():
        if(module_check==qnn.QuantConv1d):
            if isinstance(module, module_check):
                sum+=((module.quant_weight().value != 0.) * module.quant_weight().bit_width).sum() / 8
                layer_sum = ((module.quant_weight().value != 0.) * module.quant_weight().bit_width).sum() / 8
                layer_sparsity = 1. - (module.quant_weight().value != 0.).int().sum() / module.quant_weight().value.numel()
                table.add_row([f"{layer_sum}", f"{layer_sparsity}"])
                module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                    module, weight=weight, bias=bias, use_mask=conv1d_use_mask)
                num_zeros += module_num_zeros
                num_elements += module_num_elements
        else:
            if isinstance(module, module_check):
                module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                    module, weight=weight, bias=bias, use_mask=conv1d_use_mask)
                num_zeros += module_num_zeros
                num_elements += module_num_elements

    sparsity = num_zeros / num_elements
    print(table)
    print("model size=",f"{sum}")
    return num_zeros, num_elements, sparsity

def prune_model_unstructured(model, layer_type, proportion,l1_scoring):
    """
    Prunes the model.
    """
    for name, module in model.encoder.named_modules():
       
        if isinstance(module, layer_type): 
            if(l1_scoring):
                prune.l1_unstructured(module, 'weight', amount=proportion,importance_scores=module.quant_weight().value)          
            else:
                prune.random_unstructured(module,'weight',amount=proportion)

    for name, module in model.decoder.named_modules():
        if isinstance(module, layer_type):
         
            if(l1_scoring):
                prune.l1_unstructured(module, 'weight', amount=proportion,importance_scores=module.quant_weight().value)          
            else:
                prune.random_unstructured(module,'weight',amount=proportion)

    return model
def prune_model_structured(model, layer_type, proportion,l1_scoring,prune_dim):
    for name, module in model.encoder.named_modules():

        if isinstance(module, layer_type):
 
            if(l1_scoring):
                prune.ln_structured(module, 'weight', proportion, n=1, dim=prune_dim,importance_scores=module.quant_weight().value) #dim=0 output channels
            else:
                prune.random_structured(module, 'weight', proportion, dim=prune_dim)
      

    for name, module in model.decoder.named_modules():
        if isinstance(module, layer_type):
            if(l1_scoring):
                
                prune.ln_structured(module, 'weight', proportion, n=1, dim=prune_dim,importance_scores=module.quant_weight().value)
            else:
                prune.random_structured(module, 'weight', proportion, dim=prune_dim)

    return model
def remove_prune_mask(model, layer_type):
    for name, module in model.encoder.named_modules():
        if isinstance(module, layer_type):
           prune.remove(module, 'weight')

    for name, module in model.decoder.named_modules():
        if isinstance(module, layer_type):
            prune.remove(module, 'weight')   
    return model
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    _logger.info(table)
    print(table)
    _logger.info(f"Total Trainable Params: {total_params}")
    return total_params

def main(args):
    temp=args.temp
    alpha=args.alpha
    structural=args.struct
    l1_scoring=args.l1
    prune_dim=args.prune_dim
    prune_proportion=args.prune
    workdir = os.path.expanduser(args.training_directory)
    onnx_name=""
    onnx_name+=str(prune_proportion)
    _logger.info("Save path: {}".format(workdir))

    if args.force and os.path.exists(workdir):
        shutil.rmtree(workdir)
    if os.path.exists(workdir) and not args.force:
        print("[error] %s exists, remove it to continue or use -f to force delete." % workdir)
        exit(1)

    if not prune_proportion:
        _logger.warning("Specify sparsity using args --prune")
        exit(1)

    os.makedirs(workdir, exist_ok=True)
    init(args.seed, args.device)
    device = torch.device(args.device)
    
    if args.pretrained:
        dirname = args.pretrained
        if not os.path.isdir(dirname) and os.path.isdir(os.path.join(__models__, dirname)):
            dirname = os.path.join(__models__, dirname)
        config_file = os.path.join(dirname, 'config.toml')
    else:
        config_file = args.config

    config = toml.load(config_file)

    argsdict = dict(training=vars(args))

    
    toml.dump({**config, **argsdict}, open(os.path.join(workdir, 'config.toml'), 'w'))

    if (args.teacher):    
        teacher_path=os.path.expanduser(args.teacher_directory)
        _logger.info("Loading Teacher Model:{}".format(teacher_path))
        if not os.path.exists(teacher_path):
            _logger.warning("Teacher model %s does not exists" % teacher_path)
            exit(1)
        model_teacher = load_model(teacher_path, device, half=False)
    
        _logger.info("Total parameters in model teacher:{}".format(sum(p.numel() for p in model_teacher.parameters())) )
        original_stdout = sys.stdout # Save a reference to the original standard output
        with open(workdir+'/model_teacher.txt', 'w') as f:
            sys.stdout = f # Change the standard output to the file we created.
            count_parameters(model_teacher)
        sys.stdout = original_stdout # Reset the standard output to its original value



    _logger.info("Loading model for pruning")
    if args.quant:
        module_check=qnn.QuantConv1d
    else: 
        module_check=nn.Conv1d
    

    if args.pretrained:
        _logger.info("[using pretrained model {}]".format(args.pretrained))
        model_student = load_model_prune(args.pretrained, device, half=False,load_model_type=args.type)
    else:
        if (args.type=="rubiconqabas"):
            _logger.info("Training Rubicon QABAS model")
            model_student = load_symbol(config, 'RubiconQabas')(config)  
        elif (args.type=="bonito"):
                _logger.info("ONT Bonito model")
                model_student = load_symbol(config, 'Model')(config) 
        elif (args.type=="bonitostaticquant"):
                _logger.info("ONT Bonito model for static quantization")
                model_student = load_symbol(config, 'BonitoStaticQuant')(config)  
        else:
            _logger.warning("Please define a model or choose a model using --type")
            exit(1)
        

    if config.get("lr_scheduler"):
        sched_config = config["lr_scheduler"]
        lr_scheduler_fn = getattr(
            import_module(sched_config["package"]), sched_config["symbol"]
        )(**sched_config)
    else:
        lr_scheduler_fn = None

    original_stdout = sys.stdout # Save a reference to the original standard output
    with open(workdir+'/model_student.txt', 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        count_parameters(model_student)
        _logger.info("Total parameters in BASE model=%s"%sum(p.numel() for p in model_student.parameters()))
        torch.save(model_student.state_dict(), workdir+"/model_student.h5")
       
        if(structural):
            _logger.info("Structural pruning") 
            prune_model=prune_model_structured(model_student,  module_check, prune_proportion,l1_scoring,prune_dim)
    
        else:
            _logger.info("Unstructural pruning") 
            prune_model=prune_model_unstructured(model_student, module_check, prune_proportion,l1_scoring)
                                    
        torch.save(prune_model.state_dict(), workdir+"/model_prune.h5")
      
      
        _logger.info("***Measuring sparsity***") 

        num_zeros, num_elements, sparsity = measure_global_sparsity(
                prune_model, module_check,
                weight=True,
                bias=False,
                conv1d_use_mask=True)
        _logger.info("Global Sparsity:")
        _logger.info("{:.2f}".format(sparsity))

        if args.full:
            _logger.info("Full dataset training")
            train_loader_kwargs, valid_loader_kwargs = load_numpy_full(None,
                    args.directory
            )
        elif args.chunks:
            _logger.info("Not full dataset training with shuffling")
            train_loader_kwargs, valid_loader_kwargs = load_numpy(
                args.chunks,args.valid_chunks, args.directory
            )
        else:
            _logger.warning("Please define the training data correctly")
            exit(1)

        loader_kwargs = {
            "batch_size": args.batch, "num_workers": 2, "pin_memory": True
        }
        train_loader = DataLoader(**loader_kwargs, **train_loader_kwargs)
        valid_loader = DataLoader(**loader_kwargs, **valid_loader_kwargs)
        
        if(args.teacher):
            _logger.info("Pruning with Knowledge Distillation")
            trainer = PruneKDTrainer(
                model_teacher,
                prune_model, device, train_loader, valid_loader,
                use_amp=False,
                lr_scheduler_fn=lr_scheduler_fn,
                restore_optim=args.restore_optim,
                save_optim_every=args.save_optim_every,
                grad_accum_split=args.grad_accum_split,
                temp=temp,
                alpha=alpha
            )
        else:
            _logger.info("Training without Knowledge Distillation")
            trainer = Trainer(
                prune_model, device, train_loader, valid_loader,
                use_amp=False,
                lr_scheduler_fn=lr_scheduler_fn,
                restore_optim=args.restore_optim,
                save_optim_every=args.save_optim_every,
                grad_accum_split=args.grad_accum_split
            )

    
        if(args.remove_mask):
            _logger.warning("Removing mask while storing weights")
            trainer.fit(workdir, args.epochs, args.lr,remove_mask=True,quant=args.quant,load_model_type=args.type)

        else:
            trainer.fit(workdir, args.epochs, args.lr,load_model_type=args.type)
            _logger.info("Measure sparsity after training")
            num_zeros, num_elements, sparsity = measure_global_sparsity(
                prune_model,module_check,
                weight=True,
                bias=False,
                conv1d_use_mask=True)


        _logger.info("Global Sparsity:")
        _logger.info("{:.2f}".format(sparsity))
        _logger.info("Removing mask after training")
        prune_model=remove_prune_mask(model_student, qnn.QuantConv1d)
        sys.stdout = original_stdout # Reset the standard output to its original value

  
def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("training_directory")
    group = parser.add_mutually_exclusive_group()
    parser.add_argument("--teacher_directory", default="models/bonito")
    group.add_argument('--config', default="models/configs/config.toml")
    group.add_argument('--pretrained', default="")
    parser.add_argument("--directory", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--full", action="store_true", default=False)
    parser.add_argument("--quant", action="store_true", default=False)
    parser.add_argument("--remove_mask", action="store_true", default=False)
    parser.add_argument("--lr", default=2e-3, type=float)
    parser.add_argument("--temp", default=1, type=float)
    parser.add_argument("--alpha", default=0.1, type=float)
    parser.add_argument("--prune", type=float)
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--chunks", default=0, type=int)
    parser.add_argument('--name_append', default="")
    parser.add_argument("--valid-chunks", default=0, type=int)
    parser.add_argument("--no-amp", action="store_true", default=True)
    parser.add_argument("-l1",  action="store_true", default=False)
    parser.add_argument("-struct",  action="store_true", default=False)
    parser.add_argument("-f", "--force", action="store_true", default=False)
    parser.add_argument("--restore-optim", action="store_true", default=False)
    parser.add_argument("--save-optim-every", default=10, type=int)
    parser.add_argument("--grad-accum-split", default=1, type=int)
    parser.add_argument("--prune-dim", default=0, type=int)
    parser.add_argument("--type", default=None, type=str, choices=["staticquant", 'rubiconqabas','bonito'])
    parser.add_argument("--teacher", action="store_true", default=False)
    return parser
