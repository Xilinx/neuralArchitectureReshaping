#!/usr/bin/env python3
# train.py -*- Python -*-
#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2023 Advanced Micro Devices, Inc.

import os
import sys
from argparse import ArgumentParser
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
from importlib import import_module
import shutil
from rubicon.data import load_numpy_shuf,load_numpy_full 
from rubicon.tools.ont.bonito.data import load_script
from rubicon.tools.ont.bonito.util import load_symbol, init,load_model
from rubicon.util import load_model_prune,load_model_prune_for_basecall

from rubicon.training import load_state, Trainer
from rubicon.prunekdtraining import  PruneKDTrainer
import toml
import torch
import numpy as np
from torch.utils.data import DataLoader
import brevitas.nn as qnn
import torch.autograd.profiler as profiler
import subprocess
from prettytable import PrettyTable
import logging
# logging.basicConfig(filename='rubicon.log',  filemode='w', format='%(asctime)s %(name)s - %(levelname)s - %(message)s',level=logging.INFO, datefmt='%m/%d/%Y %I:%M:%S %p')

log_filename = "logs/output.log"
os.makedirs(os.path.dirname(log_filename), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt='%m/%d/%Y %I:%M:%S %p',
    handlers=[
        logging.FileHandler(log_filename,mode="w"),
        logging.StreamHandler()
    ]
)

_logger = logging.getLogger(__name__)
def measure_size(model):
    sum=0
    table = PrettyTable(["Layer_Sum"])
    for module_name, module in model.encoder.named_modules():
        
        if isinstance(module, qnn.QuantConv1d):
            sum+=((module.quant_weight().value != 0.) * module.quant_weight().bit_width).sum() / 8
            layer_sum = ((module.quant_weight().value != 0.) * module.quant_weight().bit_width).sum() / 8
           
            table.add_row([f"{layer_sum}"])

    for module_name, module in model.decoder.named_modules():

        if isinstance(module, qnn.QuantConv1d):
            sum+=((module.quant_weight().value != 0.) * module.quant_weight().bit_width).sum() / 8
            layer_sum = ((module.quant_weight().value != 0.) * module.quant_weight().bit_width).sum() / 8
            table.add_row([f"{layer_sum}"])
    print(table)
    print("model size=",f"{sum}")
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



def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        # print(name.quant_weight().bit_width())
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
    
def main(args):
    temp=args.temp
    alpha=args.alpha
    workdir = os.path.expanduser(args.training_directory)
    if(args.name_append):
         workdir+=args.name_append
    
    _logger.info("Save path: {}".format(workdir))
    
    if os.path.exists(workdir) and not args.force:
        print("[error] %s exists, remove it to continue or use -f to force delete." % workdir)
        exit(1)

    init(args.seed, args.device)
    device = torch.device(args.device)    
    
    os.makedirs(workdir, exist_ok=True)
    
    
    config_file = args.config

    config = toml.load(config_file)

    argsdict = dict(training=vars(args))

    os.makedirs(workdir, exist_ok=True)
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



    if args.pretrained:
        _logger.info("Using Pretrained Student:{}".format(args.pretrained))
        if(args.remove_mask):
            remove_mask=True
        else:
            remove_mask=False
        model_student=load_model_prune_for_basecall(
            args.pretrained,
             args.device, 
             half=False,
             load_model_type=args.type,
              weights=int(args.weights),
              remove_mask=remove_mask)

    else:
        if (args.type=="rubiconqabas-mp"):
            _logger.info("Training Rubicon QABAS model")
            model_student = load_symbol(config, 'RubiconQabas')(config)  
        elif (args.type=="bonito"):
                _logger.info("ONT Bonito model")
                model_student = load_symbol(config, 'Model')(config) 
        elif (args.type=="bonitostaticquant"):
                _logger.info("ONT Bonito model for static quantization")
                model_student = load_symbol(config, 'BonitoStaticQuant')(config)  
        
        elif (args.type=="rubiconnoskipfp"):
                _logger.info("RUBICON No SKip")
                model_student = load_symbol(config, 'RubiconNoSkipFP')(config)  
        elif (args.type=="rubiconfp"):
                _logger.info("RUBICON FP")
                model_student = load_symbol(config, 'RubiconFP')(config)  
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
    _logger.info("Loading Data")
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
        "batch_size": args.batch, "num_workers": 1, "pin_memory": True
    }
    train_loader = DataLoader(**loader_kwargs, **train_loader_kwargs)
    valid_loader = DataLoader(**loader_kwargs, **valid_loader_kwargs)
    
    _logger.info("Total parameters in model student:{}".format(sum(p.numel() for p in model_student.parameters())))
    original_stdout = sys.stdout # Save a reference to the original standard output
    with open(workdir+'/model_student.txt', 'w') as f:
        sys.stdout = f # Change the standard output to the file we created.
        count_parameters(model_student)
        model_student.to(device)
        if (args.type=="staticquant"):
            measure_size(model_student)

        if(args.teacher):
                
                _logger.info("Training with Knowledge Distillation")
                trainer = PruneKDTrainer(
                    model_teacher,
                    model_student, device, train_loader, valid_loader,
                    use_amp=False,
                    lr_scheduler_fn=lr_scheduler_fn,
                    restore_optim=args.restore_optim,
                    save_optim_every=args.save_optim_every,
                    grad_accum_split=args.grad_accum_split,
                    temp=temp,
                    alpha=alpha
                )
        else:
                if args.multi_gpu:
                    from torch.nn import DataParallel
                    model_student = DataParallel(model_student)
                    model_student.stride = model_student.module.stride
                    model_student.alphabet = model_student.module.alphabet
                _logger.info("Training without Knowledge Distillation")
                trainer = Trainer(
                    model_student, device, train_loader, valid_loader,
                    use_amp=False,
                    lr_scheduler_fn=lr_scheduler_fn,
                    restore_optim=args.restore_optim,
                    save_optim_every=args.save_optim_every,
                    grad_accum_split=args.grad_accum_split,
                    multi_gpu=args.multi_gpu
                )
        trainer.fit(workdir, args.epochs, args.lr)


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
    parser.add_argument('--name_append', default="")
    parser.add_argument("--full", action="store_true", default=False)
    parser.add_argument("--lr", default=2e-3, type=float)
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--chunks", default=0, type=int) # full dataset 1221470
    parser.add_argument("--temp", default=1, type=float)
    parser.add_argument("--alpha", default=0.1, type=float)
    parser.add_argument("--weights", default="0", type=str)
    parser.add_argument("--valid_chunks", default=0, type=int)
    parser.add_argument("--no-amp", action="store_true", default=True)
    parser.add_argument("--teacher", action="store_true", default=False)
    parser.add_argument("-f", "--force", action="store_true", default=False)
    parser.add_argument("--restore-optim", action="store_true", default=False)
    parser.add_argument("--multi_gpu", action="store_true", default=False)
    parser.add_argument("--save-optim-every", default=10, type=int)
    parser.add_argument("--grad-accum-split", default=1, type=int)
    parser.add_argument("--type", default=None, type=str, choices=["bonitostaticquant", 'rubiconfp','rubiconqabas-mp','bonito','rubiconnoskipfp'])
    parser.add_argument("--remove_mask", action="store_true", default=False)
    return parser
