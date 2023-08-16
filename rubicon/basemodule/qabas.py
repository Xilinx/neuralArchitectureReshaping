#!/usr/bin/env python3
# qabas.py -*- Python -*-
#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2023 Advanced Micro Devices, Inc.

"""
Rubicon QABAS.

"""

import os
import sys
from argparse import ArgumentParser 
from argparse import ArgumentDefaultsHelpFormatter
from pathlib import Path
from importlib import import_module
import torch.nn as nn
from os import system
from bonito.data import load_numpy
from rubicon.data import load_numpy_shuf,load_numpy_full
from rubicon.tools.ont.bonito.data import load_script
from rubicon.util import __models__, default_data
from rubicon.tools.ont.bonito.util import load_symbol, init
from rubicon.training import load_state, Trainer
import json
import toml
import torch
import numpy as np
from torch.utils.data import DataLoader
from rubicon.tools.nni.nni.retiarii.nn.pytorch.api import LayerChoice, InputChoice
from rubicon.nas.dartsbasecalling import DartsBasecalling
from rubicon.nas.proxylessbasecalling import ProxylessBasecalling
import torch.onnx

import time
import datetime
import logging
sys.setrecursionlimit(20000)

_logger = logging.getLogger(__name__)
def get_parameters(model, keys=None, mode='include'):
    if keys is None:
        for name, param in model.named_parameters():
            yield param
    elif mode == 'include':
        for name, param in model.named_parameters():
            flag = False
            for key in keys:
                if key in name:
                    flag = True
                    break
            if flag:
                yield param
    elif mode == 'exclude':
        for name, param in model.named_parameters():
            flag = True
            for key in keys:
                if key in name:
                    flag = False
                    break
            if flag:
                yield param
    else:
        raise ValueError('do not support: %s' % mode)

def main(args):
    workdir = os.path.expanduser(args.save_directory)
    _logger.info("Start date and time:{}".format(datetime.datetime.now()))
    if os.path.exists(workdir) and not args.force:
        print("[error] %s exists, use -f to force continue training." % workdir)
        exit(1)
    
    os.makedirs(workdir, exist_ok=True)
    init(args.seed, args.device)
    device = torch.device(args.device)
    
    config_file = args.config
    if not os.path.exists(config_file):
        print("[error] %s does not" % config_file)
        exit(1)
    config = toml.load(config_file)
    if not args.nas:
        _logger.warning("Please speficy which type of NAS using --nas argument")
        exit(1)
    _logger.info("[loading model]")
    if args.applied_hardware=="aie_lut":
        _logger.info("Quantized Search Space")
        model = load_symbol(config, 'BaseModelQuant')(config)     
    else:
        _logger.info("Unquantized Search Space")
        model = load_symbol(config, 'BaseModel')(config)
    
    if args.grad_reg_loss_type == 'add#linear':
        grad_reg_loss_params = {'lambda': args.grad_reg_loss_lambda}
    elif args.grad_reg_loss_type == 'mul#log':
        grad_reg_loss_params = {
            'alpha': args.grad_reg_loss_alpha,
            'beta': args.grad_reg_loss_beta,
        }
    else:
        args.grad_reg_loss_params = None

    if (args.applied_hardware=="aie_lut"):
        hardware='aie_lut'
    elif(args.applied_hardware=="cpu"):
        hardware='cortexA76cpu_tflite21'
    elif(args.applied_hardware=="gpu"):
        hardware= 'adreno640gpu_tflite21'
    else:
        hardware=None
    
    _logger.info("NAS type:{}".format(args.nas))
    _logger.info("Hardware type:{}".format(hardware))
    _logger.info("Reference latency:{}".format(args.reference_latency))
    _logger.info("lambda:{}".format(args.grad_reg_loss_lambda))
    _logger.info("[loading data]")
 
    if args.full:
        _logger.info("Full dataset training")
        train_loader_kwargs, valid_loader_kwargs = load_numpy_full(None,
                args.directory
        )
    elif args.chunks:
        _logger.info("Not full dataset training with shuffling")
        train_loader_kwargs, valid_loader_kwargs = load_numpy_shuf(
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
    
    if args.nas == 'darts':
        #### setting optimizer #######  
        if args.rubicon:
        
            optimizer = None
        else:
            _logger.info("SGD Optimizer")
            
            if args.default: #default from darts
                lr=0.025
            else:
                lr=args.lr #rubicon learning rate 2e-3
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=3.0E-4)
        
        #### setting lr scheduler #######
        if args.rubicon:
            _logger.info("Scheduler: Linear Warmup")
            if config.get("lr_scheduler"):
                sched_config = config["lr_scheduler"]
                lr_scheduler_fn = getattr(
                    import_module(sched_config["package"]), sched_config["symbol"]
                )(**sched_config)
                print("building scheduler",getattr(
                    import_module(sched_config["package"]), sched_config["symbol"]
                )(**sched_config))
            else:
                print("no scheduler")
                lr_scheduler_fn = None
        else:
            _logger.info("Scheduler: CosineAnnealing")
            lr_scheduler_fn=None 


        trainer = DartsBasecalling(
                model=model,
                train_loader=train_loader, 
                valid_loader=valid_loader,
                optimizer=optimizer,
                lr_scheduler_fn=lr_scheduler_fn,
                ctrl_learning_rate=args.ctlr,
                opt_learning_rate=args.lr,
                applied_hardware=hardware,
                metrics=lambda output, target: accuracy(output, target, topk=(1, 5,)),
                log_frequency=10,
                grad_reg_loss_type=args.grad_reg_loss_type, 
                grad_reg_loss_params=grad_reg_loss_params, 
                dummy_input=(344,1,9),
                ref_latency=args.reference_latency,
                rubicon=args.rubicon,
                default=args.default,
                num_epochs=args.epochs
            )

    elif args.nas == 'proxy':
        #### setting optimizer STEP 2 UPDATE WEIGHTS #######
        if args.rub_arch_opt:
            optimizer = None
            if args.default:
                lr=2e-3
            else:
                lr=args.lr
        elif args.prox_arch_opt:
            if args.default:
                lr=0.05
            else:
                lr=args.lr
            if args.no_decay_keys:
                keys = args.no_decay_keys
                momentum, nesterov = 0.9, True
                optimizer = torch.optim.SGD([
                    {'params': get_parameters(model, keys, mode='exclude'), 'weight_decay': 4e-5},
                    {'params': get_parameters(model, keys, mode='include'), 'weight_decay': 0},
                ], lr=lr, momentum=momentum, nesterov=nesterov)
           
            else:
                momentum, nesterov = 0.9, True
                optimizer = torch.optim.SGD(get_parameters(model), lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=4e-5)
        lr_scheduler_fn=None  

        trainer = ProxylessBasecalling(
                model=model,
                train_loader=train_loader, 
                valid_loader=valid_loader,
                optimizer=optimizer,
                lr_scheduler_fn=lr_scheduler_fn,
                ctrl_learning_rate=args.ctlr,
                applied_hardware=hardware,
                metrics=lambda output, target: accuracy(output, target, topk=(1, 5,)),
                log_frequency=10,
                grad_reg_loss_type=args.grad_reg_loss_type, 
                grad_reg_loss_params=grad_reg_loss_params, 
                dummy_input=(344,1,9),
                ref_latency=args.reference_latency,
                rubicon=args.rubicon,
                default=args.default,
                num_epochs=args.epochs,
                rub_sched=args.rub_sched,
                dart_sched=args.dart_sched,
                rub_ctrl_opt=args.rub_ctrl_opt,
                prox_ctrl_opt=args.prox_ctrl_opt               
            )

    trainer.fit(workdir, args.epochs, args.lr)
    final_architecture = trainer.export()
    _logger.info("Final architecture:{}".format(trainer.export()))
   
    # the json file where the output must be stored
    out_file = open(args.arc_checkpoint, "w")
    json.dump(final_architecture, out_file, indent = 6)
    out_file.close()
    
    _logger.info("JSON file saved at:{}".format(os.path.expanduser(args.arc_checkpoint)))
    _logger.info("End date and time:{}".format(datetime.datetime.now()))




def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("save_directory")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--config', default="models/configs/config.toml")
    parser.add_argument("--directory", type=Path,default="tools/ont/bonito/data/dna_r9.4.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--chunks", default=0, type=int)
    parser.add_argument("--arc_checkpoint", default="final_arch.json")
    parser.add_argument("--valid_chunks", default=0, type=int)
    parser.add_argument("--no-amp", action="store_true", default=True)
    parser.add_argument("--full", action="store_true", default=False)
    parser.add_argument("-f", "--force", action="store_true", default=False)

    parser.add_argument("--nas", default=None, type=str, choices=["darts", 'proxy'])
    parser.add_argument("--no_decay_keys", default='bn', type=str, choices=[None, 'bn', 'bn#bias'])
    parser.add_argument('--grad_reg_loss_type', default='add#linear', type=str, choices=['add#linear', 'mul#log'])
    parser.add_argument('--grad_reg_loss_lambda', default=6e-1, type=float)  # grad_reg_loss_params
    parser.add_argument('--grad_reg_loss_alpha', default=0.2, type=float)  # grad_reg_loss_params
    parser.add_argument('--grad_reg_loss_beta',  default=0.3, type=float)  # grad_reg_loss_params
    parser.add_argument("--applied_hardware", default="aie_lut",choices=["aie_lut", 'cpu','gpu'], type=str, help='the hardware to predict model latency')
    parser.add_argument("--reference_latency", default=65.0, type=float, help='the reference latency in specified hardware')
    parser.add_argument("--rubicon", action="store_true", default=False)

    parser.add_argument("--ctlr", default=2e-3, type=float)
    parser.add_argument("--lr", default=2e-3, type=float)
    parser.add_argument("--default", action="store_true", default=False)

    parser.add_argument("--rub_arch_opt", action="store_true", default=False)
    parser.add_argument("--prox_arch_opt", action="store_true", default=False)

    parser.add_argument("--rub_ctrl_opt", action="store_true", default=False)
    parser.add_argument("--prox_ctrl_opt", action="store_true", default=False)

    parser.add_argument("--rub_sched", action="store_true", default=False)
    parser.add_argument("--dart_sched", action="store_true", default=False)

    return parser
