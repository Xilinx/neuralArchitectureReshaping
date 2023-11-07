# skipclip.py -*- Python -*-
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
import logging
from rubicon.data import load_numpy_shuf,load_numpy_full
from bonito.data import load_script
from bonito.util import load_symbol, init,load_model
from rubicon.util import __models__, default_config, default_data
from rubicon.util import load_model_prune, load_model_prune_for_kd
from rubicon.kdtraining import load_state, KDTrainer
import toml
import torch
import numpy as np
from torch.utils.data import DataLoader
from contextlib import redirect_stdout
import subprocess
import shutil
from prettytable import PrettyTable
import logging
_logger = logging.getLogger(__name__)



def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
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
    _logger.info("Save path: {}".format(workdir))

    
    if args.force and os.path.exists(workdir) :
        shutil.rmtree(workdir)
    if os.path.exists(workdir) and not args.force:
        print("[error] %s exists, remove it to continue or use -f to force delete." % workdir)
        exit(1)
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
    os.makedirs(workdir, exist_ok=True)    
    teacher_path=os.path.expanduser(args.teacher_directory)
    _logger.info("Loading Teacher Model:{}".format(teacher_path))
    model_teacher = load_model(teacher_path, device, half=False)
    original_stdout = sys.stdout # Save a reference to the original standard output
    with open(workdir+'/model_student.txt', 'w') as f:
            sys.stdout = f # Change the standard output to the file we created.       
            if(args.pretrained):   
                    _logger.info("Using Pretrained Student:{}".format(args.pretrained))
                    model_student = load_model_prune_for_kd(args.pretrained, device, half=False,load_model_type=args.stud_type, no_prune=args.no_prune)
            else:              
                if(args.type=="rubiconqabas-mp"):
                    _logger.info("Training a new student:{}".format(args.type)) 
                    model_student = load_symbol(config, 'RubiconSkipTrim')(config)  
            count_parameters(model_student)
          
            if config.get("lr_scheduler"):
                sched_config = config["lr_scheduler"]
                lr_scheduler_fn = getattr(
                    import_module(sched_config["package"]), sched_config["symbol"]
                )(**sched_config)
            else:
                lr_scheduler_fn = None
            
            _logger.info("Total parameters in model teacher:{}".format(sum(p.numel() for p in model_teacher.parameters()))) 
            _logger.info("Total parameters in model student:{}".format(sum(p.numel() for p in model_student.parameters())) )
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

            _logger.info("Starting SkipTrim")

            trainer = KDTrainer(
                model_teacher,model_student, device, train_loader, valid_loader,
                use_amp=False,
                lr_scheduler_fn=lr_scheduler_fn,
                restore_optim=args.restore_optim,
                save_optim_every=args.save_optim_every,
                grad_accum_split=args.grad_accum_split,
                temp=temp,
                alpha=alpha
            )

            trainer.fit(workdir, args.epochs, args.lr, args.skip_stride)
   

def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("training_directory")
    parser.add_argument("--teacher_directory", default="models/bonito")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--config', default="models/configs/config.toml")
    group.add_argument('--pretrained', default="")
    parser.add_argument('--name_append', default="")
    parser.add_argument("--directory", type=Path,default="tools/ont/bonito/data/dna_r9.4.1")
    parser.add_argument("--device", default="cuda")
    parser.add_argument('--student_dir', default="")
    parser.add_argument("--full", action="store_true", default=False)
    parser.add_argument("--no_prune", action="store_true", default=True)
    parser.add_argument("--lr", default=2e-3, type=float)
    parser.add_argument("--temp", default=1, type=float)
    parser.add_argument("--alpha", default=0.1, type=float)
    parser.add_argument("--seed", default=25, type=int)
    parser.add_argument("--epochs", default=5, type=int)
    parser.add_argument("--skip_stride", default=1, type=int)
    parser.add_argument("--batch", default=64, type=int)
    parser.add_argument("--chunks", default=0, type=int)
    parser.add_argument("--valid-chunks", default=0, type=int)
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("-f", "--force", action="store_true", default=False)
    parser.add_argument("--restore-optim", action="store_true", default=False)
    parser.add_argument("--save-optim-every", default=10, type=int)
    parser.add_argument("--grad-accum-split", default=1, type=int)
    parser.add_argument("--stud_type", default="rubiconqabas", type=str, choices=['rubiconqabas'])
    parser.add_argument("--type", default="rubiconqabas-mp", type=str, choices=['rubiconqabas-mp'])
    return parser
# 