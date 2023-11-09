# common.py -*- Python -*-
#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2023 Advanced Micro Devices, Inc.

# pylint: disable=unused-import

import argparse
import torch
import torch.nn as nn
import csv
import copy
import sys

# Use MS MXFP

def add_exp_args(parser):
    parser.add_argument("--dtype", type=str, default="float32", help="data type code")
    # Use MS MXFP
    parser.add_argument("--exclude", type=str, nargs="+", default=[], help="do not quantize these layers")
    parser.add_argument("--include", type=str, nargs="+", default=[], help="only quantize these layers")
    parser.add_argument("--include2", type=str, nargs="+", default=[], help="only quantize these layers, using dtype2")
    parser.add_argument("--dtype2", type=str, default="float32", help="second data type code")
    parser.add_argument("--verbose", action="store_true", help="verbose output")
    parser.add_argument("--xfirst", action="store_true", help="exclude first layer")
    parser.add_argument("--xlast", action="store_true", help="exclude last layer")
    parser.add_argument("--profile", action="store_true", help="runtime profile")
    parser.add_argument("--metrics", type=float, default=0.0, help="probability for metric tracking")
    parser.add_argument("--metrics_float_forward", action="store_true", help="for metrics, pass reference float product forward")
    parser.add_argument("--dense", type=int, default=None, help="initial sparsity dense setting")
    parser.add_argument("--blocksize", type=int, default=None, help="initial sparsity blocksize")
    parser.add_argument("--output_path", type=str, help="file output path")
    parser.add_argument("--num_samples", type=int, default=None, help="number of data samples")
    parser.add_argument("--gpu", type=int, default=0, help="Target GPU (in non-distributed mode)")
    return parser

def init_distributed_mode(args):
    import os
    dist_url = "env://"
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
        torch.distributed.init_process_group('nccl', init_method=dist_url, world_size=args.world_size, rank=args.rank)
        torch.cuda.set_device(args.gpu)
        torch.distributed.barrier()
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
        torch.distributed.init_process_group('nccl', init_method=dist_url, rank=args.rank)
    else:
        print('Not using distributed mode')
        args.distributed = False
    return

def validate(loader, model, criterion, rank=None, dist=False, print_freq=50):
    """ run the model on the test set """
    model.eval()
    tot_loss, mean_loss, tot_top1, mean_top1, count = 0.0, 0.0, 0.0, 0.0, 0
    ddp_bucket=torch.zeros((4,)).cuda()
    with torch.no_grad():
        for iteration, (inp, target) in enumerate(loader):
            inp, target = inp.cuda(), target.cuda()
            output = model(inp)
            loss = criterion(output, target)
            tot_top1 += output.max(1).indices.eq(target.view(1, -1)).sum().item()
            tot_loss += loss.item()
            count += target.size(0)
            ddp_bucket[0] = tot_top1
            ddp_bucket[1] = tot_loss
            ddp_bucket[2] = count
            ddp_bucket[3] = iteration
            mean_top1, mean_loss = tot_top1 / count * 100, tot_loss / (iteration + 1)
            if iteration and iteration % print_freq == 0:
                print(f"iter {iteration}: top1 {mean_top1:7.4f}% loss {mean_loss:7.5f}")
    if dist:
        torch.distributed.barrier()
        torch.distributed.reduce(ddp_bucket, 0)
    if dist and rank == 0:
        m_top1, m_loss = ddp_bucket[0] / ddp_bucket[2] * 100, ddp_bucket[1] / (ddp_bucket[3] + 1)
        print(f"FINAL_rank0: top1 {m_top1:7.4f}% loss {m_loss:7.5f}")
        return m_top1.item(), m_loss.item()
    print(f"FINAL: top1 {mean_top1:7.4f}% loss {mean_loss:7.5f}")
    return mean_top1, mean_loss

def init_model_and_loader(args, train=False, pretrained=True):
    """obtain and initialize model"""
    if args.distributed:
        init_distributed_mode(args)
    model, loader, train_loader, meta = get_model_and_loaders(args, train, pretrained)
    model = model.cuda()

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    isfloat = # Use MS MXFP(code=args.dtype).is_float32()
    if isfloat:
        args.metrics = 0.0
    if args.metrics:
        if args.metrics < 1.0:
            args.metrics_float_forward = False

    return model, loader, train_loader, meta

def apply_changes(model, args):
    isfloat = # Use MS MXFP(code=args.dtype).is_float32()
    isfloat2 = # Use MS MXFP(code=args.dtype2).is_float32()
    mnames = list(model.# Use MS MXFP.keys())
    first, last = mnames[0], mnames[-1]
    excl, incl = [], args.include
    if not isfloat:
        backing = "fp32" if args.metrics else None
        if args.xfirst:
            excl.append(first)
        if args.xlast:
            excl.append(last)
        if not excl:
            excl = args.exclude
        if excl and incl:
            raise ValueError("Cannot have both include and exclude layers")
        if incl:
            change = dict(layer=dict(atype=args.dtype, btype=args.dtype, bdense=args.dense, bsbs=args.blocksize, backing=backing), include=incl, epoch=0)
        else:
            change = dict(layer=dict(atype=args.dtype, btype=args.dtype, bdense=args.dense, bsbs=args.blocksize, backing=backing), exclude=excl, epoch=0)
        model.change(# Use MS MXFP(**change))
        if not isfloat2 and args.include2:
            change = dict(layer=dict(atype=args.dtype2, btype=args.dtype2, backing=backing), include=args.include2, epoch=0)
            model.change(# Use MS MXFP(**change))

    return model, incl, excl

def run_inference(model, loader, meta, args, incl, excl):
    s = # Use MS MXFP
    isfloat = # Use MS MXFP(code=args.dtype).is_float32()
    isfloat2 = # Use MS MXFP(code=args.dtype2).is_float32()
    mnames = list(model.# Use MS MXFP.keys())
    first, last = mnames[0], mnames[-1]
    criterion = nn.CrossEntropyLoss().cuda()
    exptype = # Use MS MXFP
    desc = f"{args.dtype}"
    if excl:
        flcode = ""
        if first in excl:
            flcode += "f"
        if last in excl:
            flcode += "l"
        if not flcode:
            flcode += str(len(excl))
        if flcode:
            desc += f"_ex{flcode}"
    elif not isfloat2 and args.include2:
        desc += f"_{args.dtype2}"
    if args.# Use MS MXFP == "torch":
        desc += "_torch"
    if args.metrics:
        fwd = "qf"[args.metrics_float_forward]
        desc += f"_m{fwd}{args.metrics}"
    if args.# Use MS MXFP != "even":
        desc += f"_r{args.# Use MS MXFP}"
    if args.# Use MS MXFP:
        desc += "_noclip"
    if args.distributed:
        desc += "dist"+"_rank_"+str(args.rank)

    if (not args.distributed) or (args.distributed and args.rank==0):
        record = # Use MS MXFP
    else:
        record = # Use MS MXFP

    with record:
        torch.backends.cudnn.benchmark = False
        if args.distributed:
            top1, loss = validate(loader, model, criterion, rank=args.rank, dist=True)
        else:
            top1, loss = validate(loader, model, criterion)
        
        if (not args.distributed) or (args.distributed and args.rank==0):
            print(record.record["description"]["name"])
            record.add_to_record("weight_metadata", meta)
            record.add_results("final inference", dict(top1=top1, loss=loss))
            print(f"RESULT: top1 {top1:7.4f}% loss {loss:7.5f}, {s}")
    return top1, loss
