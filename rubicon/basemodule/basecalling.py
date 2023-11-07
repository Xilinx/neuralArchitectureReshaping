# basecalling.py -*- Python -*-
#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2023 Advanced Micro Devices, Inc.

import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from time import perf_counter

from datetime import timedelta
from itertools import islice as take
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

from bonito.ctc.basecall  import basecall as basecallctc
from bonito.crf.basecall  import basecall as basecallcrf

from bonito.aligner import align_map, Aligner

from bonito.io import CTCWriter, Writer, biofmt
from bonito.mod_util import call_mods, load_mods_model
# from rubicon.ioold import CTCWriter, Writer
# from rubicon.util import load_model_prune_for_kd
from bonito.util import load_model,column_to_set

# from rubicon.fast5 import get_reads, read_chunks
from bonito.reader import read_chunks, Reader
from bonito.multiprocessing import process_cancel
from rubicon.util import load_model_prune_for_basecall,load_model_kd_basecall
import subprocess
from prettytable import PrettyTable
# from brevitas.export.onnx.generic.manager import BrevitasONNXManager
import torch.onnx

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


    sys.stderr.write("***********************************\n")
    path, dirs, files = next(os.walk(args.reads_directory))
    reading = args.reads_directory.split('/')[1].strip()
    sys.stderr.write("> Reading:%s\n"%args.reads_directory)
    try:
        reader = Reader(args.reads_directory, args.recursive)
        sys.stderr.write("> reading %s\n" % reader.fmt)
    except FileNotFoundError:
        sys.stderr.write("> error: no suitable files found in %s\n" % args.reads_directory)
        exit(1)
    fmt = biofmt(aligned=args.reference is not None)
    torch.backends.cudnn.benchmark = True
    file_count = len(files)
    sys.stderr.write(f"> loading model {args.model_directory}\n")
    sys.stderr.write(f"> loading weights {args.weights}\n")
    infer_type=str(args.infer_type)
    # try:

    if(args.remove_mask):
        remove_mask=True
    else:
        remove_mask=False
    modeltype=str(args.type)

    if(modeltype in ["rubiconskiptrim","rubicallmp"]):
        model = load_model_kd_basecall(args.model_directory, 
        args.device, half=False,
        load_model_type=modeltype, quantize=True,
        no_prune=False)    
    # elif(modeltype=="crf-fast"): 
    #     model = load_model(
    #     args.model_directory,
    #     args.device,
    #     # type=modeltype,
    #     weights=int(args.weights),
    #     chunksize=args.chunksize,
    #     batchsize=args.batchsize,
    #     quantize=args.quantize,
    #     use_koi=True
    # )
    elif(modeltype in ["crf-sup","crf-fast"]): 
        print("RNN model")
        model = load_model(
            args.model_directory,
            args.device,
            weights=args.weights if int(args.weights) > int(0) else None,
            chunksize=args.chunksize,
            overlap=args.overlap,
            batchsize=args.batchsize,
            quantize=args.quantize,
            use_koi=True,
        )
    else:
        model = load_model_prune_for_basecall(
            args.model_directory,
            args.device, 
            half=False,
            load_model_type=modeltype,
            weights=int(args.weights),
            chunksize=args.chunksize,
            overlap=args.overlap,
            batchsize=args.batchsize,
            use_koi=False,
            remove_mask=remove_mask)

    
    totalparam=sum(p.numel() for p in model.parameters())
    sys.stderr.write("> Model parameters: %s\n" %totalparam)
    if args.reference and args.reference.endswith(".mmi") and fmt.name == "cram":
        sys.stderr.write("> error: reference cannot be a .mmi when outputting cram\n")
        exit(1)
    elif args.reference and fmt.name == "fastq":
        sys.stderr.write(f"> warning: did you really want {fmt.aligned} {fmt.name}?\n")
    else:
        sys.stderr.write(f"> outputting {fmt.aligned} {fmt.name}\n")
    mods_model = None   
    if args.modified_base_model is not None or args.modified_bases is not None:
        sys.stderr.write("> loading modified base model\n")
        mods_model = load_mods_model(
            args.modified_bases, args.model_directory, args.modified_base_model,
            device=args.modified_device,
        )
        sys.stderr.write(f"> {mods_model[1]['alphabet_str']}\n")
    if args.reference:
        sys.stderr.write("> loading reference\n")
        aligner = Aligner(args.reference, preset='map-ont', best_n=1)
        if not aligner:
            sys.stderr.write("> failed to load/build index\n")
            exit(1)
    else:
        aligner = None

    if args.save_ctc and not args.reference:
        sys.stderr.write("> a reference is needed to output ctc training data\n")
        exit(1)

    if fmt.name != 'fastq':
        groups, num_reads = reader.get_read_groups(
            args.reads_directory, args.model_directory,
            n_proc=8, recursive=args.recursive,
            read_ids=column_to_set(args.read_ids), skip=args.skip,
            cancel=process_cancel()
        )
    else:
        groups = []
        num_reads = None

    reads = reader.get_reads(
        args.reads_directory, n_proc=8, recursive=args.recursive,
        read_ids=column_to_set(args.read_ids), skip=args.skip,
        do_trim=not args.no_trim, cancel=process_cancel()
    )

    if args.max_reads:
        reads = take(reads, args.max_reads)

    if args.save_ctc:
        reads = (
            chunk for read in reads
            for chunk in read_chunks(
                read,
                chunksize=model.config["basecaller"]["chunksize"],
                overlap=model.config["basecaller"]["overlap"]
            )
        )
        ResultsWriter = CTCWriter
    else:
        ResultsWriter = Writer
    ResultsWriter = Writer
        # basecall = load_symbol(args.model_directory, "basecall")

    if(args.batchsize>0):
        cust_batch=args.batchsize
    else:
        cust_batch=model.config["basecaller"]["batchsize"]
    
    sys.stderr.write("> Batch size:%s\n"%cust_batch)

    if(modeltype in ["crf-sup","crf-fast"]): 
        print("running crf basecalling")
        results = basecallcrf(
        model, reads, reverse=args.revcomp,
        batchsize=cust_batch,
        chunksize=model.config["basecaller"]["chunksize"],
        overlap=model.config["basecaller"]["overlap"]
    )
    else:
        results = basecall(
            model, reads, reverse=args.revcomp,
            batchsize=cust_batch, chunksize=args.chunksize,
        )
        
    if mods_model is not None:
        if args.modified_device:
            results = ((k, call_mods(mods_model, k, v)) for k, v in results)
        else:
            results = process_itemmap(
                partial(call_mods, mods_model), results, n_proc=args.modified_procs
            )
    if aligner:
        results = align_map(aligner, results, n_thread=args.alignment_threads)

    writer = ResultsWriter(
        fmt.mode, tqdm(results, desc="> calling", unit=" reads", leave=False,
                    total=num_reads, smoothing=0, ascii=True, ncols=100),
        aligner=aligner, group_key=args.model_directory,
        ref_fn=args.reference, groups=groups, min_qscore=args.min_qscore
    )
    # writer = Writer(
    #     tqdm(results, desc="> calling", unit=" reads", leave=False),
    #     results=aligner
    # )
    t0 = perf_counter()
    writer.start()
    writer.join()
    duration = perf_counter() - t0
    num_samples = sum(num_samples for read_id, num_samples in writer.log)

    sys.stderr.write("> Model type: %s\n" %args.type)
    sys.stderr.write("> Model parameters: %s\n" %totalparam)
    sys.stderr.write("> completed reads: %s\n" % len(writer.log))
    sys.stderr.write("> time: {:.4f}s\n".format(duration))
    sys.stderr.write("> duration: %s\n" % timedelta(seconds=np.round(duration)))
    sys.stderr.write("> samples per second %.1E\n" % (num_samples / duration))
    sys.stderr.write("> done\n")
    sys.stderr.write("***********************************\n")

def argparser():
    parser = ArgumentParser(
        formatter_class=ArgumentDefaultsHelpFormatter,
        add_help=False
    )
    parser.add_argument("model_directory")
    parser.add_argument("reads_directory")
    parser.add_argument("--reference")
    parser.add_argument("--modified-bases", nargs="+")
    parser.add_argument("--modified-base-model")
    parser.add_argument("--modified-procs", default=8, type=int)
    parser.add_argument("--modified-device", default=None)

    parser.add_argument("--read-ids")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", default=25, type=int)

    parser.add_argument("--weights", default="1", type=str)
    parser.add_argument("--skip", action="store_true", default=False)
    parser.add_argument("--no-trim", action="store_true", default=False)
    parser.add_argument("--save-ctc", action="store_true", default=False)
    parser.add_argument("--revcomp", action="store_true", default=False)
    parser.add_argument("--infer_type", type=str)
    parser.add_argument("--recursive", action="store_true", default=False)
    quant_parser = parser.add_mutually_exclusive_group(required=False)
    quant_parser.add_argument("--quantize", dest="quantize", action="store_true")
    quant_parser.add_argument("--no-quantize", dest="quantize", action="store_false")
    parser.set_defaults(quantize=None)
    parser.add_argument("--overlap", default=800, type=int)
    parser.add_argument("--chunksize", default=10000, type=int)
    parser.add_argument("--batchsize", default=0, type=int)
    parser.add_argument("--max-reads", default=0, type=int)
    parser.add_argument("--alignment-threads", default=8, type=int)
    parser.add_argument('-v', '--verbose', action='count', default=0)
    parser.add_argument("--min-qscore", default=0, type=int)
    parser.add_argument("--type", default=None, type=str, choices=["staticquant","crf-sup","crf-fast",'rubiconqabas','bonito','rubiconskiptrim','rubicallmp','rubiconnoskipfp'])
    parser.add_argument("--remove_mask", action="store_true", default=True)
    parser.add_argument("--no_prune", action="store_true", default=False)
    return parser

 