# data.py -*- Python -*-
#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2023 Advanced Micro Devices, Inc.

import importlib
import os
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader
import logging
__dir__ = os.path.dirname(os.path.realpath(__file__))
__data__ = os.path.join(__dir__, "data")
__models__ = os.path.join(__dir__, "models")
__configs__ = os.path.join(__dir__, "models/configs")
default_data = os.path.join(__data__, "dna_r9.4.1")
_logger = logging.getLogger(__name__)

class ChunkDataSet:
    def __init__(self, chunks, targets, lengths):
        self.chunks = np.expand_dims(chunks, axis=1)
        self.targets = targets
        self.lengths = lengths

    def __getitem__(self, i):
        return (
            self.chunks[i].astype(np.float32),
            self.targets[i].astype(np.int64),
            self.lengths[i].astype(np.int64),
        )

    def __len__(self):
        return len(self.lengths)
def load_numpy_full(limit, directory):
    """
    Returns training and validation DataLoaders for data in directory without shuffling
    """

    train_data = load_numpy_datasets_full(limit=limit, directory=directory)
    if directory is None:
        directory = default_data
    
    _logger.info("Loading training dataset from:{}".format(directory)) 
    if os.path.exists(os.path.join(directory, 'validation')):
        _logger.info("Loading validation dataset from:{}".format(os.path.join(directory, 'validation')))
        valid_data = load_numpy_datasets_full(
            directory=os.path.join(directory, 'validation')
        )
    else:
        _logger.info("Validation set not found: splitting training set")
        split = np.floor(len(train_data[0]) * 0.97).astype(np.int32)
        valid_data = [x[split:] for x in train_data]
        train_data = [x[:split] for x in train_data]

    train_loader_kwargs = {"dataset": ChunkDataSet(*train_data), "shuffle": True}
    valid_loader_kwargs = {"dataset": ChunkDataSet(*valid_data), "shuffle": False}
    return train_loader_kwargs, valid_loader_kwargs
    
def load_numpy_shuf(train_limit, valid_limit, directory):
    """
    Returns training and validation DataLoaders for data in directory with shuffling
    """
    
    train_data = load_numpy_datasets(limit=train_limit, directory=directory)
    if directory is None:
        directory = default_data
    _logger.info("Loading training dataset from:{}".format(directory))

    if os.path.exists(os.path.join(directory, 'validation')):
        _logger.info("Loading validation dataset from:{}".format(os.path.join(directory, 'validation')))
        valid_data = load_numpy_datasets(limit=valid_limit, 
            directory=os.path.join(directory, 'validation')
        )
    else:
        _logger.info("Validation set not found: splitting training set")
        split = np.floor(len(train_data[0]) * 0.97).astype(np.int32)
        valid_data = [x[split:] for x in train_data]
        train_data = [x[:split] for x in train_data]

    train_loader_kwargs = {"dataset": ChunkDataSet(*train_data), "shuffle": True}
    valid_loader_kwargs = {"dataset": ChunkDataSet(*valid_data), "shuffle": False}
    return train_loader_kwargs, valid_loader_kwargs

def load_numpy_datasets_full(limit=None, directory=None):
    """
    Returns numpy chunks, targets and lengths arrays.
    """
    if directory is None:
        directory = default_data

    chunks = np.load(os.path.join(directory, "chunks.npy"), mmap_mode='r')
    targets = np.load(os.path.join(directory, "references.npy"), mmap_mode='r')
    lengths = np.load(os.path.join(directory, "reference_lengths.npy"), mmap_mode='r')

    indices = os.path.join(directory, "indices.npy")

    if os.path.exists(indices):
        idx = np.load(indices, mmap_mode='r')
        idx = idx[idx < lengths.shape[0]]
        if limit:
            idx = idx[:limit]
        return chunks[idx, :], targets[idx, :], lengths[idx]

    if limit:
        chunks = chunks[:limit]
        targets = targets[:limit]
        lengths = lengths[:limit]

    return np.array(chunks), np.array(targets), np.array(lengths)


def load_numpy_datasets(limit=None, directory=None):
    """
    Returns numpy chunks, targets and lengths arrays.
    """
    if directory is None:
        directory = default_data

    chunks_full = np.load(os.path.join(directory, "chunks.npy"), mmap_mode='r')
    targets_full = np.load(os.path.join(directory, "references.npy"), mmap_mode='r')
    lengths_full = np.load(os.path.join(directory, "reference_lengths.npy"), mmap_mode='r')

    indices = os.path.join(directory, "indices.npy")
    if(limit):
        _logger.info("Dataset length: {}/{}".format(limit,len(chunks_full)))

    else:
        _logger.info("Dataset length: {}/{}".format(len(chunks_full),len(chunks_full)))
    # print("seed is ",int(time.perf_counter())) #giving a random seed
    # np.random.seed(int(time.perf_counter()))
    shuffler = np.random.permutation(len(chunks_full))
    # print(shuffler)
    chunks=chunks_full[shuffler]
    targets=targets_full[shuffler]
    lengths=lengths_full[shuffler]

    if os.path.exists(indices):
        _logger.info("Shuffling indices")
        idx = np.load(indices, mmap_mode='r')
        idx = idx[idx < lengths.shape[0]]
        if limit:
            idx = idx[:limit]
        return chunks[idx, :], targets[idx, :], lengths[idx]
    
    if limit:
        chunks = chunks[:limit]
        targets = targets[:limit]
        lengths = lengths[:limit]

    return np.array(chunks), np.array(targets), np.array(lengths)


