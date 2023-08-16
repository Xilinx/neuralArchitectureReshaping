# prunekdtraining.py -*- Python -*-
#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2023 Advanced Micro Devices, Inc.

"""
rubicon train
"""

import os
import re
from glob import glob
from functools import partial
from time import perf_counter
from collections import OrderedDict
from datetime import datetime

from rubicon.tools.ont.bonito.schedule import linear_warmup_cosine_decay
from rubicon.tools.ont.bonito.util  import accuracy, decode_ref, permute, concat, match_names
import rubicon
from rubicon.tools.ont.bonito.io import CSVLogger
import torch.nn.functional as F
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.cuda.amp as amp
import brevitas.nn as qnn
from prettytable import PrettyTable
from torch.nn.utils import prune
import logging
_logger = logging.getLogger(__name__)

def remove_prune_mask(model, layer_type):
        for name, module in model.encoder.named_modules():
            # print("Name={},  Layer type={}".format(name, layer_type))
            if isinstance(module, layer_type):
                prune.remove(module, 'weight')

        for name, module in model.decoder.named_modules():
            if isinstance(module, layer_type):
                prune.remove(module, 'weight')   
        return model

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
def load_state(dirname, device, model_student, optim=None):
    """
    Load a model_student state dict from disk
    """
    model_student.to(device)
    if hasattr(model_student, "module"):
        model_student = model_student.module

    weight_no = optim_no = None

    optim_files = glob(os.path.join(dirname, "optim_*.tar"))
    optim_nos = {int(re.sub(".*_([0-9]+).tar", "\\1", w)) for w in optim_files}

    weight_files = glob(os.path.join(dirname, "weights_*.tar"))
    weight_nos = {int(re.sub(".*_([0-9]+).tar", "\\1", w)) for w in weight_files}

    if optim is not None:
        weight_no = optim_no = max(optim_nos & weight_nos, default=None)
    else:
        weight_no = max(weight_nos, default=None)

    to_load = []
    if weight_no:
        to_load.append(("weights", model_student))
    if optim_no:
        to_load.append(("optim", optim))

    if to_load:
        print("[picking up %s state from epoch %s]" % (', '.join([n for n, _ in to_load]), weight_no))
        for name, obj in to_load:
            state_dict = torch.load(
                os.path.join(dirname, '%s_%s.tar' % (name, weight_no)), map_location=device
            )
            if name == "weights":
                state_dict = {k2: state_dict[k1] for k1, k2 in match_names(state_dict, obj).items()}
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace('module.', '')
                    new_state_dict[name] = v
                state_dict = new_state_dict
            obj.load_state_dict(state_dict)
        epoch = weight_no
    else:
        epoch = 0

    return epoch


class PruneKDTrainer:
    def __init__(
        self, model_teacher,model_student, device, train_loader, valid_loader, criterion=None,
        use_amp=True, lr_scheduler_fn=None, restore_optim=False,
        save_optim_every=10, grad_accum_split=1, alpha=0.9, temp=1
    ):
        self.model_teacher = model_teacher.to(device)
        self.model_student = model_student.to(device)
        self.device = device
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion or (model_student.seqdist.ctc_loss if hasattr(model_student, 'seqdist') else model_student.ctc_label_smoothing_loss)
        self.use_amp = use_amp
        self.lr_scheduler_fn = lr_scheduler_fn or linear_warmup_cosine_decay()
        self.restore_optim = restore_optim
        self.save_optim_every = save_optim_every
        self.grad_accum_split = grad_accum_split
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.optimizer = None
        self.alpha=alpha
        self.temp=temp
    def loss_fn_kd(self,student_outputs, teacher_outputs, targets, lengths):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        "Hyperparameters": temperature and alpha
        NOTE: the KL Divergence for PyTorch comparing the softmaxs of teacher
        and student expects the input tensor to be log probabilities! See Issue #2
        """
        alpha = self.alpha
        T = self.temp
        KD_loss = nn.KLDivLoss()(F.log_softmax((student_outputs/T), dim=1),
                                F.softmax((teacher_outputs/T), dim=1)) * (alpha * T * T) + \
                self.criterion(student_outputs, targets,lengths)["loss"] * (1. - alpha)
        # KD_loss = nn.KLDivLoss()(F.log_softmax(student_outputs/T, dim=1),
        #                         F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
        #         F.cross_entropy(student_outputs, labels) * (1. - alpha)

        return KD_loss
    def train_one_step(self, batch):
        self.optimizer.zero_grad()
        # print("batch size", batch)
        losses = None
        with amp.autocast(enabled=self.use_amp):
            for data_, targets_, lengths_ in zip(*map(lambda t: t.chunk(self.grad_accum_split, dim=0), batch)):
                data_, targets_, lengths_ = data_.to(self.device), targets_.to(self.device), lengths_.to(self.device)
                # print(data_.size())
                scores_student_ = self.model_student(data_)
                with torch.no_grad():
                    scores_teacher_ = self.model_teacher(data_)
                losses_=self.loss_fn_kd(scores_student_.to(torch.float32), scores_teacher_.to(torch.float32), targets_, lengths_)
                # losses_ = self.criterion(scores_student_.to(torch.float32), targets_, lengths_)
                # print(losses_["loss"])
                if not isinstance(losses_, dict): losses_ = {'loss': losses_}

                losses_['loss'] = losses_['loss'] / self.grad_accum_split

                self.scaler.scale(losses_['loss']).backward()

                losses = {
                    k: (v.item() if losses is None else v.item() + losses[k])
                    for k, v in losses_.items()
                }

        self.scaler.unscale_(self.optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model_student.parameters(), max_norm=2.0).item()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return losses, grad_norm

    def train_one_epoch(self, loss_log, lr_scheduler):
        t0 = perf_counter()
        chunks = 0
        self.model_teacher.eval()
        self.model_student.train()

        progress_bar = tqdm(
            total=len(self.train_loader), desc='[0/{}]'.format(len(self.train_loader.sampler)),
            ascii=True, leave=True, ncols=100, bar_format='{l_bar}{bar}| [{elapsed}{postfix}]'
        )
        smoothed_loss = None

        with progress_bar:

            for batch in self.train_loader:

                chunks += batch[0].shape[0]

                losses, grad_norm = self.train_one_step(batch)

                smoothed_loss = losses['loss'] if smoothed_loss is None else (0.01 * losses['loss'] + 0.99 * smoothed_loss)

                progress_bar.set_postfix(loss='%.4f' % smoothed_loss)
                progress_bar.set_description("[{}/{}]".format(chunks, len(self.train_loader.sampler)))
                progress_bar.update()

                if loss_log is not None:
                    lr = lr_scheduler.get_last_lr() if lr_scheduler is not None else [pg["lr"] for pg in optim.param_groups]
                    if len(lr) == 1: lr = lr[0]
                    loss_log.append({
                        'chunks': chunks,
                        'time': perf_counter() - t0,
                        'grad_norm': grad_norm,
                        'lr': lr,
                        **losses
                    })

                if lr_scheduler is not None: lr_scheduler.step()

        return smoothed_loss, perf_counter() - t0
    

    def validate_one_step(self, batch):
        data, targets, lengths = batch

        scores = self.model_student(data.to(self.device))
        losses = self.criterion(scores.to(torch.float32), targets.to(self.device), lengths.to(self.device))
        losses = {k: v.item() for k, v in losses.items()} if isinstance(losses, dict) else losses.item()
        if hasattr(self.model_student, 'decode_batch'):
            seqs = self.model_student.decode_batch(scores)
        else:
            seqs = [self.model_student.decode(x) for x in permute(scores, 'TNC', 'NTC')]
        refs = [decode_ref(target, self.model_student.alphabet) for target in targets]
        accs = [
            accuracy(ref, seq, min_coverage=0.5) if len(seq) else 0. for ref, seq in zip(refs, seqs)
        ]
        return seqs, refs, accs, losses

    def validate_one_epoch(self):
        self.model_student.eval()
        with torch.no_grad():
            seqs, refs, accs, losses = zip(*(self.validate_one_step(batch) for batch in self.valid_loader))
        seqs, refs, accs = (sum(x, []) for x in (seqs, refs, accs))
        loss = np.mean([(x['ctc_loss'] if isinstance(x, dict) else x) for x in losses])
        return loss, np.mean(accs), np.median(accs)

    def init_optimizer(self, lr, **kwargs):
        self.optimizer = torch.optim.AdamW(self.model_student.parameters(), lr=lr, **kwargs)

    def get_lr_scheduler(self, epochs, last_epoch=0):
        return self.lr_scheduler_fn(self.optimizer, self.train_loader, epochs, last_epoch)

    def fit(self, workdir, epochs=1, lr=2e-3,remove_mask=False,quant=False,load_model_type="random"):
        print("TRAINING WITH MODEL AS::::",load_model_type)
        # if load_model_channel in ["small","smallnoskip"]:
            
        #     module_check=nn.Conv1d
        # else:
        #     module_check=qnn.QuantConv1d

        if quant:
            module_check=qnn.QuantConv1d
        else: 
            module_check=nn.Conv1d

        if load_model_type in ["rubiconskiptrim"]: 
            print("*****REMOVED SKIPS******")
            self.model_student.encoder.encoder[1].set_switch(True)
            self.model_student.encoder.encoder[2].set_switch(True)
            self.model_student.encoder.encoder[3].set_switch(True)
       
            self.model_student.encoder.encoder[4].set_switch(True)
            # print("******EPOCH={} REMOVED B3 SKIPS******".format(epoch))
            self.model_student.encoder.encoder[5].set_switch(True)
            self.model_student.encoder.encoder[6].set_switch(True)
            self.model_student.encoder.encoder[7].set_switch(True)
            # print("******EPOCH={} REMOVED B4 SKIPS******".format(epoch))
            self.model_student.encoder.encoder[8].set_switch(True)
            self.model_student.encoder.encoder[9].set_switch(True)
            self.model_student.encoder.encoder[10].set_switch(True)
            self.model_student.encoder.encoder[11].set_switch(True)
            # print("******EPOCH={} REMOVED B5 SKIPS******".format(epoch))
            self.model_student.encoder.encoder[12].set_switch(True)
            self.model_student.encoder.encoder[13].set_switch(True)
            self.model_student.encoder.encoder[14].set_switch(True)
            self.model_student.encoder.encoder[15].set_switch(True)

        if self.optimizer is None:
            self.init_optimizer(lr)

        last_epoch = load_state(workdir, self.device, self.model_student, self.optimizer if self.restore_optim else None)
        # override learning rate to new value
        for pg in self.optimizer.param_groups: pg["initial_lr"] = pg["lr"] = lr

        lr_scheduler = self.get_lr_scheduler(epochs, last_epoch=last_epoch)

        for epoch in range(1 + last_epoch, epochs + 1 + last_epoch):
            try:
                with CSVLogger(os.path.join(workdir, 'losses_{}.csv'.format(epoch))) as loss_log:
                    train_loss, duration = self.train_one_epoch(loss_log, lr_scheduler)
                    
                model_state = self.model_student.module.state_dict() if hasattr(self.model_student, 'module') else self.model_student.state_dict()
                
                # print("***************WHILE PRUNING****************")
                # print(dict(self.model_student.named_parameters()).keys())
                # print("*********PARAMETERS (len={})**********".format(len(dict(self.model_student.named_parameters()).keys())))
                # print(self.model_student.state_dict().keys())
                # print("GAGAN")
                # print("*********DICT (len={})**********".format(len(self.model_student.state_dict().keys())))

                torch.save(model_state, os.path.join(workdir, "weights_%s.tar" % epoch))

                if(remove_mask):
                    import subprocess
                    print("**************remove_mask*******************")
                    self.model_student=remove_prune_mask(self.model_student, module_check)
                    torch.save(self.model_student.state_dict(), os.path.join(workdir, "WITHOUT_MASK_WEIGHT_%s.tar" % epoch))
                    torch.save(self.model_student.state_dict(), os.path.join(workdir, "WITHOUT_MASK_WEIGHT_%s.h5" % epoch))
                    command="gzip -qf "+workdir+"/WITHOUT_MASK_WEIGHT_%s.h5" % epoch 
                    subprocess.getoutput(command)
                    command="du -h "+workdir+"/WITHOUT_MASK_WEIGHT_%s.h5.gz | awk '{print$1}'"% epoch 
                    print("command::::::",command)
                    print("WITHOUT_MASK_WEIGHT={}".format(subprocess.getoutput(command)))
                    self.model_student = load_model_prune_for_kd(workdir, self.device, quant=quant,load_model_type=load_model_type)
        

                if epoch % self.save_optim_every == 0:
                    torch.save(self.optimizer.state_dict(), os.path.join(workdir, "optim_%s.tar" % epoch))

                val_loss, val_mean, val_median = self.validate_one_epoch()

              
                    
                
                # num_zeros, num_elements, sparsity = measure_global_sparsity(
                # self.model_student,
                #             weight=True,
                #             bias=False,
                #             conv1d_use_mask=True)
         
                # print("Global Sparsity:")
                # print("{:.2f}".format(sparsity))
            except KeyboardInterrupt:
                break

            print("[epoch {}] directory={} loss={:.4f} mean_acc={:.3f}% median_acc={:.3f}%".format(
                epoch, workdir, val_loss, val_mean, val_median
            ))

            with CSVLogger(os.path.join(workdir, 'training.csv')) as training_log:
                training_log.append({
                    'time': datetime.today(),
                    'duration': int(duration),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'validation_loss': val_loss,
                    'validation_mean': val_mean,
                    'validation_median': val_median
                })
