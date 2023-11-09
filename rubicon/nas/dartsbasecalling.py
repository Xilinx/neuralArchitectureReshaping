# dartsbasecalling.py -*- Python -*-
#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2023 Advanced Micro Devices, Inc.

import os
import re
from glob import glob
from functools import partial
from time import perf_counter
from collections import OrderedDict
from datetime import datetime
from tqdm import tqdm
import torch.cuda.amp as amp
import numpy as np
import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from bonito.schedule import linear_warmup_cosine_decay
from bonito.util import accuracy, decode_ref, permute, concat, match_names
import rubicon
import sys
import os

from pathlib import Path

# cwd=Path(__file__).parent.parent 
# nni_rubicon = os.path.join(cwd, "tools/nni")
# print(nni_rubicon)
# sys.path.insert(0, nni_rubicon)
# print(sys.path)
from rubicon.tools.nni.nni.retiarii.oneshot.interface import BaseOneShotTrainer
from rubicon.tools.nni.nni.retiarii.oneshot.pytorch.utils import AverageMeterGroup, replace_layer_choice, replace_input_choice


_logger = logging.getLogger(__name__)


class DartsLayerChoice(nn.Module):
    def __init__(self, layer_choice):
        # print(layer_choice)
        super(DartsLayerChoice, self).__init__()
        self.name = layer_choice.label
        self.op_choices = nn.ModuleDict(OrderedDict([(name, layer_choice[name]) for name in layer_choice.names]))
        self.alpha = nn.Parameter(torch.randn(len(self.op_choices)) * 1e-3)

    def forward(self, *args, **kwargs):
        op_results = torch.stack([op(*args, **kwargs) for op in self.op_choices.values()])
        alpha_shape = [-1] + [1] * (len(op_results.size()) - 1)
        return torch.sum(op_results * F.softmax(self.alpha, -1).view(*alpha_shape), 0)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super(DartsLayerChoice, self).named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return list(self.op_choices.keys())[torch.argmax(self.alpha).item()]

    
    def export_prob(self):
        return F.softmax(self.alpha, dim=-1)

class DartsInputChoice(nn.Module):
    def __init__(self, input_choice):
        super(DartsInputChoice, self).__init__()
        self.name = input_choice.label
        self.alpha = nn.Parameter(torch.randn(input_choice.n_candidates) * 1e-3)
        self.n_chosen = input_choice.n_chosen or 1

    def forward(self, inputs):
        inputs = torch.stack(inputs)
        alpha_shape = [-1] + [1] * (len(inputs.size()) - 1)
        return torch.sum(inputs * F.softmax(self.alpha, -1).view(*alpha_shape), 0)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self):
        for name, p in super(DartsInputChoice, self).named_parameters():
            if name == 'alpha':
                continue
            yield name, p

    def export(self):
        return torch.argsort(-self.alpha).cpu().numpy().tolist()[:self.n_chosen]
class HardwareLatencyEstimator:
    def __init__(self, applied_hardware, model, dummy_input=(1, 3, 224, 224), dump_lat_table='data/latency.yaml',quant=False):
        from rubicon.tools.nn_Meter.nn_meter.predictor.nn_meter_predictor import load_latency_predictor   # pylint: disable=import-error
        _logger.info(f'Load latency predictor for applied hardware: {applied_hardware}.')
        self.predictor_name = applied_hardware
       
        if(self.predictor_name=="aie_lut"):
            import inspect
            install_path=os.path.dirname(inspect.getfile(rubicon))
            import yaml
            with open(install_path+"/data/latency_quant_aie.yaml", "r") as stream:
                try:
                    
                    self.block_latency_table=yaml.safe_load(stream)
               
                except yaml.YAMLError as exc:
                    _logger.info(exc)
        else:
            self.latency_predictor = nn_meter.load_latency_predictor(applied_hardware)
            self.block_latency_table = self._form_latency_table(model, dummy_input, dump_lat_table=dump_lat_table)
    
     
        
      
    def _form_latency_table(self, model, dummy_input, dump_lat_table):
        latency_table = {}
        import sys

        from rubicon.tools.nni.nni.retiarii.converter import convert_to_graph
        from rubicon.tools.nni.nni.retiarii.converter.graph_gen import GraphConverterWithShape
        from rubicon.tools.nni.nni.retiarii.converter.graph_gen.utils import flatten_model_graph_without_layerchoice, is_layerchoice_node
        script_module = torch.jit.script(model)
        # rand_inp = torch.randn(344, 1, 9)
        # script_module = torch.jit.trace(model,rand_inp)
        original_stdout = sys.stdout # Save a reference to the original standard output

        with open('script_def.txt', 'w') as f:
            sys.stdout = f # Change the standard output to the file we created.
            # print(script_module)
            sys.stdout = original_stdout # Reset the standard output to its original value
        
        with open('model_def.txt', 'w') as f:
            sys.stdout = f # Change the standard output to the file we created.
            # print(model)
            sys.stdout = original_stdout # Reset the standard output to its original value
        
        base_model_ir = convert_to_graph(script_module, model,
                                         converter=GraphConverterWithShape(), dummy_input=torch.randn(*dummy_input))

        # form the latency of layerchoice blocks for the latency table
        temp_ir_model = base_model_ir.fork()
        # print("********************************************************************/n")
        # print("temp_ir_model",temp_ir_model)
        # print("********************************************************************/n")
        cell_nodes = base_model_ir.get_cell_nodes()
        layerchoice_nodes = [node for node in cell_nodes if is_layerchoice_node(node)]
        for lc_node in layerchoice_nodes:
            cand_lat = {}
            for candidate in lc_node.operation.parameters['candidates']:
                node_graph = base_model_ir.graphs.get(candidate)
                if node_graph is not None:
                    temp_ir_model._root_graph_name = node_graph.name
                    # print("********************************************************************/n")
                    # print("temp_ir_model",temp_ir_model)
                    # print("********************************************************************/n")
                    latency = self.latency_predictor.predict(temp_ir_model, model_type = 'nni-ir')
                else:
                    _logger.warning(f"Could not found graph for layerchoice candidate {candidate}")
                    latency = 0
                cand_lat[candidate.split('_')[-1]] = float(latency)
            
            latency_table[lc_node.operation.parameters['label']] = cand_lat
            # print(lc_node.operation.parameters['label'])
            # print(cand_lat)

        # form the latency of the stationary block in the latency table
        temp_ir_model._root_graph_name = base_model_ir._root_graph_name
        temp_ir_model = flatten_model_graph_without_layerchoice(temp_ir_model)
        latency = self.latency_predictor.predict(temp_ir_model, model_type = 'nni-ir')
        latency_table['stationary_block'] = {'root': float(latency)}

        # save latency table
        if dump_lat_table:
            import os, yaml
            os.makedirs(os.path.dirname(dump_lat_table), exist_ok=True)
            with open(dump_lat_table, 'a') as fp:
                yaml.dump([{
                    "applied_hardware": self.predictor_name,
                    'latency_table': latency_table
                    }], fp)
        _logger.info("Latency lookup table form done")
        # print(latency_table)
        return latency_table

    def cal_expected_latency(self, current_architecture_prob):
        lat = self.block_latency_table['stationary_block']['root']
        for module_name, probs in current_architecture_prob.items():
            # print("[cal_expected_latency] probs:::",probs)
            # print("[cal_expected_latency] len probs:::",len(probs))
            # print("[cal_expected_latency] self.block_latency_table[module_name]:::",self.block_latency_table[module_name])
            # print("[cal_expected_latency] len self.block_latency_table[module_name]:::",len(self.block_latency_table[module_name]))
            # print("[proxylessnas]module_name:::",module_name)
            assert len(probs) == len(self.block_latency_table[module_name])
            lat += torch.sum(torch.tensor([probs[i] * self.block_latency_table[module_name][str(i)]
                                for i in range(len(probs))]))
            # print("[proxylessnas]lat:::",lat)
        return lat

    def export_latency(self, current_architecture):
        lat = self.block_latency_table['stationary_block']['root']
        for module_name, selected_module in current_architecture.items():
            lat += self.block_latency_table[module_name][str(selected_module)]
        return lat


class DartsBasecalling(BaseOneShotTrainer):
    """
    DARTS trainer.

    Parameters
    ----------
    model : nn.Module
        PyTorch model to be trained.
    loss : callable
        Receives logits and ground truth label, return a loss tensor.
    metrics : callable
        Receives logits and ground truth label, return a dict of metrics.
    optimizer : Optimizer
        The optimizer used for optimizing the model.
    num_epochs : int
        Number of epochs planned for training.
    dataset : Dataset
        Dataset for training. Will be split for training weights and architecture weights.
    grad_clip : float
        Gradient clipping. Set to 0 to disable. Default: 5.
    learning_rate : float
        Learning rate to optimize the model.
    batch_size : int
        Batch size.
    workers : int
        Workers for data loading.
    device : torch.device
        ``torch.device("cpu")`` or ``torch.device("cuda")``.
    log_frequency : int
        Step count per logging.
    arc_learning_rate : float
        Learning rate of architecture parameters.
    unrolled : float
        ``True`` if using second order optimization, else first order optimization.
    """

    def __init__(self, model,  metrics, train_loader,valid_loader,
                 num_epochs, optimizer=None, grad_clip=5.,use_amp=True,criterion=None,
                 restore_optim=False,lr_scheduler_fn=None,
                 learning_rate=2.5E-3, batch_size=64, workers=4,
                 device=None, log_frequency=None,grad_accum_split=1,
                 opt_learning_rate=2.5E-3,
                 ctrl_learning_rate=1.0E-3,
                 rubicon=None,default=None,
                  grad_reg_loss_type=None, grad_reg_loss_params=None,
                 applied_hardware=None,  dummy_input=(344, 1, 9),
                 ref_latency=65.0,
                  unrolled=False, quant=False):
        _logger.info("Initializing DARTS")
        self.model = model
        self.quant=quant
        self.metrics = metrics
        self.num_epochs = num_epochs
        self.use_amp = use_amp
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion or (model.seqdist.ctc_loss if hasattr(model, 'seqdist') else model.ctc_label_smoothing_loss)
        # self.criterion =  model.ctc_label_smoothing_loss

        self.lr_scheduler_fn = lr_scheduler_fn or linear_warmup_cosine_decay()
        self.restore_optim = restore_optim
        self.batch_size = batch_size
        self.grad_accum_split = grad_accum_split
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.workers = workers
   
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.log_frequency = log_frequency

        # latency predictor
        if applied_hardware:
            self.latency_estimator = HardwareLatencyEstimator(applied_hardware, self.model, dummy_input,quant=self.quant)
        else:
            self.latency_estimator = None
        self.reg_loss_type = grad_reg_loss_type
        self.reg_loss_params = {} if grad_reg_loss_params is None else grad_reg_loss_params
        self.ref_latency = ref_latency
        _logger.info("reference latency%s" %self.ref_latency)
        self.model.to(self.device)

        self.nas_modules = []
        replace_layer_choice(self.model, DartsLayerChoice, self.nas_modules)
        replace_input_choice(self.model, DartsInputChoice, self.nas_modules)
        for _, module in self.nas_modules:
            module.to(self.device)
        # print("nas moduels",self.nas_modules)
        self.model_optim=optimizer
        # use the same architecture weight for modules with duplicated names
        ctrl_params = {}
        for _, m in self.nas_modules:
            # print(m.name)
            if m.name in ctrl_params:
                assert m.alpha.size() == ctrl_params[m.name].size(), 'Size of parameters with the same label should be same.'
                m.alpha = ctrl_params[m.name]
            else:
                ctrl_params[m.name] = m.alpha
        # self.ctrl_optim = None
        self.rubicon=rubicon
        # if(rubicon):
        #     self.ctrl_optim = torch.optim.AdamW(list(ctrl_params.values()), arc_learning_rate, betas=(0.1, 0.999),
        #                                     weight_decay=1.0E-4)     
        # else:
        self.default=default
        
        if (self.default):
            self.ctrl_learning_rate=3.0E-4
        else:
            self.ctrl_learning_rate=ctrl_learning_rate
        self.ctrl_optim = torch.optim.Adam(list(ctrl_params.values()), self.ctrl_learning_rate, betas=(0.5, 0.999),
                                           weight_decay=1.0E-3)
            
        self.unrolled = unrolled
        self.grad_clip = 5.

        # self._init_dataloader()
    
    def init_optimizer(self, lr, **kwargs):
        # self.ctrl_optim = torch.optim.AdamW(self.model.parameters(), lr=lr, **kwargs)
        self.model_optim = torch.optim.AdamW(self.model.parameters(), lr=lr, **kwargs)
        _logger.info("Optimizer Learning rate=%s"%lr)
        _logger.info("\n****************************************************\n")

    def get_lr_scheduler(self, epochs, last_epoch=0):
        if(self.rubicon):
            return self.lr_scheduler_fn(self.model_optim, self.train_loader, epochs, last_epoch)   
        else:
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.model_optim, epochs, eta_min=0.001)  
           
    def get_lr_ctrl_scheduler(self, epochs, last_epoch=0):
        if(self.rubicon):
            return self.lr_scheduler_fn(self.ctrl_optim, self.valid_loader, epochs, last_epoch)
             
        else:
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.ctrl_optim, epochs, eta_min=0.001)  
           
     
    
    def validate_one_step(self, batch):
        data, targets, lengths = batch

        scores = self.model(data.to(self.device))
        losses = self.criterion(scores.to(torch.float32), targets.to(self.device), lengths.to(self.device))
        losses = {k: v.item() for k, v in losses.items()} if isinstance(losses, dict) else losses.item()
        if hasattr(self.model, 'decode_batch'):
            seqs = self.model.decode_batch(scores)
        else:
            seqs = [self.model.decode(x) for x in permute(scores, 'TNC', 'NTC')]
        refs = [decode_ref(target, self.model.alphabet) for target in targets]
        accs = [
            accuracy(ref, seq, min_coverage=0.5) if len(seq) else 0. for ref, seq in zip(refs, seqs)
        ]
        return seqs, refs, accs, losses

    def validate_one_epoch(self):
        self.model.eval()
        with torch.no_grad():
            seqs, refs, accs, losses = zip(*(self.validate_one_step(batch) for batch in self.valid_loader))
        seqs, refs, accs = (sum(x, []) for x in (seqs, refs, accs))
        loss = np.mean([(x['ctc_loss'] if isinstance(x, dict) else x) for x in losses])
        return loss, np.mean(accs), np.median(accs)
    
    def train_one_step_weight_update(self, batch,optimizer):
       
        # print("batch size", batch)
        losses_ = None
        losses=None
    
        for data_, targets_, lengths_ in zip(*map(lambda t: t.chunk(self.grad_accum_split, dim=0), batch)):
            data_, targets_, lengths_ = data_.to(self.device), targets_.to(self.device), lengths_.to(self.device)
            # print(data_.size())
            scores_ = self.model(data_) #forward pass, get logits
            losses_ = self.criterion(scores_.to(torch.float32), targets_, lengths_)  #find the loss

            if not isinstance(losses_, dict): losses_ = {'loss': losses_}

            losses_['loss'] = losses_['loss'] / self.grad_accum_split

            # self.scaler.scale(losses_['loss']).backward()
            losses_['loss'].backward()

            losses = {
                k: (v.item() if losses is None else v.item() + losses[k])
                for k, v in losses_.items()
            }

        # # self.scaler.unscale_(optimizer)
        # grad_norm=0
        # if self.grad_clip > 0:
        #         grad_norm=nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)  # gradient clipping
            
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0).item()
        # self.scaler.step(optimizer)
        # self.scaler.update()
        self.model_optim.step() #update the weigths
        return losses

    def train_one_step_phase1(self, batch,optimizer):
        optimizer.zero_grad() #clear the gradients
        # print("batch size", batch)
        losses = None
    
        for data_, targets_, lengths_ in zip(*map(lambda t: t.chunk(self.grad_accum_split, dim=0), batch)):
            data_, targets_, lengths_ = data_.to(self.device), targets_.to(self.device), lengths_.to(self.device)
            # print(data_.size())
            scores_ = self.model(data_) #forward pass, get logits
            losses_ = self.criterion(scores_.to(torch.float32), targets_, lengths_) #find the loss

            if not isinstance(losses_, dict): losses_ = {'loss': losses_}

            losses_['loss'] = losses_['loss'] / self.grad_accum_split
    
            losses_['loss'].backward() #calculate the gradietns

         
        optimizer.step() #update the weigths
       
    def validate_one_step(self, batch):
        data, targets, lengths = batch

        scores = self.model(data.to(self.device))
        losses = self.criterion(scores.to(torch.float32), targets.to(self.device), lengths.to(self.device))
        losses = {k: v.item() for k, v in losses.items()} if isinstance(losses, dict) else losses.item()
        if hasattr(self.model, 'decode_batch'):
            seqs = self.model.decode_batch(scores)
        else:
            seqs = [self.model.decode(x) for x in permute(scores, 'TNC', 'NTC')]
        refs = [decode_ref(target, self.model.alphabet) for target in targets]
        accs = [
            accuracy(ref, seq, min_coverage=0.5) if len(seq) else 0. for ref, seq in zip(refs, seqs)
        ]
        return seqs, refs, accs, losses

    def validate_one_epoch(self):
        self.model.eval()
        with torch.no_grad():
            seqs, refs, accs, losses = zip(*(self.validate_one_step(batch) for batch in self.valid_loader))
        seqs, refs, accs = (sum(x, []) for x in (seqs, refs, accs))
        loss = np.mean([(x['ctc_loss'] if isinstance(x, dict) else x) for x in losses])
        return loss, np.mean(accs), np.median(accs)
    
    def train_one_step_arch_update(self, batch,optimizer):
           # optimizer.zero_grad() #clear the gradients
        # print("batch size", batch)
        losses_ = None
        losses=None
        
        for data_, targets_, lengths_ in zip(*map(lambda t: t.chunk(self.grad_accum_split, dim=0), batch)):
                data_, targets_, lengths_ = data_.to(self.device), targets_.to(self.device), lengths_.to(self.device)
                # print("one step size",data_.size())
                scores_ = self.model(data_) #forward pass, get logits
                losses_ = self.criterion(scores_.to(torch.float32), targets_, lengths_) #find the loss

                if not isinstance(losses_, dict): losses_ = {'loss': losses_}

                losses_['loss'] = losses_['loss'] / self.grad_accum_split
                # print("losses::::",losses_['loss'])
                if self.latency_estimator:
                    current_architecture_prob = {}
                    for module_name, module in self.nas_modules:
                        probs = module.export_prob()
                        # print("[train_one_step_arch_update] module_name:{}/{}".format(module_name, probs))
                        # print("[train_one_step_arch_update] probs::",probs)
                        current_architecture_prob[module_name] = probs
                    
                    expected_latency = self.latency_estimator.cal_expected_latency(current_architecture_prob)
                    
                    # print("[train_one_step_arch_update]::::expected_latency", expected_latency)
                    if self.reg_loss_type == 'mul#log':
                        import math
                        alpha = self.reg_loss_params.get('alpha', 1)
                        beta = self.reg_loss_params.get('beta', 0.6)
                        # noinspection PyUnresolvedReferences
                        reg_loss = (torch.log(expected_latency) / math.log(self.ref_latency)) ** beta
                        losses_['loss']= alpha * losses_['loss'] * reg_loss
                        # return logits, alpha * ce_loss * reg_loss
                    elif self.reg_loss_type == 'add#linear':
                        reg_lambda = self.reg_loss_params.get('lambda', 2e-1)
                        # print("reg_lambda::",reg_lambda)
                        # print("[train_one_step_arch_update]::::reg_lambda", reg_lambda)
                        reg_loss = reg_lambda * (expected_latency - self.ref_latency) / self.ref_latency
                        # print("[train_one_step_arch_update]::::reg_loss", reg_loss)
                        losses_['loss']= losses_['loss'] + reg_loss
                    elif self.reg_loss_type is None:
                        losses_['loss']=losses_['loss']
                    else:
                        raise ValueError(f'Do not support: {self.reg_loss_type}')
                
                # self.scaler_ctrl.scale(losses_['loss']).backward() #calculate the gradietns
                losses = {
                        k: (v.item() if losses is None else v.item() + losses[k])
                        for k, v in losses_.items()
                    }
                losses_['loss'].backward() #calculate the gradietns
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0).item()
        self.ctrl_optim.step() #update the weigths

        # self.scaler_ctrl.unscale_(self.ctrl_optim)
        # grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0).item()
        # self.scaler_ctrl.step(self.ctrl_optim)
        # self.scaler_ctrl.update()

        return reg_loss
             #try removing below
        # grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0).item()

    def train_one_epoch(self, epoch, loss_log, lr_scheduler):
        t0 = perf_counter()
        chunks = 0
        chunks_valid=0
        
        self.model.train()
        meters = AverageMeterGroup()
        progress_bar = tqdm(
            total=len(self.train_loader), desc='[0/{}]'.format(len(self.train_loader.sampler)),
            ascii=True, leave=True, ncols=100, bar_format='{l_bar}{bar}| [{elapsed}{postfix}]'
        )
        smoothed_loss = None
        with progress_bar:
            # for batch_train, batch_valid in self.train_loader:
            for  i, (batch_train, batch_valid) in enumerate(zip(self.train_loader, self.valid_loader)):
                    # print("step::",i)
                    # print("batch_train={}, batch_valid={}".format(batch_train.shape,batch_valid.shape))
               
                    chunks += batch_train[0].shape[0]
                    chunks_valid += batch_valid[0].shape[0]
                    
                    # phase 1. architecture step
                    self.ctrl_optim.zero_grad()
                    reg_loss=self.train_one_step_arch_update(batch_valid,self.ctrl_optim)
                    
                    # phase 2: child network step
                    self.model_optim.zero_grad()
                    losses = self.train_one_step_weight_update(batch_train,self.model_optim)
                    # smoothed_loss = losses['loss'] if smoothed_loss is None else (0.01 * losses['loss'] + 0.99 * smoothed_loss)

                    latency_arch=0          
                    if self.latency_estimator:
                        latency_arch = self._export_latency()
                    progress_bar.set_postfix(loss='%.4f' % losses['loss'], loss_hw='%.4f' %reg_loss , latency='%.4f' % latency_arch)
                    progress_bar.set_description("Epoch {}[{}/{}]".format(epoch,chunks, len(self.train_loader.sampler)))
                    progress_bar.update()

                    if loss_log is not None:
                        lr = lr_scheduler.get_last_lr() if lr_scheduler is not None else [pg["lr"] for pg in optim.param_groups]
                        # if len(lr) == 1: lr = lr[0]
                        # loss_log.append({
                        #     'chunks': chunks,
                        #     'time': perf_counter() - t0,
                        #     'grad_norm': grad_norm,
                        #     'lr': lr,
                        #     **losses
                        # })
                    if(self.rubicon):
                        # if lr_ctrl_scheduler is not None: lr_ctrl_scheduler.step()
                        if lr_scheduler is not None: lr_scheduler.step()
            _logger.info('Epoch {} latency: {}'. format(epoch, latency_arch))
            _logger.info('Epoch {} loss: {}'. format(epoch, losses['loss']))
            _logger.info('Epoch {} loss HW: {}'. format(epoch, reg_loss))
            return losses['loss'], perf_counter() - t0

    def _export_latency(self):
        current_architecture = {}
        for module_name, module in self.nas_modules:
            selected_module = module.export()
            current_architecture[module_name] = selected_module
        return self.latency_estimator.export_latency(current_architecture)
    def fit(self, workdir, epochs=1, lr=2e-3):
        if self.model_optim is None:
            _logger.info("getting rubicon optim")
            self.init_optimizer(lr)

  
        # override learning rate to new value
        # for pg in self.optimizer.param_groups: pg["initial_lr"] = pg["lr"] = lr

        # for pg in self.ctrl_optim.param_groups: pg["initial_lr"] = pg["lr"] = lr

        lr_scheduler = self.get_lr_scheduler(epochs)
        # lr_ctrl_scheduler = self.get_lr_ctrl_scheduler(epochs, last_epoch=last_epoch)
        # lr_scheduler=None
        for epoch in range(self.num_epochs):
            try:
                with rubicon.io.CSVLogger(os.path.join(workdir, 'losses_{}.csv'.format(epoch))) as loss_log:
                    train_loss, duration = self.train_one_epoch(epoch, loss_log, lr_scheduler)
                    
                    _logger.info('Epoch {} Final architecture:{}' .format(epoch,self.export()))
                    _logger.info("*************************************************************************")
                # model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
                # torch.save(model_state, os.path.join(workdir, "weights_%s.tar" % epoch))
                # if epoch % self.save_optim_every == 0:
                # #     torch.save(self.optimizer.state_dict(), os.path.join(workdir, "optim_%s.tar" % epoch))
                # print("validation part")
                val_loss=0 
                val_mean=0
                val_median=0
                # print("Total parameters in model", sum(p.numel() for p in self.model.parameters()))
       
                # val_loss, val_mean, val_median = self.validate_one_epoch()
            except KeyboardInterrupt:
                break

            # print("[epoch {}] directory={} loss={:.4f} mean_acc={:.3f}% median_acc={:.3f}%".format(
            #     epoch, workdir, val_loss, val_mean, val_median
            # ))

            with rubicon.io.CSVLogger(os.path.join(workdir, 'training.csv')) as training_log:
                training_log.append({
                    'time': datetime.today(),
                    'duration': int(duration),
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'validation_loss': val_loss,
                    'validation_mean': val_mean,
                    'validation_median': val_median
                })


    @torch.no_grad()
    def export(self):
        result = dict()
        for name, module in self.nas_modules:
            if name not in result:
                result[name] = module.export()
        return result