# proxylessbasecalling.py -*- Python -*-
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

import bonito
from rubicon.tools.nni.nni.retiarii.oneshot.interface import BaseOneShotTrainer
from rubicon.tools.nni.nni.retiarii.oneshot.pytorch.utils import AverageMeterGroup, replace_layer_choice, replace_input_choice, to_device
from rubicon.tools.ont.bonito.schedule import linear_warmup_cosine_decay
from rubicon.tools.ont.bonito.util import accuracy, decode_ref, permute, concat, match_names

import rubicon

_logger = logging.getLogger(__name__)


class ArchGradientFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, binary_gates, run_func, backward_func):
        ctx.run_func = run_func
        ctx.backward_func = backward_func

        detached_x = x.detach()
        detached_x.requires_grad = x.requires_grad
        with torch.enable_grad():
            output = run_func(detached_x)
        ctx.save_for_backward(detached_x, output)
        return output.data

    @staticmethod
    def backward(ctx, grad_output):
        detached_x, output = ctx.saved_tensors

        grad_x = torch.autograd.grad(output, detached_x, grad_output, only_inputs=True)
        # compute gradients w.r.t. binary_gates
        binary_grads = ctx.backward_func(detached_x.data, output.data, grad_output.data)

        return grad_x[0], binary_grads, None, None


class ProxylessLayerChoice(nn.Module):
    def __init__(self, ops):
        super(ProxylessLayerChoice, self).__init__()
        self.ops = nn.ModuleList(ops)
        self.alpha = nn.Parameter(torch.randn(len(self.ops)) * 1E-3)
        self._binary_gates = nn.Parameter(torch.randn(len(self.ops)) * 1E-3)
        self.sampled = None

    def forward(self, *args):
        def run_function(ops, active_id):
            def forward(_x):
                return ops[active_id](_x)
            return forward

        def backward_function(ops, active_id, binary_gates):
            def backward(_x, _output, grad_output):
                binary_grads = torch.zeros_like(binary_gates.data)
                with torch.no_grad():
                    for k in range(len(ops)):
                        if k != active_id:
                            out_k = ops[k](_x.data)
                        else:
                            out_k = _output.data
                        grad_k = torch.sum(out_k * grad_output)
                        binary_grads[k] = grad_k
                return binary_grads
            return backward

        assert len(args) == 1
        x = args[0]
        return ArchGradientFunction.apply(
            x, self._binary_gates, run_function(self.ops, self.sampled),
            backward_function(self.ops, self.sampled, self._binary_gates)
        )

    def resample(self):
        probs = F.softmax(self.alpha, dim=-1)
        sample = torch.multinomial(probs, 1)[0].item()
        self.sampled = sample
        with torch.no_grad():
            self._binary_gates.zero_()
            self._binary_gates.grad = torch.zeros_like(self._binary_gates.data)
            self._binary_gates.data[sample] = 1.0

    def finalize_grad(self):
        binary_grads = self._binary_gates.grad
        with torch.no_grad():
            if self.alpha.grad is None:
                self.alpha.grad = torch.zeros_like(self.alpha.data)
            probs = F.softmax(self.alpha, dim=-1)
            for i in range(len(self.ops)):
                for j in range(len(self.ops)):
                    self.alpha.grad[i] += binary_grads[j] * probs[j] * (int(i == j) - probs[i])

    def export(self):
        return torch.argmax(self.alpha).item()

    def export_prob(self):
        return F.softmax(self.alpha, dim=-1)


class ProxylessInputChoice(nn.Module):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError('Input choice is not supported for ProxylessNAS.')


class HardwareLatencyEstimator:
    def __init__(self, applied_hardware, model, dummy_input=(1, 3, 224, 224), dump_lat_table='data/latency_table.yaml',quant=False):
        
        # from rubicon.tools.nn_Meter.nn_meter.predictor.nn_meter_predictor import load_latency_predictor   # pylint: disable=import-error
        import nn_meter 
        _logger.info(f'Load latency predictor for applied hardware: {applied_hardware}.')
        self.predictor_name = applied_hardware
       
        if(self.predictor_name=="aie_lut"):
            import yaml
            with open("data/latency_quant_aie.yaml", "r") as stream:
                try:
                    
                    self.block_latency_table=yaml.safe_load(stream)
                
                except yaml.YAMLError as exc:
                    _logger.warning(exc)
        else:
            self.latency_predictor = nn_meter.load_latency_predictor(applied_hardware)
            self.block_latency_table = self._form_latency_table(model, dummy_input, dump_lat_table=dump_lat_table)
    
     
        
      
    def _form_latency_table(self, model, dummy_input, dump_lat_table):
        latency_table = {}
        import sys
        from nni.retiarii.converter import convert_to_graph
        from nni.retiarii.converter.graph_gen import GraphConverterWithShape
        from nni.retiarii.converter.utils import flatten_model_graph_without_layerchoice, is_layerchoice_node
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
        cell_nodes = base_model_ir.get_cell_nodes()
        layerchoice_nodes = [node for node in cell_nodes if is_layerchoice_node(node)]
        for lc_node in layerchoice_nodes:
            cand_lat = {}
            for candidate in lc_node.operation.parameters['candidates']:
                node_graph = base_model_ir.graphs.get(candidate)
                if node_graph is not None:
                    temp_ir_model._root_graph_name = node_graph.name
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
            # print("[cal_expected_latency] probs:::",len(probs))
            # print("[cal_expected_latency] len(self.block_latency_table[module_name]):::",len(self.block_latency_table[module_name]))
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


class ProxylessBasecalling(BaseOneShotTrainer):
    """
    Proxyless trainer.

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
    warmup_epochs : int
        Number of epochs to warmup model parameters.
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
    grad_reg_loss_type: string
        Regularization type to add hardware related loss, allowed types include
        - ``"mul#log"``: ``regularized_loss = (torch.log(expected_latency) / math.log(self.ref_latency)) ** beta``
        - ``"add#linear"``: ``regularized_loss = reg_lambda * (expected_latency - self.ref_latency) / self.ref_latency``
        - None: do not apply loss regularization.
    grad_reg_loss_params: dict
        Regularization params, allowed params include
        - ``"alpha"`` and ``"beta"`` is required when ``grad_reg_loss_type == "mul#log"``
        - ``"lambda"`` is required when ``grad_reg_loss_type == "add#linear"``
    applied_hardware: string
        Applied hardware for to constraint the model's latency. Latency is predicted by Microsoft
        nn-Meter (https://github.com/microsoft/nn-Meter).
    dummy_input: tuple
        The dummy input shape when applied to the target hardware.
    ref_latency: float
        Reference latency value in the applied hardware (ms).
    """

    def __init__(self, model,  metrics,train_loader,valid_loader, 
                 num_epochs, optimizer=None,  grad_clip=5.,use_amp=True,criterion=None, warmup_epochs=0,
                 batch_size=64, workers=4, device=None, log_frequency=None,grad_accum_split=1,
                 restore_optim=False,lr_scheduler_fn=None,
                 opt_learning_rate=2.5E-3,
                 ctrl_learning_rate=1.0E-3,
                 rubicon=None,
                 grad_reg_loss_type=None, grad_reg_loss_params=None,
                 applied_hardware=None,  dummy_input=(344, 1, 9),
                 ref_latency=65.0,unrolled=False,opt_typ='proxy', quant=False,
                 rub_sched=False,
                dart_sched=False,
                rub_ctrl_opt=False,
                prox_ctrl_opt=False,
                default=False
                ):
        _logger.info("Initializing ProxylessNAS")
        self.model = model
        # self.loss = loss
        self.metrics = metrics
        self.use_amp = use_amp
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion or (model.seqdist.ctc_loss if hasattr(model, 'seqdist') else model.ctc_label_smoothing_loss) ########3
        self.default=default
        if(rub_ctrl_opt): # we only set STEP 1 here
            if (self.default):
                self.ctrl_learning_rate=2.0E-3
            else:
                self.ctrl_learning_rate=ctrl_learning_rate
        if(prox_ctrl_opt):
            if (self.default):
                self.ctrl_learning_rate=2.0E-3
            else:
                self.ctrl_learning_rate=ctrl_learning_rate

        # self.criterion =  model.ctc_label_smoothing_loss
        self.rub_sched=rub_sched
        self.dart_sched=dart_sched
        if(self.rub_sched):
            self.lr_scheduler_fn = linear_warmup_cosine_decay()
        if(self.dart_sched):
            self.lr_scheduler_fn = None

        self.restore_optim = restore_optim
        self.batch_size = batch_size
        self.grad_accum_split = grad_accum_split
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.scaler_ctrl = torch.cuda.amp.GradScaler(enabled=use_amp)
        self.workers = workers
        self.optimizer = optimizer
        self.quant=quant
        self.num_epochs = num_epochs
        self.warmup_epochs = warmup_epochs
        if(self.warmup_epochs>0):
            _logger.info("*****WARMING ENABLED for={} epoch***".format(self.warmup_epochs))
        # self.dataset = dataset
        self.batch_size = batch_size
        self.workers = workers
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self.log_frequency = log_frequency
        self.rubicon=rubicon
        self.default=default

        
        # latency predictor
        if applied_hardware:
            self.latency_estimator = HardwareLatencyEstimator(applied_hardware, self.model, dummy_input,quant=self.quant)
        else:
            self.latency_estimator = None
        self.reg_loss_type = grad_reg_loss_type
        self.reg_loss_params = {} if grad_reg_loss_params is None else grad_reg_loss_params
        reg_lambda = self.reg_loss_params.get('lambda', 2e-1)
       
        self.ref_latency = ref_latency
        
        self.model.to(self.device)
        self.nas_modules = []
        replace_layer_choice(self.model, ProxylessLayerChoice, self.nas_modules)
        replace_input_choice(self.model, ProxylessInputChoice, self.nas_modules)
        for _, module in self.nas_modules:
            module.to(self.device)

        # print("dumping nas modules:::",self.nas_modules)
        # self.model_optim = None
        # self.model_optim = optimizer
        # self.optimizer = optimizer
        # we do not support deduplicate control parameters with same label (like DARTS) yet.
        if(rub_ctrl_opt):
              
            self.ctrl_optim = torch.optim.AdamW([m.alpha for _, m in self.nas_modules], self.ctrl_learning_rate,
                                    weight_decay=0, betas=(0, 0.999), eps=1e-8)
        if(prox_ctrl_opt):
            self.ctrl_optim = torch.optim.Adam([m.alpha for _, m in self.nas_modules], self.ctrl_learning_rate,
                                    weight_decay=0, betas=(0, 0.999), eps=1e-8)
       
        _logger.info("[Step 1] CTRL Optimizer Learning Rate=%s"%self.ctrl_learning_rate)
        _logger.info("[Step 1] CTRL Optimizer=%s"%self.ctrl_optim)
        
        # self.ctrl_optim = torch.optim.Adam([m.alpha for _, m in self.nas_modules], arc_learning_rate, 
        #                             betas=(0.1, 0.999), weight_decay=1.0E-4) ##darts
        # self._init_dataloader()
        self.unrolled = unrolled
        self.grad_clip = 5.
        
    def init_optimizer(self, lr, **kwargs):
        # self.ctrl_optim = torch.optim.AdamW(self.model.parameters(), lr=lr, **kwargs)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, **kwargs)


    def get_lr_scheduler(self, epochs, last_epoch=0):
        if(self.rub_sched):
            return self.lr_scheduler_fn(self.optimizer, self.train_loader, epochs, last_epoch)   
        elif(self.dart_sched):
            return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, epochs, eta_min=0.001)  
        else:
            return None

    # def _init_dataloader(self):
    #     n_train = len(self.dataset)
    #     split = n_train // 2
    #     indices = list(range(n_train))
    #     train_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[:split])
    #     valid_sampler = torch.utils.data.sampler.SubsetRandomSampler(indices[split:])
    #     self.train_loader = torch.utils.data.DataLoader(self.dataset,
    #                                                     batch_size=self.batch_size,
    #                                                     sampler=train_sampler,
    #                                                     num_workers=self.workers)
    #     self.valid_loader = torch.utils.data.DataLoader(self.dataset,
    #                                                     batch_size=self.batch_size,
    #                                                     sampler=valid_sampler,
    #                                                     num_workers=self.workers)
    
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
               
                
                # self.scaler_ctrl.scale(losses_['loss']).backward() #calculate the gradietns
                losses = {
                        k: (v.item() if losses is None else v.item() + losses[k])
                        for k, v in losses_.items()
                    }
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
                        reg_loss=0
                        losses_['loss']=losses_['loss']
                    else:
                        raise ValueError(f'Do not support: {self.reg_loss_type}')
                
                losses_['loss'].backward() #calculate the gradietns <----------------------- GAGAN outside
        
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0).item()
        for _, module in self.nas_modules:
            module.finalize_grad()
        self.ctrl_optim.step() #update the weigths

        # self.scaler_ctrl.unscale_(self.ctrl_optim)
        
        # self.scaler_ctrl.step(self.ctrl_optim)
        # self.scaler_ctrl.update()

        return reg_loss
             #try removing below
        # grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0).item()

            

         
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
                
                # self.scaler.scale(losses_['loss']).backward()
                losses_['loss'].backward()

                losses = {
                    k: (v.item() if losses is None else v.item() + losses[k])
                    for k, v in losses_.items()
                }

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0).item()
        self.optimizer.step() #update the weigths

        # self.scaler.unscale_(self.optimizer)
        # grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=2.0).item()
        # self.scaler.step(self.optimizer)
        # self.scaler.update()

        return losses
       
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
        reg_loss=0
        with progress_bar:
            # for batch_train, batch_valid in self.train_loader:
            for  i, (batch_train, batch_valid) in enumerate(zip(self.train_loader, self.valid_loader)):
                    # print("step::",i)
                    # print("batch_train={}, batch_valid={}".format(batch_train[0].shape[0],batch_valid[0].shape[0]))
               
                    chunks += batch_train[0].shape[0]
                    chunks_valid += batch_valid[0].shape[0]

                    if epoch >= self.warmup_epochs:
                        # phase 1. architecture step
                        for _, module in self.nas_modules:
                            module.resample()
                        self.ctrl_optim.zero_grad()
                        reg_loss=self.train_one_step_arch_update(batch_valid,self.ctrl_optim)


                    # phase 2: child network step
                    for _, module in self.nas_modules:
                        module.resample()
                    self.optimizer.zero_grad()
                    losses = self.train_one_step_weight_update(batch_train,self.optimizer)
                    smoothed_loss = losses['loss'] if smoothed_loss is None else (0.01 * losses['loss'] + 0.99 * smoothed_loss)
                    
                    
                    latency_arch=0          
                    if self.latency_estimator:
                        latency_arch = self._export_latency()
                    progress_bar.set_postfix(loss='%.4f' % losses['loss'], loss_hw='%.4f' %reg_loss , latency='%.4f' % latency_arch)
                    progress_bar.set_description("Epoch {}[{}/{}]".format(epoch, chunks, len(self.train_loader.sampler)))
                    progress_bar.update()
                    

                    if loss_log is not None:
                        if lr_scheduler is not None: 
                            lr = lr_scheduler.get_last_lr() 
                    #     if len(lr) == 1: lr = lr[0]
                    #     loss_log.append({
                    #         'chunks': chunks,
                    #         'time': perf_counter() - t0,
                    #         'grad_norm': grad_norm,
                    #         'lr': lr,
                    #         **losses
                    #     })
                    if(self.dart_sched ):
                        if lr_scheduler is not None: lr_scheduler.step()
                    if(self.rub_sched ):
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
        if self.optimizer is None:
            self.init_optimizer(lr)
        _logger.info("[Step 2] ARCH Optimizer Learning Rate=%s"%lr)
        _logger.info("[Step 2] ARCH Optimizer=%s"%self.optimizer)      
        

        lr_scheduler = self.get_lr_scheduler(epochs)

        for epoch in range(self.num_epochs):
            try:
                with bonito.io.CSVLogger(os.path.join(workdir, 'losses_{}.csv'.format(epoch))) as loss_log:
                    train_loss, duration = self.train_one_epoch(epoch,loss_log, lr_scheduler)
                    
                    _logger.info('Epoch {} Architecture:{}' .format(epoch,self.export()))
                    _logger.info("*************************************************************************")
                # model_state = self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict()
                # torch.save(model_state, os.path.join(workdir, "weights_%s.tar" % epoch))
                # if epoch % self.save_optim_every == 0:
                #     torch.save(self.optimizer.state_dict(), os.path.join(workdir, "optim_%s.tar" % epoch))
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

            with bonito.io.CSVLogger(os.path.join(workdir, 'training.csv')) as training_log:
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
