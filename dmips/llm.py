# llm.py -*- Python -*-
#
# This file is licensed under the MIT License.
# SPDX-License-Identifier: MIT
# 
# (c) Copyright 2023 Advanced Micro Devices, Inc.

import sys
import argparse
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import evaluate

import opt
# Use MS MXFP

TAU = 2.0

@dataclass
class ModelArguments(opt.ModelArguments):
    w_dtypes: Optional[List[str]] = field(
        default_factory=lambda: ["fp32"],
        metadata={
            "help": (
                "Data types for model weights."
            )
        }
    )

    w_dense: Optional[List[int]] = field(
        default_factory=lambda: [16],
        metadata={
            "help": (
                "Densities for weight pruning (0 < density <= blocksize)."
                "If dense == blocksize, no pruning."
            )
        }
    )

    blocksize: Optional[int] = field(
        default=16,
        metadata={
            "help": (
                "Block size for weight pruning (0 < density <= blocksize)."
                "If dense == blocksize, no pruning."
            )
        }
    )

    a_dtypes: Optional[List[str]] = field(
        default_factory=lambda: ["fp32"],
        metadata={
            "help": (
                "Data types for activations."
            )
        }
    )

    prune_first: Optional[bool] = field(
        default=False,
        metadata={
            "help": (
                "Whether to prune tensors before the quantization."
            )
        }
    )

    initial_theta: Optional[float] = field(
        default=0.01,
        metadata={
            "help": (
                "The initial value for the quantized weight edges."
            )
        }
    )

    initial_phi: Optional[float] = field(
        default=0.01,
        metadata={
            "help": (
                "The initial value for the weight mask edges."
            )
        }
    )

    initial_psi: Optional[float] = field(
        default=0.01,
        metadata={
            "help": (
                "The initial value for the quantized activation edges."
            )
        }
    )

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    size_ratio_threshold: Optional[float] = field(
        default=1,
        metadata={
            "help": (
                "The constraint on the size of the quantized model, relative to the original model."
                "The ratio is computed as the size of the quantized model over the size of the original"
            )
        }
    )

    mu: Optional[float] = field(
        default=0.0005,
        metadata={
            "help": (
                "The parameter that defines the strength of the barrier function."
                "As it reaches zero, the solution of the optimization problem will converge to the true constrained problem solution."
            )
        }
    )

    temperature: Optional[float] = field(
        default=1,
        metadata={
            "help": (
                "The initial temperature used during the Gumbel Softmax computation."
            )
        }
    )

    temp_drop_rate: Optional[float] = field(
        default=0.25, 
        metadata={
            "help": (
                "The drop rate used to update the Gumbel Softmax temperature parameter every epoch."
            )
        }
    )

    freeze_weights: Optional[bool] = field(
        default=True, metadata={"help": ("Whether to freeze weights during the search.")}
    )

class QuantSparseLinear(nn.Module):
    """Quantize weights first then Prune.
    """
    def __init__(self, name, in_features, out_features, w_dtypes, w_dense, blocksize, a_dtypes):
        super(QuantSparseLinear, self).__init__()
        self.name = name
        self.in_features = in_features
        self.out_features = out_features
        self.w_dtypes = w_dtypes
        self.w_dense = w_dense
        self.blocksize = blocksize
        self.a_dtypes = a_dtypes
        self.tau = TAU

        # constant parameters
        # size and FLOPS computation parameters
        self.w_num_param = nn.Parameter(torch.Tensor([in_features * out_features]), requires_grad=False)
        self.w_bit_width = nn.Parameter(torch.Tensor([dtype.bits_per_value for dtype in self.w_dtypes]), requires_grad=False)
        self.w_bit_dense = nn.Parameter(torch.Tensor([float(density / self.blocksize) for density in self.w_dense]), requires_grad=False)
        self.a_bit_width = nn.Parameter(torch.Tensor([dtype.bits_per_value for dtype in self.a_dtypes]), requires_grad=False)

        # learned parameters
        # linear parameters
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        # quantized weight edge parameter
        self.theta = nn.Parameter(torch.Tensor(len(w_dtypes)))
        # activation edge parameter
        self.phi = nn.Parameter(torch.Tensor(len(w_dense)))
        # quantized activation edge parameter
        self.psi = nn.Parameter(torch.Tensor(len(a_dtypes)))

    def quant_nodes(self, tensor, dtypes):
        quantized_tensors = []
        for dtype in dtypes:
            new_tensor = # Use MS MXFP.quantize(tensor, dtype=dtype, axis=-1)
            quantized_tensors.append(new_tensor)
        nodes = torch.stack(quantized_tensors, dim=-1)
        return nodes

    def sparse_nodes(self, tensor, dense_list, blocksize):
        sparse_tensors = []
        for dense in dense_list:
            if dense == blocksize:
                new_tensor = torch.ones_like(self.weight)
            else:
                new_tensor = # Use MS MXFP.sparsify(tensor, dense=dense, blocksize=blocksize, axis=-1)
            sparse_tensors.append(new_tensor)
        nodes = torch.stack(sparse_tensors, dim=-1)
        return nodes

    def combine(self, nodes, edges):
        return torch.sum(nodes * edges, dim=-1)

    def sample_layer_spec(self):
        dtype = self.w_dtypes[torch.argmax(F.gumbel_softmax(self.theta, tau=self.tau, dim=0))]
        dense = self.w_dense[torch.argmax(F.gumbel_softmax(self.phi, tau=self.tau, dim=0))]
        if dense > 0 and dense < self.blocksize:
            return # Use MS MXFP.LayerSpec(atype=dtype, btype=dtype, bdense=dense, bsbs=self.blocksize)
        else:
            return # Use MS MXFP.LayerSpec(atype=dtype, btype=dtype)
        
    def forward(self, x):
        # quantize first
        weight_nodes = self.quant_nodes(self.weight, self.w_dtypes)
        self.p_theta = F.gumbel_softmax(self.theta, tau=self.tau, dim=0)
        quant_weight = self.combine(weight_nodes, self.p_theta)

        # then prune
        mask_nodes = self.sparse_nodes(quant_weight, self.w_dense, self.blocksize)
        p_phi = F.gumbel_softmax(self.phi, tau=self.tau, dim=0)
        sparse_weight = quant_weight * self.combine(mask_nodes, p_phi)

        output = F.linear(x, sparse_weight, self.bias)
        return output

    def w_size(self):
        avg_bit_dense = torch.dot(F.gumbel_softmax(self.phi, tau=self.tau, dim=0), self.w_bit_dense)
        avg_bit_width = torch.dot(F.gumbel_softmax(self.theta, tau=self.tau, dim=0), self.w_bit_width)
        return avg_bit_dense * avg_bit_width * self.w_num_param 

class SparseQuantLinear(QuantSparseLinear):
    """Prune weights first then quantize.
    """
    def forward(self, x):
        # prune first
        mask_nodes = self.sparse_nodes(self.weight, self.w_dense, self.blocksize)
        p_phi = F.gumbel_softmax(self.phi, tau=self.tau, dim=0)
        sparse_weight = self.weight * self.combine(mask_nodes, p_phi)
    
        # then quantize
        weight_nodes = self.quant_nodes(sparse_weight, self.w_dtypes)
        p_theta = F.gumbel_softmax(self.theta, tau=self.tau, dim=0)
        quant_weight = self.combine(weight_nodes, p_theta)

        output = F.linear(x, quant_weight, self.bias)
        return output

def supernet(model, w_dtypes, a_dtypes, w_dense, blocksize, initial_theta, initial_phi, quant_first):
    """Builds the supernet into the model directly.

    Returns a list of the modified Linear modules.
    """
    def aux(module, name, accumulator):
        for n, child in module.named_children():
            if len(name):
                newname = f"{name}.{n}"
            else:
                newname = n
            if isinstance(child, nn.Linear):
                in_features = child.in_features
                out_features = child.out_features
                if quant_first:
                    custom_linear = QuantSparseLinear(newname, in_features, out_features, w_dtypes, w_dense, blocksize, a_dtypes)
                else:
                    custom_linear = SparseQuantLinear(newname, in_features, out_features, w_dtypes, w_dense, blocksize, a_dtypes)
                custom_linear.weight.data = child.weight.data
                if child.bias is None:
                    custom_linear.bias = None
                else:
                    custom_linear.bias.data = child.bias.data.clone()
                custom_linear.theta.data = torch.ones(len(w_dtypes)) * initial_theta
                custom_linear.phi.data = torch.ones(len(w_dense)) * initial_phi
                setattr(module, n, custom_linear)
                accumulator.append(custom_linear)
            else:
                aux(child, newname, accumulator)

    accumulator = []
    aux(model, "model", accumulator)
    return accumulator

def sample_subnet(supernet):
    """Samples the subnetwork from the supernet.

    Modifies the network in-place.
    """
    def aux(module):
        for n, child in module.named_children():
            if isinstance(child, QuantSparseLinear):
                layerspec = child.sample_layer_spec()
                name = child.name
                # Use MS MXFP_linear = # Use MS MXFP.Linear(child, name, layerspec)
                setattr(module, n, # Use MS MXFP_linear)
            aux(child)

    aux(supernet)

def snapshot(model):
    state ={}
    for layer in model.modules():
        if isinstance(layer, QuantSparseLinear):
            pi = F.gumbel_softmax(layer.theta, tau=layer.tau, dim=0).data
            state[layer.name] = {}
            state[layer.name]["num_param"] = layer.w_num_param.data.item()
            state[layer.name]["sampled_dtype"] = layer.w_dtypes[torch.argmax(pi).item()].fullname
            state[layer.name]["weight_edges"] = {}
            for i in range(len(layer.w_dtypes)):
                state[layer.name]["weight_edges"][layer.w_dtypes[i].fullname] = pi[i].item()
    return state

class SnapshotCallback(transformers.TrainerCallback):
    def __init__(self, model):
        self.model = model
        self.snapshots = {}

    def on_epoch_end(self, args, state, control, **kwargs):
        snap = snapshot(self.model)
        self.snapshots[f"epoch_{round(state.epoch)}"] = snap

class TemperatureUpdateCallback(transformers.TrainerCallback):
    def __init__(self, update_fn):
        super(TemperatureUpdateCallback, self).__init__()
        self.update = update_fn

    def on_epoch_begin(self, args, state, control, **kwargs):
        args.temperature = 2.0 * math.exp(-args.temp_drop_rate * state.epoch)
        self.update()

class DebugPrinterCallback(transformers.TrainerCallback):
    def __init__(self, print_fn):
        super(DebugPrinterCallback, self).__init__()
        self.print_fn = print_fn

    def on_epoch_end(self, args, state, control, **kwargs):
        self.print_fn()

class Compressor(transformers.Trainer):
    def __init__(self, layers, **kwargs):
        super(Compressor, self).__init__(callbacks=[TemperatureUpdateCallback(self._update_temp)], **kwargs)
        self.layers = layers

    def compute_loss(self, model, inputs, return_outputs=False):
        if self.is_in_train:
            # crossentropy loss computation
            loss, outputs = super(Compressor, self).compute_loss(model, inputs, return_outputs=True)
            # size ratio computation for the size constraint
            num_param = torch.stack([l.w_num_param for l in self.layers]).flatten()
            # variance_regularizer = torch.sum(torch.prod(1 - pi, dim=1), dim=0)
            # quant_size = (avg_bit_dense * avg_w_bit_width).T@num_param
            quant_size = torch.sum(torch.stack([l.w_size() for l in self.layers]).flatten()) # should be a list of scalars summed up
            original_size = torch.sum(32 * num_param)
            size_ratio = quant_size / original_size
            # logarithmic barrier function + SOFTPLUS
            constraint = torch.log(F.softplus(self.args.size_ratio_threshold - size_ratio, beta=1000))
            # constrained problem is now expressed as a single objective function
            objective = loss - self.args.mu * constraint
            print(f"loss: {loss.mean().item()}, size_ratio: {size_ratio.item()}, constraint: {-self.args.mu*constraint}")
            if torch.isnan(constraint):
                raise ValueError("NaN value detected when computing the barrier function. Consider tuning the constraints and their parameters.")
            return (objective, outputs) if return_outputs else objective
        else:
            return super(Compressor, self).compute_loss(model, inputs, return_outputs)

    def _update_temp(self):
        # for debugging purpose
        for layer in self.layers:
            layer.tau = self.args.temperature

    def _check_temp(self):
        # for debugging purpose
        for layer in self.layers:
            print(layer.tau)

def train(training_args, data_args, trainer, train_dataset, last_checkpoint):
    # Training
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    return metrics

def evaluate_model(data_args, trainer, eval_dataset):
    # Evaluation
    metrics = trainer.evaluate()
    max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
    metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
    try:
        perplexity = math.exp(metrics["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    metrics["perplexity"] = perplexity
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)

    return metrics

def main():
    # # Use MS MXFP specific setup
    parser = argparse.ArgumentParser(description="Differentiable Mixed Precision Search")
    parser = # Use MS MXFP_common.add_common_args(parser)
    args, _ = parser.parse_known_args()
    # Use MS MXFP.initialize_core(enable=True, cuda=True, **dict(vars(args)))

    # HF specific setup
    parser = transformers.HfArgumentParser((ModelArguments, opt.DataTrainingArguments, TrainingArguments))
    model_args, data_args, training_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    transformers.utils.send_example_telemetry("run_clm", model_args, data_args)
    logger = opt.get_logger(training_args)
    last_checkpoint = opt.get_last_checkpoint(training_args)

    # Set seed before initializing model.
    transformers.set_seed(training_args.seed)
    raw_datasets = opt.get_datasets(model_args, data_args)
    model = opt.get_model(model_args, logger)
    tokenizer = opt.get_tokenizer(model_args)

    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    tokenized_datasets = opt.tokenize_datasets(data_args, training_args, raw_datasets, tokenizer)

    if data_args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                " override this default with `--block_size xxx`."
            )
            block_size = 1024
    else:
        if data_args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({data_args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(data_args.block_size, tokenizer.model_max_length)

    group_texts = opt.get_group_texts_fn(block_size)

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
    with training_args.main_process_first(desc="grouping texts together"):
        if not data_args.streaming:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                load_from_cache_file=not data_args.overwrite_cache,
                desc=f"Grouping texts in chunks of {block_size}",
            )
        else:
            lm_datasets = tokenized_datasets.map(
                group_texts,
                batched=True,
            )

    if training_args.do_train:
        if "train" not in tokenized_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = lm_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))

    if training_args.do_eval:
        if "validation" not in tokenized_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = lm_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        def preprocess_logits_for_metrics(logits, labels):
            if isinstance(logits, tuple):
                # Depending on the model and config, logits may contain extra tensors,
                # like past_key_values, but logits always come first
                logits = logits[0]
            return logits.argmax(dim=-1)
        metric = evaluate.load("accuracy")
        def compute_metrics(eval_preds):
            preds, labels = eval_preds
            # preds have the same shape as the labels, after the argmax(-1) has been calculated
            # by preprocess_logits_for_metrics but we need to shift the labels
            labels = labels[:, 1:].reshape(-1)
            preds = preds[:, :-1].reshape(-1)
            return metric.compute(predictions=preds, references=labels)

    w_dtypes = [# Use MS MXFP.DataType(code=c) for c in model_args.w_dtypes]
    a_dtypes = [# Use MS MXFP.DataType(code=c) for c in model_args.a_dtypes]

    if "bloom" in model_args.model_name_or_path:
        layers = supernet(model=model.transformer, 
                      w_dtypes=w_dtypes, 
                      a_dtypes=a_dtypes, 
                      w_dense=model_args.w_dense, 
                      blocksize=model_args.blocksize, 
                      initial_theta=model_args.initial_theta, 
                      initial_phi=model_args.initial_phi,
                      quant_first=not model_args.prune_first)
    else:
        layers = supernet(model=model.model, 
                          w_dtypes=w_dtypes, 
                          a_dtypes=a_dtypes, 
                          w_dense=model_args.w_dense, 
                          blocksize=model_args.blocksize, 
                          initial_theta=model_args.initial_theta, 
                          initial_phi=model_args.initial_phi,
                          quant_first=not model_args.prune_first)

    trainer = Compressor(layers=layers,
                  model=model,
                  args=training_args,
                  train_dataset=train_dataset if training_args.do_train else None,
                  eval_dataset=eval_dataset if training_args.do_eval else None,
                  tokenizer=tokenizer,
                  data_collator=transformers.default_data_collator,
                  compute_metrics=None
                  if training_args.do_eval and not transformers.is_torch_tpu_available() 
                  else None,
                  preprocess_logits_for_metrics=preprocess_logits_for_metrics
                  if training_args.do_eval and not transformers.is_torch_tpu_available()
                  else None)

    snapshot = SnapshotCallback(model)
    trainer.add_callback(snapshot)

    if training_args.freeze_weights:
        # freeze non-edge parameters
        for name, param in trainer.model.named_parameters():
            if not ("theta" in name or "phi" in name):
                param.requires_grad = False

    if training_args.do_train:
        logger.info("*** EDGES TRAINING ***")
        train_metrics = train(training_args, data_args, trainer, train_dataset, last_checkpoint)
    else:
        logger.info("Skipping mixed precision search.")

    # test subnet
    sample_subnet(trainer.model)

    if training_args.do_eval:
        logger.info("*** QUANTIZED MODEL EVALUATION ***")
        eval_metrics = evaluate_model(data_args, trainer, eval_dataset)
    else:
        logger.info("Skipping final evaluation.")

    print()

    # this part is extra logging with date, time, model and data args and per-epoch snapshots of the supernet parameters
    # some logs are duplicate of what huggingface logger will produce
    import os
    path = f"{training_args.output_dir}/{model_args.model_name_or_path}/{data_args.dataset_name}/{data_args.max_train_samples}/{training_args.size_ratio_threshold}"

    if not os.path.exists(path):
        os.makedirs(path)

    with open(f"{path}/training_arguments.json", mode="w+") as f:
            f.write(json.dumps(training_args.to_dict(), indent=1))
    
    if training_args.do_train:
        with open(f"{path}/train_results.json", mode="w+") as f:
            f.write(json.dumps(train_metrics, indent=1))

    if training_args.do_eval:
        with open(f"{path}/eval_results.json", mode="w+") as f:
            f.write(json.dumps(eval_metrics, indent=1))

    with open(f"{path}/model_arguments.json", mode="w+") as f:
        f.write(json.dumps(asdict(model_args), indent=1))

    with open(f"{path}/data_arguments.json", mode="w+") as f:
        f.write(json.dumps(asdict(data_args), indent=1))

    if not os.path.exists(f"{path}/supernet_state"):
        os.makedirs(f"{path}/supernet_state")

    for epoch, state in snapshot.snapshots.items():
        supernet_state = json.dumps(state, indent=1)
        with open(f"{path}/supernet_state/{epoch}.json", mode="w+") as f:
            f.write(supernet_state)

# BASELINE
# python src/dmips/llm.py --model_name_or_path facebook/opt-125m --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --per_device_eval_batch_size 1 --do_eval --output_dir $WORK/baseline/facebook/opt-125m/wikitext --overwrite_output_dir
# QUANTIZATION ONLY EXAMPLE (learn best dtypes):
# python src/dmips/llm.py --model_name_or_path facebook/opt-125m --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --do_eval --output_dir $WORK/out/ --mu 0.0025 --learning_rate 0.05 --overwrite_output_dir --size_ratio_threshold 0.18 --num_train_epochs 10 --w_dtypes bfp15b16p2 bfp13b16p2 bfp11b16p2 --do_train --max_train_samples 236
# SPARSITY ONLY EXAMPLE (learn best sparsity masks):
# python src/dmips/llm.py --model_name_or_path facebook/opt-125m --dataset_name wikitext --dataset_config_name wikitext-2-raw-v1 --per_device_train_batch_size 1 --per_device_eval_batch_size 1 --do_eval --output_dir $WORK/out/ --mu 0.0025 --learning_rate 0.1 --overwrite_output_dir --size_ratio_threshold 0.80 --num_train_epochs 10 --w_dense 8 16 --do_train --max_train_samples 436
if __name__ == "__main__":
    main()