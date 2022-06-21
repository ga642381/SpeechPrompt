import contextlib
import logging
import os
import sys
import time
from argparse import Namespace
from collections import OrderedDict
from itertools import chain
from pathlib import Path
from typing import Any, Dict, List

import torch
from fairseq import checkpoint_utils, models, optim, utils
from fairseq.dataclass.configs import FairseqConfig
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.distributed import utils as distributed_utils
from fairseq.file_io import PathManager
from fairseq.logging import meters, metrics
from fairseq.models.ema import build_ema
from fairseq.nan_detector import NanDetector
from fairseq.optim import lr_scheduler
from fairseq.trainer import Trainer as FairseqTrainer
from fairseq.utils import safe_hasattr
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


class PromptTrainer(FairseqTrainer):
    """
    Override save_checkpoint and load_checkpoint in Fairseq Trainer
    """

    def __init__(self, cfg: FairseqConfig, task, model, criterion, quantizer=None):
        super().__init__(cfg, task, model, criterion, quantizer)

    def save_checkpoint(self, filename, extra_state):
        """Save all training state in a checkpoint file."""
        logger.info(f"Saving checkpoint to {filename}")
        # call state_dict on all ranks in case it needs internal communication
        state_dict = utils.move_to_cpu(self.state_dict())
        state_dict["extra_state"].update(extra_state)

        # === save a template(base model) for prompt lm === #
        prompt_base_model_filename = Path(filename).parent / "base_prompt_model.pt"
        if not prompt_base_model_filename.is_file():
            checkpoint_utils.torch_persistent_save(
                state_dict,
                str(prompt_base_model_filename),
                async_write=self.cfg.checkpoint.write_checkpoints_asynchronously,
            )
            logger.info(f"Finished saving prompt base model to {str(prompt_base_model_filename)}")

        # === only save prompts (task embedding) === #
        only_save_prompt = False
        prompt_state_dict = OrderedDict()
        for name, params in state_dict["model"].items():
            if "prompt" in name:
                prompt_state_dict[name] = params
                if params.numel() > 0:  # if we are using prompt
                    only_save_prompt = True
            if "sep" in name:
                prompt_state_dict[name] = params
        if only_save_prompt:
            state_dict["model"] = prompt_state_dict

        # === save checkpoint === #
        if self.should_save_checkpoint_on_current_rank:
            checkpoint_utils.torch_persistent_save(
                state_dict,
                filename,
                async_write=self.cfg.checkpoint.write_checkpoints_asynchronously,
            )
        logger.info(f"Finished saving checkpoint to {filename}")

    def load_checkpoint(
        self,
        filename,
        reset_optimizer=False,
        reset_lr_scheduler=False,
        optimizer_overrides=None,
        reset_meters=False,
    ):
        """
        Load all training state from a checkpoint file.
        rank = 0 will load the checkpoint, and then broadcast it to all
        other ranks.
        """
        extra_state, self._optim_history, last_optim_state = None, [], None

        logger.info(f"Preparing to load checkpoint {filename}")
        is_distributed = self.data_parallel_world_size > 1
        bexists = PathManager.isfile(filename)
        if bexists:
            load_on_all_ranks = (
                self.cfg.checkpoint.load_checkpoint_on_all_dp_ranks
                # TPUs don't support broadcast yet, so load checkpoints
                # on every worker for now
                or self.tpu
                # FSDP requires loading checkpoint shards on all ranks
                or (self.is_fsdp and self.cfg.distributed_training.use_sharded_state)
                or getattr(self.cfg.model, "base_layers", 0) > 0
            )

            if load_on_all_ranks or self.data_parallel_rank == 0:
                state = checkpoint_utils.load_checkpoint_to_cpu(filename, load_on_all_ranks=load_on_all_ranks)
                last_optim_state = state.get("last_optimizer_state", None)

                # If doing zero_sharding, do not broadcast global optimizer
                # state. Later we will broadcast sharded states to each rank
                # to avoid memory from exploding.
                if (
                    not load_on_all_ranks
                    and self.cfg.distributed_training.zero_sharding == "os"
                    and "last_optimizer_state" in state
                    and is_distributed
                ):
                    state["last_optimizer_state"] = "SHARDED"
            else:
                last_optim_state = None
                state = None

            if is_distributed and not load_on_all_ranks:
                state = distributed_utils.broadcast_object(
                    state,
                    src_rank=0,
                    group=self.data_parallel_process_group,
                    dist_device=self.device,
                )
                if self.data_parallel_rank > 0:
                    last_optim_state = state.get("last_optimizer_state", None)

            # load model parameters
            try:
                if (
                    "optimizer_history" in state
                    and len(state["optimizer_history"]) > 0
                    and "num_updates" in state["optimizer_history"][-1]
                ):
                    self.model.set_num_updates(state["optimizer_history"][-1]["num_updates"])

                # this is the code related to AdaPrune
                # In short, it removes redundant heads in multi-head attention module based on heads importance provided
                # For more info, please refer to the paper: https://openreview.net/forum?id=_CMSV7FTzGI
                # The idea of prune in mha can be summarized as
                # Fine tune model (e.g. roberta encoder) on a certain datasets with regularization
                # After the model is trained. User could use get_reserve_head_index and _adaptive_prune_heads functions to get the top X heads with most importance.
                # Then user uses the rank to prune a new roberta encoder and save the pruned ckpt manually.
                # User will fine tune the the new roberta encoder via the ckpt saved above
                # To get rid of registering different pruned version of Roberta, I use the argument --mha-heads-to-keep to prune the Roberta model into a pruned version which matches the pruned ckpt.
                if (
                    safe_hasattr(self.model, "args")
                    and safe_hasattr(self.model.args, "mha_heads_to_keep")
                    and self.model.args.mha_heads_to_keep != -1
                ):
                    logger.info(
                        f"Prune model: keep {self.model.args.mha_heads_to_keep} heads for each multihead attention module"
                    )
                    for layer in self.model.encoder.sentence_encoder.layers:
                        reserve_head_index = layer.self_attn._get_reserve_head_index(
                            num_heads_to_keep=self.model.args.mha_heads_to_keep
                        )
                        layer.self_attn._adaptive_prune_heads(reserve_head_index=reserve_head_index)
                        layer.self_attn._set_skip_embed_dim_check()
                    logger.info(self.model)
                # this is the code related to AdaPrune
                # In short, it removes redundant units in feedforward layer in each transformer layer based on importance
                # For more info, please refer to the paper: https://openreview.net/forum?id=_CMSV7FTzGI
                # The idea of prune in ffn can be summarized as
                # Fine tune model (e.g. roberta encoder) on a certain datasets with regularization
                # After the model is trained. User could use _get_fc_rank and _prune_fc_layer functions to get the top X units with most importance.
                # Then user uses the rank to prune a new roberta encoder and save the pruned ckpt manually.
                # User will fine tune the the new roberta encoder via the ckpt saved above
                # To get rid of registering different pruned version of Roberta, I use the argument --ffn-blocks-to-remove to prune the Roberta model into a pruned version which matches the pruned ckpt.
                if (
                    safe_hasattr(self.model, "args")
                    and safe_hasattr(self.model.args, "ffn_blocks_to_remove")
                    and self.model.args.ffn_blocks_to_remove != -1
                ):
                    logger.info(
                        f"Prune model: remove {self.model.args.ffn_blocks_to_remove} ffn blocks for each transformer layer"
                    )
                    for layer in self.model.encoder.sentence_encoder.layers:
                        remove_index = layer._get_fc_rank(remove_num=self.model.args.ffn_blocks_to_remove)
                        layer._prune_fc_layer(remove_index=remove_index)
                    logger.info(self.model)

                # ===== !!! Prompt !!! ===== #
                # Since we modified the model architecture (build prompt embeddings)
                # we have to set strict=False to load the pretrained uLM
                # self.model.load_state_dict(
                #     state["model"], strict=True, model_cfg=self.cfg.model
                # )
                self.model.load_state_dict(state["model"], strict=False, model_cfg=self.cfg.model)
                # save memory for later steps
                del state["model"]
                if utils.has_parameters(self.get_criterion()):
                    self.get_criterion().load_state_dict(state["criterion"], strict=True)
                    del state["criterion"]

            except Exception:
                raise Exception(
                    "Cannot load model parameters from checkpoint {}; "
                    "please ensure that the architectures match.".format(filename)
                )
            extra_state = state["extra_state"]
            self._optim_history = state["optimizer_history"]

        if last_optim_state is not None and not reset_optimizer:
            # rebuild optimizer after loading model, since params may have changed
            self._build_optimizer()

            # only reload optimizer and lr_scheduler if they match
            last_optim = self._optim_history[-1]
            assert (
                last_optim["criterion_name"] == self.get_criterion().__class__.__name__
            ), f"Criterion does not match; please reset the optimizer (--reset-optimizer). {last_optim['criterion_name']} vs {self.get_criterion().__class__.__name__}"
            assert (
                last_optim["optimizer_name"] == self.optimizer.__class__.__name__
            ), f"Optimizer does not match; please reset the optimizer (--reset-optimizer). {last_optim['optimizer_name']} vs {self.optimizer.__class__.__name__}"

            if not reset_lr_scheduler:
                self.lr_scheduler.load_state_dict(last_optim["lr_scheduler_state"])

            if self.is_fsdp and not self.model.use_sharded_state:
                # if use_sharded_state, the last_optim_state is already sharded, skip this
                last_optim_state = self.model.get_shard_from_optim_state_dict(last_optim_state)
            elif not load_on_all_ranks and is_distributed:
                last_optim_state = self.optimizer.broadcast_global_state_dict(last_optim_state)

            self.optimizer.load_state_dict(last_optim_state, optimizer_overrides)

            self.set_num_updates(last_optim["num_updates"])

        if extra_state is not None:
            itr_state = extra_state["train_iterator"]
            epoch = itr_state["epoch"]

            if "previous_training_time" in extra_state:
                self._previous_training_time = extra_state["previous_training_time"]
                self._start_time = time.time()

            self.lr_step(epoch)

            if itr_state.get("version", 1) >= 2 and itr_state["iterations_in_epoch"] == 0:
                # reset meters at start of epoch
                reset_meters = True

            if "metrics" in extra_state and not reset_meters:
                metrics.load_state_dict(extra_state["metrics"])

                # reset TimeMeters, since their start times don't make sense anymore
                for meter in metrics.get_meters("default"):
                    if isinstance(meter, meters.TimeMeter):
                        meter.reset()

            if self.cfg.ema.store_ema:
                if "ema" not in extra_state:
                    logger.warn(
                        "EMA not found in checkpoint. But store_ema is True. " "EMA is re-initialized from checkpoint."
                    )
                    self.ema.restore(state["model"], build_fp32_params=self.cfg.ema.ema_fp32)
                else:
                    logger.info("Loading EMA from checkpoint")
                    self.ema.restore(extra_state["ema"], build_fp32_params=False)

                    if self.cfg.ema.ema_fp32:
                        if "ema_fp32_params" in extra_state:
                            logger.info("Loading EMA fp32 params from checkpoint")
                            self.ema.build_fp32_params(extra_state["ema_fp32_params"])
                        else:
                            logger.info("Building EMA fp32 params from EMA model in checkpoint")
                            self.ema.build_fp32_params()

            logger.info("Loaded checkpoint {} (epoch {} @ {} updates)".format(filename, epoch, self.get_num_updates()))

        else:
            logger.info("No existing checkpoint found {}".format(filename))

        return extra_state
