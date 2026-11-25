# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import inspect
import logging
import math
import os
import time
from types import SimpleNamespace
from typing import Optional

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, compute_self_distillation_loss, get_policy_loss_fn, kl_penalty
from verl.workers.utils.losses import apply_samod_aux_losses, read_samod_dreg_loss_weight
from verl.workers.utils.ohisa import (
    build_response_token_mask,
    compute_hidden_trajectory_loss,
    project_token_mask_to_valid_span,
    resolve_ohisa_from_module,
)
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.metric import AggregationType, Metric, reduce_metrics
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, slice_input_tensor, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class TrustRegionTeacher(nn.Module):
    def __init__(self, ref_module: nn.Module, student_module: nn.Module, mix_coef: float) -> None:
        super().__init__()
        self.ref_module = ref_module
        self.student_module = student_module
        self.mix_coef = float(mix_coef)

    def forward(self, *args, **kwargs):
        ref_out = self.ref_module(*args, **kwargs)
        student_out = self.student_module(*args, **kwargs)
        ref_logits = ref_out.logits if hasattr(ref_out, "logits") else ref_out[0]
        student_logits = student_out.logits if hasattr(student_out, "logits") else student_out[0]
        logits = torch.lerp(ref_logits, student_logits, self.mix_coef)
        return SimpleNamespace(logits=logits)


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.teacher_module: Optional[nn.Module] = None
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

    def _update_teacher(self) -> None:
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        if not self_distillation_cfg or loss_mode != "vopd":
            return
        teacher_model_source = getattr(self_distillation_cfg, "teacher_model_source", "legacy")
        if teacher_model_source != "legacy":
            return
        teacher_regularization = getattr(self_distillation_cfg, "teacher_regularization", "ema")
        if self.teacher_module is None or self.teacher_module is self.actor_module:
            raise ValueError("Teacher updates require a separate teacher_module in the actor worker.")
        with torch.no_grad():
            if teacher_regularization == "ema":
                update_rate = getattr(self_distillation_cfg, "teacher_update_rate", 0.0)
                if update_rate == 0.0:
                    return
                for teacher_param, student_param in zip(
                    self.teacher_module.parameters(),
                    self.actor_module.parameters(),
                ):
                    student_data = student_param.data.to(device=teacher_param.device)
                    teacher_param.data.mul_(1.0 - update_rate).add_(student_data, alpha=update_rate)
                return

            if teacher_regularization == "progressive":
                teacher_update_interval = getattr(self_distillation_cfg, "teacher_update_interval", None)
                if teacher_update_interval is None:
                    raise ValueError("Progressive teacher requires self_distillation.teacher_update_interval.")
                global_steps = getattr(self, "_current_global_steps", None)
                if global_steps is None or global_steps % teacher_update_interval != 0:
                    return
                for teacher_param, student_param in zip(
                    self.teacher_module.parameters(),
                    self.actor_module.parameters(),
                ):
                    teacher_param.data.copy_(student_param.data.to(device=teacher_param.device))
                for teacher_buffer, student_buffer in zip(
                    self.teacher_module.buffers(),
                    self.actor_module.buffers(),
                ):
                    teacher_buffer.data.copy_(student_buffer.data.to(device=teacher_buffer.device))
                return

            return

    @staticmethod
    def _has_non_empty_multi_modal_inputs(multi_modal_inputs) -> bool:
        if multi_modal_inputs is None:
            return False
        for inputs in multi_modal_inputs:
            if inputs is None:
                continue
            inputs = getattr(inputs, "data", inputs)
            if isinstance(inputs, dict):
                if not inputs:
                    continue
                for value in inputs.values():
                    if value is None:
                        continue
                    if isinstance(value, torch.Tensor) and value.numel() == 0:
                        continue
                    return True
            else:
                return True
        return False

    @staticmethod
    def _add_tail_bucket(log_probs: torch.Tensor) -> torch.Tensor:
        log_s = torch.logsumexp(log_probs, dim=-1, keepdim=True)
        log_s = torch.clamp(log_s, max=-1e-7)
        tail_log = torch.log(-torch.expm1(log_s))
        return torch.cat([log_probs, tail_log], dim=-1)

    @staticmethod
    def _sanitize_logits(logits: torch.Tensor) -> torch.Tensor:
        if bool(torch.isfinite(logits).all()):
            return logits
        limit = 1e4
        return torch.nan_to_num(logits, nan=0.0, posinf=limit, neginf=-limit).clamp_(-limit, limit)

    @staticmethod
    def _sanitize_log_tensor(values: torch.Tensor) -> torch.Tensor:
        if bool(torch.isfinite(values).all()):
            return values
        limit = 1e4
        return torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=-limit).clamp_(min=-limit, max=0.0)

    @staticmethod
    def _sanitize_finite_tensor(values: torch.Tensor) -> torch.Tensor:
        if bool(torch.isfinite(values).all()):
            return values
        return torch.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

    @staticmethod
    def _sanitize_scalar_loss(loss: torch.Tensor) -> tuple[torch.Tensor, bool]:
        if bool(torch.isfinite(loss.detach()).all()):
            return loss, False
        return torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0), True

    @staticmethod
    def _build_response_positions(
        response_start_idx: torch.Tensor,
        response_length: int,
        seqlen: int,
    ) -> torch.Tensor:
        if response_start_idx.dim() != 1:
            raise ValueError(f"response_start_idx must be rank-1, got shape {tuple(response_start_idx.shape)}")
        if (response_start_idx < 1).any():
            raise ValueError("response_start_idx must be >= 1 so response logits have a preceding context token.")

        offsets = torch.arange(response_length, device=response_start_idx.device, dtype=response_start_idx.dtype)
        response_positions = response_start_idx.unsqueeze(1) - 1 + offsets.unsqueeze(0)
        if response_positions.numel() > 0:
            if response_positions.min().item() < 0 or response_positions.max().item() >= seqlen:
                raise ValueError(
                    f"Response positions out of bounds for seqlen={seqlen}: "
                    f"min={response_positions.min().item()}, max={response_positions.max().item()}"
                )
        return response_positions.to(dtype=torch.long)

    @staticmethod
    def _select_response_positions(
        hidden_states: torch.Tensor,
        response_length: int,
        response_start_idx: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if response_start_idx is None:
            return hidden_states[:, -response_length - 1 : -1, ...]

        response_positions = DataParallelPPOActor._build_response_positions(
            response_start_idx=response_start_idx,
            response_length=response_length,
            seqlen=hidden_states.size(1),
        )
        if hidden_states.dim() == 2:
            return torch.gather(hidden_states, dim=1, index=response_positions)

        gather_index = response_positions.view(
            response_positions.size(0),
            response_positions.size(1),
            *([1] * (hidden_states.dim() - 2)),
        ).expand(response_positions.size(0), response_positions.size(1), *hidden_states.shape[2:])
        return torch.gather(hidden_states, dim=1, index=gather_index)

    def _dump_self_distillation_log_probs(
        self,
        *,
        meta_info: dict,
        self_distillation_cfg,
        dump_chunks: list[dict[str, torch.Tensor]],
    ) -> None:
        dump_root = self_distillation_cfg.get("log_prob_dump_dir", None)
        if not dump_root or not dump_chunks:
            return

        global_step = meta_info.get("global_steps")
        if global_step is None:
            return

        experiment_name = os.environ.get("EXPERIMENT", "unknown_experiment")
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        else:
            rank = 0

        normalized_root = os.path.normpath(dump_root)
        if os.path.basename(normalized_root) == experiment_name:
            save_dir = normalized_root
        else:
            save_dir = os.path.join(normalized_root, experiment_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{int(global_step)}.rank{rank}.pt")

        student_log_probs = torch.cat([chunk["student_log_probs"] for chunk in dump_chunks], dim=0)
        teacher_log_probs = torch.cat([chunk["teacher_log_probs"] for chunk in dump_chunks], dim=0)

        torch.save(
            {
                "student_log_probs": student_log_probs,
                "teacher_log_probs": teacher_log_probs,
                "global_step": int(global_step),
                "rank": rank,
                "experiment_name": experiment_name,
                "distribution_size": int(student_log_probs.shape[-1]),
                "num_valid_tokens": int(student_log_probs.shape[0]),
                "distillation_topk": self_distillation_cfg.get("distillation_topk", None),
                "distillation_add_tail": bool(self_distillation_cfg.get("distillation_add_tail", False)),
            },
            save_path,
        )

    @staticmethod
    def _unwrap_module(module: nn.Module) -> nn.Module:
        current = module
        while hasattr(current, "module"):
            current = current.module
        return current

    def _model_accepts_kwarg(self, module: nn.Module, name: str) -> bool:
        """True if forward accepts ``name`` explicitly, or via verl patched ``**kwargs``."""
        try:
            forward = self._unwrap_module(module).forward
            params = inspect.signature(forward).parameters
        except (TypeError, ValueError):
            return False
        if name in params:
            return True
        # Monkey-patched dense/sparse forwards pop these from **kwargs.
        if name in {
            "samod_cu_seqlens",
            "logits_response_length",
            "logits_response_start_idx",
            "logits_rmpad_indices",
            "logits_batch_size",
            "logits_seqlen",
        } and any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            fwd_module = getattr(forward, "__module__", "") or ""
            return (
                fwd_module.startswith("verl.models.transformers.qwen3_5")
                or fwd_module.startswith("verl.models.transformers.qwen3_vl")
            )
        return False

    def _select_forward_kwargs(self, module: nn.Module, kwargs: dict) -> dict:
        """Drop multimodal keys the concrete forward does not accept.

        ``images_seqlens`` is metrics-only. Models with ``**kwargs`` still receive
        the full dict and pop what they use.
        """
        try:
            params = inspect.signature(self._unwrap_module(module).forward).parameters
        except (TypeError, ValueError):
            return kwargs
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in params.values()):
            return kwargs
        return {key: value for key, value in kwargs.items() if key in params}

    def _forward_micro_batch(
        self,
        micro_batch: dict[str, torch.Tensor],
        temperature: float,
        calculate_entropy: bool = False,
        return_all_logps: bool = False,
        distill_topk: Optional[int] = None,
        topk_indices: Optional[torch.Tensor] = None,
        module: Optional[nn.Module] = None,
        return_samod_trajectory: bool = False,
        output_hidden_states: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
                if distill_topk or topk_indices is set:
                    topk_logps: (bs, response_len, k)
                    topk_indices: (bs, response_len, k)
                optionally samod_dreg_loss / extra_metrics from SAMoD models
                optionally samod_trajectory when return_samod_trajectory=True
                optionally hidden_states when output_hidden_states=True
        """
        temperature = float(temperature)
        if not math.isfinite(temperature) or temperature <= 0.0:
            raise ValueError(
                f"Training/log-prob forward requires a finite positive temperature, got {temperature}. "
                "For greedy rollout set do_sample=false and keep temperature=1.0."
            )
        if return_samod_trajectory and output_hidden_states:
            raise ValueError(
                "return_samod_trajectory and output_hidden_states are mutually exclusive in one forward "
                "(student uses trajectory; teacher uses output_hidden_states)."
            )
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        use_topk = distill_topk is not None or topk_indices is not None
        compute_all_logps = return_all_logps and not use_topk
        return_topk_indices = use_topk and topk_indices is None
        if (return_all_logps or use_topk) and self.use_fused_kernels:
            raise ValueError("Logit distillation requires disabling fused kernels.")

        model = module or self.actor_module
        use_remove_padding = self.use_remove_padding
        # Capability gates: SAMoD/Qwen3.5-sparse accept these; vanilla HF models must not receive them.
        accepts_samod_cu_seqlens = self._model_accepts_kwarg(model, "samod_cu_seqlens")
        accepts_response_window_logits = self._model_accepts_kwarg(model, "logits_response_length")

        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            if return_samod_trajectory or output_hidden_states:
                raise ValueError("OHiSA / output_hidden_states is incompatible with prefix_grouper.")
            can_use_pg = (
                not use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
                and not return_all_logps
                and not use_topk
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=model,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            response_start_idx = micro_batch.get("response_start_idx")
            if response_start_idx is not None:
                response_start_idx = response_start_idx.to(device=input_ids.device, dtype=torch.long)
                if response_start_idx.shape != (batch_size,):
                    raise ValueError(
                        f"response_start_idx shape must be ({batch_size},), got {tuple(response_start_idx.shape)}"
                    )
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(model, "module", model).config,
                        "vision_config",
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                # Qwen3.5 linear_attention / SAMoD TextModel need per-sample boundaries on rmpad.
                if accepts_samod_cu_seqlens:
                    extra_args["samod_cu_seqlens"] = cu_seqlens
                # Sequence-aligned multimodal masks must match flat input_ids [1, nnz].
                if "mm_token_type_ids" in multi_modal_inputs and multi_modal_inputs["mm_token_type_ids"] is not None:
                    mm_ids = multi_modal_inputs["mm_token_type_ids"]
                    if mm_ids.dim() == 2 and mm_ids.shape == (batch_size, seqlen):
                        mm_rmpad, _, _, *_ = unpad_input(mm_ids.unsqueeze(-1), attention_mask)
                        multi_modal_inputs["mm_token_type_ids"] = mm_rmpad.view(1, -1)
                # Avoid [nnz, V] lm_head OOM: pad hiddens → response window → lm_head inside FSDP.
                if not self.use_fused_kernels and accepts_response_window_logits:
                    extra_args["logits_rmpad_indices"] = indices
                    extra_args["logits_batch_size"] = batch_size
                    extra_args["logits_seqlen"] = seqlen
                    extra_args["logits_response_length"] = response_length
                    if response_start_idx is not None:
                        extra_args["logits_response_start_idx"] = response_start_idx
                if return_samod_trajectory:
                    extra_args["return_samod_trajectory"] = True
                if output_hidden_states:
                    extra_args["output_hidden_states"] = True
                # NEO THW indexes count left padding and reset per sample. Recomputing
                # them on the packed [1, nnz] row shifts RoPE and the image time ids
                # relative to the padded teacher / rollout forward.
                chat = self._unwrap_module(model)
                if not is_mask_all_zero and hasattr(chat, "get_thw_indexes_batch"):
                    indexes_full = chat.get_thw_indexes_batch(
                        input_ids, multi_modal_inputs.get("grid_hw")
                    )
                    indexes_rmpad = index_first_axis(
                        rearrange(indexes_full, "c b s -> (b s) c"), indices
                    )
                    extra_args["indexes"] = indexes_rmpad.transpose(0, 1).unsqueeze(1).contiguous()

                output = model(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **self._select_forward_kwargs(model, multi_modal_inputs),
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = self._sanitize_log_tensor(output.log_probs.squeeze(0))  # (total_nnz,)
                    entropy_rmpad = self._sanitize_finite_tensor(output.entropy.squeeze(0))  # (total_nnz,)

                    # gather log_prob if sp > 1
                    if self.use_ulysses_sp:
                        log_probs = gather_outputs_and_unpad(
                            log_probs,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        if calculate_entropy:
                            entropy_rmpad = gather_outputs_and_unpad(
                                entropy_rmpad,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )

                    if is_mask_all_zero:
                        log_probs = log_probs[:0]
                        if calculate_entropy:
                            entropy_rmpad = entropy_rmpad[:0]

                    if calculate_entropy:
                        full_entropy = pad_input(
                            hidden_states=entropy_rmpad.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                        entropy = self._select_response_positions(
                            full_entropy.squeeze(-1),
                            response_length=response_length,
                            response_start_idx=response_start_idx,
                        )
                    full_log_probs = pad_input(
                        hidden_states=log_probs.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    log_probs = self._select_response_positions(
                        full_log_probs.squeeze(-1),
                        response_length=response_length,
                        response_start_idx=response_start_idx,
                    )
                    all_logps = None
                    topk_logps = None
                    if compute_all_logps or use_topk:
                        raise ValueError(
                            "rmpad + fused kernels does not support full/topk logit distillation; "
                            "disable fused kernels for VOPD."
                        )
                else:
                    # Response-window logits [B, R, V] when model supports logits_response_*; else full rmpad.
                    logits = output.logits
                    if logits is None:
                        raise RuntimeError(
                            "rmpad forward returned logits=None; expected response-window lm_head "
                            "(logits_rmpad_indices / logits_response_length) for SAMoD models."
                        )
                    if accepts_response_window_logits and bool(
                        getattr(output, "logits_are_response_window", False)
                    ):
                        logits.div_(temperature)
                        logits = self._sanitize_logits(logits)
                        log_probs = logprobs_from_logits(
                            logits,
                            micro_batch["responses"],
                            inplace_backward=not (calculate_entropy or compute_all_logps or use_topk),
                        )
                        all_logps = torch.log_softmax(logits, dim=-1) if compute_all_logps else None
                        if use_topk:
                            logits_for_distill = logits.float()
                            if topk_indices is None:
                                topk = min(distill_topk, logits_for_distill.size(-1))
                                topk_logits, topk_indices = torch.topk(logits_for_distill, topk, dim=-1)
                            else:
                                topk_logits = torch.gather(logits_for_distill, dim=-1, index=topk_indices)
                            logsumexp = torch.logsumexp(logits_for_distill, dim=-1, keepdim=True)
                            topk_logps = topk_logits - logsumexp
                        if calculate_entropy:
                            entropy = (
                                verl_F.entropy_from_logits(logits)
                                if not self.config.entropy_checkpointing
                                else torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                            )
                        if calculate_sum_pi_squared:
                            sum_pi_squared = (
                                self.calculate_sum_pi_squared_from_logits(logits)
                                if not sum_pi_squared_checkpointing
                                else torch.utils.checkpoint.checkpoint(
                                    self.calculate_sum_pi_squared_from_logits, logits
                                )
                            )
                    elif accepts_response_window_logits:
                        raise RuntimeError(
                            "rmpad path received non-response-window logits from a model that "
                            f"accepts logits_response_length (shape={tuple(logits.shape)})."
                        )
                    else:
                        # Official dense path: full rmpad logits then pad + slice response.
                        logits_rmpad = logits.squeeze(0)
                        logits_rmpad.div_(temperature)
                        logits_rmpad = self._sanitize_logits(logits_rmpad)

                        if self.use_ulysses_sp:
                            logits_rmpad = gather_outputs_and_unpad(
                                logits_rmpad,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                        if is_mask_all_zero:
                            logits_rmpad = logits_rmpad[:0]

                        full_logits = pad_input(
                            hidden_states=logits_rmpad,
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                        logits = self._select_response_positions(
                            full_logits,
                            response_length=response_length,
                            response_start_idx=response_start_idx,
                        )
                        log_probs = logprobs_from_logits(
                            logits,
                            micro_batch["responses"],
                            inplace_backward=not (calculate_entropy or compute_all_logps or use_topk),
                        )
                        all_logps = torch.log_softmax(logits, dim=-1) if compute_all_logps else None
                        if use_topk:
                            logits_for_distill = logits.float()
                            if topk_indices is None:
                                topk = min(distill_topk, logits_for_distill.size(-1))
                                topk_logits, topk_indices = torch.topk(logits_for_distill, topk, dim=-1)
                            else:
                                topk_logits = torch.gather(logits_for_distill, dim=-1, index=topk_indices)
                            logsumexp = torch.logsumexp(logits_for_distill, dim=-1, keepdim=True)
                            topk_logps = topk_logits - logsumexp
                        if calculate_entropy:
                            entropy = (
                                verl_F.entropy_from_logits(logits)
                                if not self.config.entropy_checkpointing
                                else torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                            )
                        if calculate_sum_pi_squared:
                            sum_pi_squared = (
                                self.calculate_sum_pi_squared_from_logits(logits)
                                if not sum_pi_squared_checkpointing
                                else torch.utils.checkpoint.checkpoint(
                                    self.calculate_sum_pi_squared_from_logits, logits
                                )
                            )

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True
                # Avoid materializing full-sequence [B, S, V] logits (OOM on long padded seqs).
                if not self.use_fused_kernels and accepts_response_window_logits:
                    extra_args["logits_response_length"] = response_length
                    if response_start_idx is not None:
                        extra_args["logits_response_start_idx"] = response_start_idx
                if return_samod_trajectory:
                    extra_args["return_samod_trajectory"] = True
                if output_hidden_states:
                    extra_args["output_hidden_states"] = True
                output = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **self._select_forward_kwargs(model, multi_modal_inputs),
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = self._sanitize_log_tensor(output.log_probs[:, -response_length - 1 : -1])
                    entropy = self._sanitize_finite_tensor(output.entropy[:, -response_length - 1 : -1])  # (bsz, response_length)

                else:
                    logits = output.logits
                    if not bool(getattr(output, "logits_are_response_window", False)):
                        logits = self._select_response_positions(
                            logits,
                            response_length=response_length,
                            response_start_idx=response_start_idx,
                        )

                    logits.div_(temperature)
                    logits = self._sanitize_logits(logits)
                    log_probs = logprobs_from_logits(
                        logits,
                        micro_batch["responses"],
                        inplace_backward=not (calculate_entropy or compute_all_logps or use_topk),
                    )
                    if compute_all_logps:
                        all_logps = torch.log_softmax(logits, dim=-1)
                    if use_topk:
                        logits_for_distill = logits.float()
                        if topk_indices is None:
                            topk = min(distill_topk, logits_for_distill.size(-1))
                            topk_logits, topk_indices = torch.topk(logits_for_distill, topk, dim=-1)
                        else:
                            topk_logits = torch.gather(logits_for_distill, dim=-1, index=topk_indices)
                        logsumexp = torch.logsumexp(logits_for_distill, dim=-1, keepdim=True)
                        topk_logps = topk_logits - logsumexp
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )

            outputs = {"log_probs": log_probs}
            samod_dreg_loss = getattr(output, "samod_dreg_loss", None)
            if samod_dreg_loss is not None:
                outputs["samod_dreg_loss"] = samod_dreg_loss
                outputs["samod_dreg_loss_weight"] = read_samod_dreg_loss_weight(
                    getattr(self._unwrap_module(model), "config", None)
                )
            samod_dreg_loss_per_sample = getattr(output, "samod_dreg_loss_per_sample", None)
            if samod_dreg_loss_per_sample is not None:
                outputs["samod_dreg_loss_per_sample"] = samod_dreg_loss_per_sample
            samod_extra_metrics = getattr(output, "extra_metrics", None)
            if samod_extra_metrics:
                outputs["extra_metrics"] = samod_extra_metrics
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            if compute_all_logps:
                outputs["all_logps"] = all_logps
            if use_topk:
                outputs["topk_logps"] = topk_logps
                if return_topk_indices:
                    outputs["topk_indices"] = topk_indices
            if return_samod_trajectory:
                traj = getattr(output, "samod_trajectory", None)
                if traj is None:
                    raise RuntimeError(
                        "return_samod_trajectory=True but model output has no samod_trajectory. "
                        "Use an SAMoD student that supports OHiSA trajectory capture."
                    )
                outputs["samod_trajectory"] = traj
            if output_hidden_states:
                hiddens = getattr(output, "hidden_states", None)
                if hiddens is None:
                    raise RuntimeError(
                        "output_hidden_states=True but model output has no hidden_states. "
                        "Teacher forward must propagate output_hidden_states to the backbone."
                    )
                if use_remove_padding and hiddens and tuple(hiddens[0].shape[:2]) != (batch_size, seqlen):
                    # Qwen3.5 sparse rmpad internally left-aligns each valid row.
                    # Restore Teacher hiddens to the original padded columns so
                    # they align with cache-replayed student trajectories.
                    valid_lengths = attention_mask.to(dtype=torch.long).sum(dim=1).tolist()
                    restored = []
                    for hidden in hiddens:
                        flat_hidden = torch.cat(
                            [hidden[row, : int(length)] for row, length in enumerate(valid_lengths)],
                            dim=0,
                        )
                        restored.append(
                            pad_input(
                                hidden_states=flat_hidden,
                                indices=indices,
                                batch=batch_size,
                                seqlen=seqlen,
                            )
                        )
                    hiddens = tuple(restored)
                outputs["hidden_states"] = hiddens
            return outputs

    def _optimizer_step(self):
        self._last_nonfinite_grad_norm = False
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
            self._last_nonfinite_grad_norm = True
            return grad_norm

        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if "response_start_idx" in data.batch:
            select_keys.append("response_start_idx")
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                # Rollout log-probs stay on the normal forward. OHiSA is actor update only.
                outputs = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    return_samod_trajectory=False,
                    output_hidden_states=False,
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_topk_log_prob(self, data: DataProto) -> dict[str, torch.Tensor]:
        """Compute log probabilities on caller-provided or student-selected top-k token ids."""
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        distill_topk = data.meta_info.get("distill_topk", None)
        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        if "response_start_idx" in data.batch:
            select_keys.append("response_start_idx")
        if "topk_indices" in data.batch:
            select_keys.append("topk_indices")
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        topk_logps_lst = []
        topk_indices_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            requested_topk_indices = model_inputs.get("topk_indices", None)
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs,
                    temperature=temperature,
                    calculate_entropy=False,
                    distill_topk=None if requested_topk_indices is not None else distill_topk,
                    topk_indices=requested_topk_indices,
                )
            topk_logps_lst.append(outputs["topk_logps"])
            if requested_topk_indices is None:
                topk_indices_lst.append(outputs["topk_indices"])

        topk_logps = torch.concat(topk_logps_lst, dim=0)
        if use_dynamic_bsz:
            topk_logps = restore_dynamic_batch(topk_logps, batch_idx_list)
        result = {"topk_logps": topk_logps}

        if topk_indices_lst:
            topk_indices = torch.concat(topk_indices_lst, dim=0)
            if use_dynamic_bsz:
                topk_indices = restore_dynamic_batch(topk_indices, batch_idx_list)
            result["topk_indices"] = topk_indices

        return result

    @staticmethod
    def _effective_loss_mask(batch: DataProto) -> torch.Tensor:
        """Response mask actually used by the actor loss, including skipped prompts."""
        mask = batch.batch["response_mask"]
        overlong = batch.batch.get("overlong_prompt_mask")
        if overlong is not None:
            overlong = overlong.to(dtype=torch.bool).view(-1)
            if overlong.shape[0] == mask.shape[0] and bool(overlong.any()):
                mask = mask.clone()
                mask[overlong] = 0
        distill = batch.batch.get("self_distillation_mask")
        if distill is not None:
            mask = mask * distill.unsqueeze(1)
        return mask

    def _global_distillation_batch_info(self, mini_batch: DataProto) -> dict[str, int]:
        """Denominators that stay fixed when the same mini-batch is split differently.

        FSDP averages gradients. Dividing each micro-batch by its own token count
        and then by ``grad_accum_steps`` is the mean of those micro-batch means,
        so a longer response changes the update when the split changes. Every
        micro-batch instead divides by this all-reduced count. Multiplying by the
        data-parallel size cancels that average and recovers the global token mean.
        """
        mask = self._effective_loss_mask(mini_batch)
        local_tokens = int(mask.sum().item())
        local_seqs = int((mask.sum(dim=-1) > 0).sum().item())
        counts = torch.tensor(
            [local_tokens, local_seqs], dtype=torch.long, device=get_device_id()
        )
        dp_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(counts, op=torch.distributed.ReduceOp.SUM)
            dp_size = torch.distributed.get_world_size()
        return {
            "batch_num_tokens": max(int(counts[0].item()), 1),
            "global_batch_size": max(int(counts[1].item()), 1),
            "dp_size": int(dp_size),
            "local_num_tokens": max(local_tokens, 1),
            "local_num_samples": int(mini_batch.batch["response_mask"].shape[0]),
        }

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        self._current_global_steps = data.meta_info.get("global_steps")
        # make sure we are in training mode
        self.actor_module.train()
        # A fixed/ref teacher is inference-only. This is especially important when
        # it uses the SAMoD architecture for rollout-aligned cache replay: eval mode
        # disables training-only score-loss work and keeps its router deterministic.
        if self.teacher_module is not None and self.teacher_module is not self.actor_module:
            self.teacher_module.eval()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")

        self_distillation_enabled = loss_mode == "vopd"
        self_distillation_cfg = getattr(self.config, "self_distillation", None)
        external_teacher_enabled = False
        if self_distillation_enabled:
            if self_distillation_cfg is None:
                raise ValueError(f"loss_mode={loss_mode} requires actor.self_distillation config.")
            external_teacher_enabled = bool(self_distillation_cfg.get("external_teacher", False))
            self_distillation_required_keys = {
                "teacher_input_ids",
                "teacher_attention_mask",
                "teacher_position_ids",
                "teacher_response_start_idx",
                "self_distillation_mask",
            }
            if external_teacher_enabled:
                if not self_distillation_cfg.full_logit_distillation or self_distillation_cfg.distillation_topk is None:
                    raise ValueError("external_teacher requires full_logit_distillation with distillation_topk set.")
                self_distillation_required_keys.update({"student_topk_indices", "teacher_topk_logps"})
            assert self_distillation_required_keys.issubset(set(data.batch.keys())), f"Missing required keys: {self_distillation_required_keys - set(data.batch.keys())}"

        ohisa_cfg = resolve_ohisa_from_module(self.actor_module)
        # ``track``: compute/log L_hidden even when λ==0; ``enabled``: λ enters optim.
        ohisa_needs_teacher_trajectory = False
        if ohisa_cfg.track:
            if not self_distillation_enabled:
                raise ValueError(
                    "OHiSA (config present / weight!=0) requires loss_mode=vopd with a teacher forward."
                )
            if external_teacher_enabled:
                raise ValueError(
                    "OHiSA requires an in-process teacher_module with output_hidden_states; "
                    "external_teacher is not supported."
                )
            if self.use_fused_kernels:
                raise ValueError("OHiSA requires disabling fused kernels.")
            teacher_regularization = (
                self_distillation_cfg.get("teacher_regularization", "ema") if self_distillation_cfg else "ema"
            )
            if teacher_regularization == "trust-region":
                raise ValueError(
                    "OHiSA is incompatible with trust-region teacher (logits mix has no layer hiddens)."
                )

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
        ]
        if "response_start_idx" in data.batch.keys():
            select_keys.append("response_start_idx")
        if not self_distillation_enabled or "advantages" in data.batch.keys():
            select_keys.append("advantages")
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        if self_distillation_enabled:
            select_keys.extend(list(self_distillation_required_keys))
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("multi_modal_inputs")
        )
        has_teacher_multi_modal_inputs = self._has_non_empty_multi_modal_inputs(
            data.non_tensor_batch.get("teacher_multi_modal_inputs")
        )
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if has_teacher_multi_modal_inputs:
            non_tensor_select_keys.append("teacher_multi_modal_inputs")
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        if self_distillation_enabled:
            metrics["actor/grpo_loss"] = 0.0
            metrics["actor/vopd_loss"] = 0.0
            metrics["actor/vopd_loss_weighted"] = 0.0
        distill_dump_chunks = []
        stage_wall_time_totals = None
        if self_distillation_enabled:
            stage_wall_time_totals = {
                "timing_s/update_actor/student_forward": 0.0,
                "timing_s/update_actor/teacher_forward": 0.0,
                "timing_s/update_actor/loss_compute": 0.0,
                "timing_s/update_actor/backward": 0.0,
                "timing_s/update_actor/optimizer_step": 0.0,
                "timing_s/update_actor/teacher_ema_update": 0.0,
            }
        did_update = False
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    grad_accum_steps = int(self.config.get("grad_accum_steps", 1) or 1)
                    if grad_accum_steps > 1:
                        if len(mini_batch) % grad_accum_steps != 0:
                            raise ValueError(
                                f"local mini-batch size {len(mini_batch)} must be divisible by "
                                f"grad_accum_steps {grad_accum_steps}"
                            )
                        chunk_size = len(mini_batch) // grad_accum_steps
                        micro_batches = []
                        for chunk in mini_batch.split(chunk_size):
                            chunk_micros, _ = prepare_dynamic_batch(chunk, max_token_len=max_token_len)
                            micro_batches.extend(chunk_micros)
                    else:
                        micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    # Fixed sample count on every rank. grad_accum_steps>1 splits the
                    # local mini-batch into that many equal micro-batches; otherwise
                    # the split is ppo_micro_batch_size_per_gpu. One optimizer step
                    # follows the whole loop, with loss scaled by 1 / num_micros.
                    grad_accum_steps = int(self.config.get("grad_accum_steps", 1) or 1)
                    if grad_accum_steps > 1:
                        if len(mini_batch) % grad_accum_steps != 0:
                            raise ValueError(
                                f"local mini-batch size {len(mini_batch)} must be divisible by "
                                f"grad_accum_steps {grad_accum_steps}"
                            )
                        micro_batch_size = len(mini_batch) // grad_accum_steps
                        self.gradient_accumulation = grad_accum_steps
                    else:
                        micro_batch_size = self.config.ppo_micro_batch_size_per_gpu
                        self.gradient_accumulation = (
                            self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                        )
                    micro_batches = mini_batch.split(micro_batch_size)

                self.actor_optimizer.zero_grad()
                batch_info = (
                    self._global_distillation_batch_info(mini_batch) if self_distillation_enabled else None
                )
                if batch_info is None:
                    local_tokens = max(int(self._effective_loss_mask(mini_batch).sum().item()), 1)
                    local_samples = int(mini_batch.batch["response_mask"].shape[0])
                else:
                    local_tokens = batch_info["local_num_tokens"]
                    local_samples = batch_info["local_num_samples"]
                n_micros = max(len(micro_batches), 1)

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    overlong_prompt_mask = model_inputs.get("overlong_prompt_mask")
                    if overlong_prompt_mask is not None:
                        overlong_prompt_mask = overlong_prompt_mask.to(
                            device=response_mask.device, dtype=torch.bool
                        ).view(-1)
                        if (
                            overlong_prompt_mask.shape[0] == response_mask.shape[0]
                            and bool(overlong_prompt_mask.any())
                        ):
                            response_mask = response_mask.clone()
                            response_mask[overlong_prompt_mask] = 0
                            model_inputs["response_mask"] = response_mask
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs.get("advantages")

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)
                    self_distillation_mask = model_inputs.get("self_distillation_mask") if self_distillation_enabled else None
                    policy_fallback_mask = None
                    if self_distillation_enabled and self_distillation_mask is not None:
                        policy_fallback_mask = (self_distillation_mask <= 0.5).to(response_mask.dtype)
                        micro_batch_metrics["actor/policy_fallback_fraction"] = (
                            policy_fallback_mask.float().mean().detach().item()
                        )

                    micro_loss_mask = response_mask
                    if self_distillation_mask is not None:
                        micro_loss_mask = response_mask * self_distillation_mask.unsqueeze(1)
                    token_frac = float(micro_loss_mask.sum().item()) / local_tokens
                    sample_frac = response_mask.shape[0] / max(local_samples, 1)

                    teacher_regularization = self_distillation_cfg.get("teacher_regularization", "ema") if self_distillation_cfg else "ema"
                    teacher_model_source = self_distillation_cfg.get("teacher_model_source", "legacy") if self_distillation_cfg else "legacy"
                    use_trust_region_teacher = teacher_model_source == "legacy" and teacher_regularization == "trust-region"
                    if use_trust_region_teacher and self.use_fused_kernels:
                        raise ValueError("trust-region teacher requires disabling fused kernels to access logits.")
                    # all return: (bsz, response_length)
                    return_all_logps = bool(
                        self_distillation_enabled
                        and self_distillation_cfg.full_logit_distillation
                        and not self_distillation_cfg.distillation_topk
                    )
                    distill_topk = None
                    if self_distillation_enabled and self_distillation_cfg.full_logit_distillation:
                        distill_topk = self_distillation_cfg.distillation_topk
                    precomputed_topk_indices = None
                    if external_teacher_enabled:
                        precomputed_topk_indices = model_inputs.get("student_topk_indices")
                    teacher_outputs = None
                    teacher_inputs = None
                    if self_distillation_enabled and not external_teacher_enabled:
                        teacher_inputs = {
                            "responses": model_inputs["responses"],
                            "input_ids": model_inputs["teacher_input_ids"],
                            "attention_mask": model_inputs["teacher_attention_mask"],
                            "position_ids": model_inputs["teacher_position_ids"],
                            "response_start_idx": model_inputs["teacher_response_start_idx"],
                        }
                        if "teacher_multi_modal_inputs" in model_inputs:
                            teacher_inputs["multi_modal_inputs"] = model_inputs["teacher_multi_modal_inputs"]
                        teacher_model = self.teacher_module or self.actor_module
                        if use_trust_region_teacher and (
                            self.teacher_module is None or self.teacher_module is self.actor_module
                        ):
                            raise ValueError("trust-region teacher requires a separate teacher_module in the actor worker.")
                    # Surrogate backward reads Teacher deltas from the context object
                    # captured during the student forward. Fill that object first.
                    # Top-k still uses the student support: a no-grad student pass
                    # only collects indices, then the Teacher runs, then the grad pass.
                    student_distill_indices = precomputed_topk_indices
                    student_forward_start = time.perf_counter()
                    outputs = self._forward_micro_batch(
                        model_inputs,
                        temperature=temperature,
                        calculate_entropy=calculate_entropy,
                        return_all_logps=return_all_logps,
                        distill_topk=None if student_distill_indices is not None else distill_topk,
                        topk_indices=student_distill_indices,
                        return_samod_trajectory=ohisa_cfg.track,
                    )
                    if self_distillation_enabled:
                        student_forward_time = time.perf_counter() - student_forward_start
                        stage_wall_time_totals["timing_s/update_actor/student_forward"] += student_forward_time
                    if epoch == 0:
                        outputs.pop("samod_dreg_loss_per_sample", None)
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None
                    student_all_logps = outputs.get("all_logps") if return_all_logps else None
                    student_topk_logps = outputs.get("topk_logps") if distill_topk else None
                    student_topk_indices = (
                        precomputed_topk_indices if precomputed_topk_indices is not None else outputs.get("topk_indices")
                    ) if distill_topk else None

                    # for fully_async_policy
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    if self_distillation_enabled:
                        if external_teacher_enabled:
                            teacher_log_prob = log_prob.detach()
                            teacher_all_logps = None
                            teacher_topk_logps = model_inputs["teacher_topk_logps"].to(
                                device=student_topk_logps.device,
                                dtype=student_topk_logps.dtype,
                            )
                        else:
                            if teacher_outputs is None:
                                teacher_model = self.teacher_module or self.actor_module
                                with torch.no_grad():
                                    teacher_forward_start = time.perf_counter()
                                    teacher_outputs = self._forward_micro_batch(
                                        teacher_inputs,
                                        temperature=temperature,
                                        calculate_entropy=False,
                                        return_all_logps=return_all_logps,
                                        distill_topk=distill_topk,
                                        topk_indices=student_topk_indices,
                                        module=teacher_model,
                                        output_hidden_states=ohisa_needs_teacher_trajectory,
                                    )
                                    teacher_forward_time = time.perf_counter() - teacher_forward_start
                                stage_wall_time_totals["timing_s/update_actor/teacher_forward"] += teacher_forward_time
                            teacher_log_prob = teacher_outputs["log_probs"]
                            teacher_all_logps = teacher_outputs.get("all_logps") if return_all_logps else None
                            teacher_topk_logps = teacher_outputs.get("topk_logps") if distill_topk else None
                        if self_distillation_cfg.get("log_prob_dump_dir", None):
                            if distill_topk:
                                student_distill_log_probs = student_topk_logps
                                teacher_distill_log_probs = teacher_topk_logps
                                if self_distillation_cfg.distillation_add_tail:
                                    student_distill_log_probs = self._add_tail_bucket(student_distill_log_probs)
                                    teacher_distill_log_probs = self._add_tail_bucket(teacher_distill_log_probs)
                            else:
                                student_distill_log_probs = student_all_logps
                                teacher_distill_log_probs = teacher_all_logps

                            if student_distill_log_probs is None or teacher_distill_log_probs is None:
                                raise ValueError("Missing distillation log_probs for dump.")

                            loss_mask = response_mask
                            if self_distillation_mask is not None:
                                loss_mask = loss_mask * self_distillation_mask.unsqueeze(1)
                            valid_rows = loss_mask > 0
                            if valid_rows.any():
                                distill_dump_chunks.append(
                                    {
                                        "student_log_probs": student_distill_log_probs[valid_rows].detach().cpu().to(torch.float32),
                                        "teacher_log_probs": teacher_distill_log_probs[valid_rows].detach().cpu().to(torch.float32),
                                    }
                                )
                        loss_compute_start = time.perf_counter()
                        vopd_loss, vopd_metrics = compute_self_distillation_loss(
                            student_log_probs=log_prob,
                            teacher_log_probs=teacher_log_prob,
                            response_mask=response_mask,
                            self_distillation_config=self_distillation_cfg,
                            old_log_probs=old_log_prob,
                            student_all_log_probs=student_all_logps,
                            teacher_all_log_probs=teacher_all_logps,
                            student_topk_log_probs=student_topk_logps,
                            teacher_topk_log_probs=teacher_topk_logps,
                            self_distillation_mask=self_distillation_mask,
                            loss_agg_mode=loss_agg_mode,
                            rollout_is_weights=rollout_is_weights,
                            batch_num_tokens=batch_info["batch_num_tokens"],
                            global_batch_size=batch_info["global_batch_size"],
                            loss_scale_factor=self.config.global_batch_info.get("loss_scale_factor"),
                            dp_size=batch_info["dp_size"],
                            responses=model_inputs.get("responses", None),
                        )
                        loss_compute_time = time.perf_counter() - loss_compute_start
                        stage_wall_time_totals["timing_s/update_actor/loss_compute"] += loss_compute_time

                        vopd_metrics.pop("_per_sample_vopd_loss", None)
                        micro_batch_metrics.update(vopd_metrics)

                        if policy_fallback_mask is not None and policy_fallback_mask.any().item():
                            if advantages is None:
                                raise ValueError(
                                    "Mixed SDPO/GRPO fallback requires advantages for samples without teacher images."
                                )
                            policy_loss_fn = get_policy_loss_fn("vanilla")
                            grpo_loss, grpo_metrics = policy_loss_fn(
                                old_log_prob=old_log_prob,
                                log_prob=log_prob,
                                advantages=advantages,
                                response_mask=response_mask * policy_fallback_mask.unsqueeze(1),
                                loss_agg_mode=loss_agg_mode,
                                config=self.config,
                                rollout_is_weights=rollout_is_weights,
                            )
                            pg_loss = vopd_loss + grpo_loss * token_frac
                            micro_batch_metrics.update(
                                {f"actor/policy_fallback/{key.split('/', 1)[1]}": value for key, value in grpo_metrics.items()}
                            )
                        else:
                            grpo_loss = None
                            pg_loss = vopd_loss
                    else:
                        # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                        # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                        policy_loss_fn = get_policy_loss_fn(loss_mode)

                        # Compute policy loss (any function is expected to return 2 values)
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                        )
                        micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_with_aux = apply_samod_aux_losses(
                        pg_loss, outputs, micro_batch_metrics, metric_prefix="actor/", metric_as_metric=False
                    )
                    aux_loss = policy_with_aux - pg_loss
                    # VOPD is already micro_sum / global_tokens * dp_size. Summing those
                    # partials is the global mean, so it must not be divided by the
                    # micro-batch count. A local mean still needs its own fraction.
                    if self_distillation_enabled:
                        policy_loss = pg_loss + aux_loss * sample_frac
                    else:
                        policy_loss = pg_loss * token_frac + aux_loss * sample_frac
                    if ohisa_cfg.track:
                        traj = outputs.get("samod_trajectory")
                        if traj is None:
                            raise RuntimeError(
                                "OHiSA track but student forward did not return samod_trajectory."
                            )
                        if teacher_outputs is None:
                            raise RuntimeError(
                                "OHiSA track but teacher forward did not return outputs."
                            )
                        if ohisa_needs_teacher_trajectory and teacher_outputs.get("hidden_states") is None:
                            raise RuntimeError(
                                "This OHiSA mode requires the Teacher hidden trajectory."
                            )
                        teacher_model = self.teacher_module or self.actor_module
                        h_ref = traj["hidden_states_in"][0]
                        response_start_idx = model_inputs.get("response_start_idx")
                        if response_start_idx is None:
                            response_start_idx = model_inputs.get("teacher_response_start_idx")
                        attention_mask = model_inputs["attention_mask"]
                        response_token_mask = build_response_token_mask(
                            seq_len=int(attention_mask.shape[1]),
                            response_mask=response_mask,
                            response_start_idx=response_start_idx,
                        )
                        # rmpad expands each sample to left-aligned valid tokens. The
                        # trajectory stays on that [B, max_valid] layout, so the
                        # response mask has to follow the same packing.
                        if tuple(response_token_mask.shape) != tuple(h_ref.shape[:2]):
                            response_token_mask = project_token_mask_to_valid_span(
                                response_token_mask, attention_mask
                            )
                        if tuple(response_token_mask.shape) != tuple(h_ref.shape[:2]):
                            raise RuntimeError(
                                "OHiSA response mask "
                                f"{tuple(response_token_mask.shape)} does not match trajectory hidden "
                                f"{tuple(h_ref.shape[:2])}."
                            )
                        ohisa_weight = float(ohisa_cfg.loss_weight)
                        ctx = torch.enable_grad() if ohisa_cfg.enabled else torch.no_grad()
                        with ctx:
                            hidden_loss, hidden_metrics = compute_hidden_trajectory_loss(
                                traj,
                                teacher_model,
                                response_token_mask=response_token_mask,
                            )
                        if ohisa_cfg.enabled:
                            policy_loss = policy_loss + ohisa_weight * hidden_loss * token_frac
                        # Unweighted L_hidden; optim uses weight * L_hidden when weight!=0.
                        micro_batch_metrics["actor/ohisa_loss"] = float(hidden_loss.detach().item())
                        hidden_metrics.pop("_per_sample_ohisa_loss", None)
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff * token_frac

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef * token_frac
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * token_frac
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if int(response_mask.sum().item()) == 0:
                        # Fully skipped micro-batch (prompt longer than max_prompt_length).
                        # Keep a graph-connected zero so FSDP backward still runs.
                        policy_loss = log_prob.sum() * 0.0

                    loss, nonfinite_loss = self._sanitize_scalar_loss(policy_loss)
                    if nonfinite_loss:
                        micro_batch_metrics["actor/nonfinite_loss"] = 1.0
                    backward_start = time.perf_counter()
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()
                    if self_distillation_enabled:
                        backward_time = time.perf_counter() - backward_start
                        stage_wall_time_totals["timing_s/update_actor/backward"] += backward_time

                    if self_distillation_enabled:
                        metrics["actor/pg_loss"] += pg_loss.detach().item()
                        metrics["actor/vopd_loss"] += vopd_loss.detach().item()
                        metrics["actor/vopd_loss_weighted"] += vopd_loss.detach().item()
                        if grpo_loss is not None:
                            metrics["actor/grpo_loss"] += grpo_loss.detach().item() * token_frac
                    else:
                        metrics["actor/pg_loss"] += pg_loss.detach().item() * token_frac
                    for metric_key in (
                        "self_distillation/raw_jsd_token_mean",
                        "self_distillation/weighted_jsd_token_mean",
                        "actor/entropy",
                    ):
                        metric_value = micro_batch_metrics.get(metric_key)
                        if isinstance(metric_value, (int, float)):
                            micro_batch_metrics[metric_key] = metric_value * token_frac * n_micros
                    samod_score = micro_batch_metrics.get("actor/samod_dreg_loss")
                    if isinstance(samod_score, (int, float)):
                        micro_batch_metrics["actor/samod_dreg_loss"] = samod_score * sample_frac * n_micros
                    append_to_dict(metrics, micro_batch_metrics)

                optimizer_step_start = time.perf_counter()
                grad_norm = self._optimizer_step()
                nonfinite_grad_norm = bool(getattr(self, "_last_nonfinite_grad_norm", False))
                if self_distillation_enabled:
                    optimizer_step_time = time.perf_counter() - optimizer_step_start
                if not nonfinite_grad_norm and torch.isfinite(grad_norm).item():
                    did_update = True
                mini_batch_metrics = {}
                if not nonfinite_grad_norm:
                    mini_batch_metrics["actor/grad_norm"] = grad_norm.detach().item()
                if self_distillation_enabled:
                    stage_wall_time_totals["timing_s/update_actor/optimizer_step"] += optimizer_step_time
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        if self_distillation_enabled and distill_dump_chunks:
            self._dump_self_distillation_log_probs(
                meta_info=data.meta_info,
                self_distillation_cfg=self_distillation_cfg,
                dump_chunks=distill_dump_chunks,
            )
        if did_update:
            teacher_update_start = time.perf_counter()
            self._update_teacher()
            if self_distillation_enabled:
                stage_wall_time_totals["timing_s/update_actor/teacher_ema_update"] += (
                    time.perf_counter() - teacher_update_start
                )
        if self_distillation_enabled:
            for key, total_time in stage_wall_time_totals.items():
                metrics[key] = Metric(aggregation=AggregationType.MAX, value=total_time)
        metric_keys_to_keep_unreduced = set(stage_wall_time_totals.keys()) if stage_wall_time_totals is not None else set()
        local_metrics_to_reduce = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, list) or (isinstance(value, Metric) and key not in metric_keys_to_keep_unreduced)
        }
        if local_metrics_to_reduce:
            reduced_local_metrics = reduce_metrics(local_metrics_to_reduce)
            metrics.update(reduced_local_metrics)
        # The worker must not advance the LR schedule when every optimizer step
        # was skipped because of non-finite gradients.
        self._last_update_succeeded = did_update
        return metrics
