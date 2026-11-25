# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5ForConditionalGeneration,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def fast_pos_embed_interpolate(self, grid_thw):
    grid_thw_list = grid_thw.tolist()
    grid_ts = [row[0] for row in grid_thw_list]
    grid_hs = [row[1] for row in grid_thw_list]
    grid_ws = [row[2] for row in grid_thw_list]
    device = grid_thw.device

    idx_list = [[] for _ in range(4)]
    weight_list = [[] for _ in range(4)]

    for t, h, w in grid_thw_list:
        h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)

        h_idxs_floor = h_idxs.int()
        w_idxs_floor = w_idxs.int()
        h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
        w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

        dh = h_idxs - h_idxs_floor
        dw = w_idxs - w_idxs_floor

        base_h = h_idxs_floor * self.num_grid_per_side
        base_h_ceil = h_idxs_ceil * self.num_grid_per_side

        indices = [
            (base_h[None].T + w_idxs_floor[None]).flatten(),
            (base_h[None].T + w_idxs_ceil[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
            (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
        ]

        weights = [
            ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
            ((1 - dh)[None].T * dw[None]).flatten(),
            (dh[None].T * (1 - dw)[None]).flatten(),
            (dh[None].T * dw[None]).flatten(),
        ]

        for i in range(4):
            idx_list[i].extend(indices[i].tolist())
            weight_list[i].extend(weights[i].tolist())

    idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
    weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
    pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
    patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

    patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws, strict=False)])

    patch_pos_embeds_permute = []
    merge_size = self.config.spatial_merge_size
    for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws, strict=False):
        pos_embed = pos_embed.repeat(t, 1)
        pos_embed = (
            pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        patch_pos_embeds_permute.append(pos_embed)
    patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
    return patch_pos_embeds


def _get_input_embeds(
    model: "Qwen3_5CausalLMOutputWithPast",
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    if inputs_embeds is None:
        if input_ids is None:
            raise ValueError("Either input_ids or inputs_embeds must be provided.")
        inputs_embeds = model.get_input_embeddings()(input_ids)
    if pixel_values is not None:
        if input_ids is None:
            raise ValueError("input_ids is required when pixel_values are provided.")
        pixel_values = pixel_values.type(model.visual.dtype)
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw).pooler_output
        n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        mask = input_ids == model.config.image_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        image_mask = mask_expanded.to(inputs_embeds.device)

        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

    if pixel_values_videos is not None:
        if input_ids is None:
            raise ValueError("input_ids is required when pixel_values_videos are provided.")
        pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
        video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw).pooler_output
        n_video_tokens = (input_ids == model.config.video_token_id).sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )

        mask = input_ids == model.config.video_token_id
        mask_unsqueezed = mask.unsqueeze(-1)
        mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
        video_mask = mask_expanded.to(inputs_embeds.device)

        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

    if pixel_values is None and pixel_values_videos is None:
        config = model.config.vision_config
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        pixel_values = torch.zeros((16, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw).pooler_output
        inputs_embeds = inputs_embeds + 0.0 * image_embeds.mean()

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    return {"inputs_embeds": inputs_embeds, "attention_mask": attention_mask}


def qwen3_5_base_forward(
    self: "Qwen3_5ForConditionalGeneration",
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs,
):
    # Actor rmpad packs to [1, nnz]. Qwen3.5 linear_attention needs per-sample batch
    # boundaries; restore [B, max_len] when cu_seqlens are provided (same idea as SAMoD).
    samod_cu_seqlens = kwargs.pop("samod_cu_seqlens", None)
    input_kwargs = _get_input_embeds(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
    )
    kwargs.update(input_kwargs)

    varlen_valid_mask = None
    if samod_cu_seqlens is not None:
        from verl.models.qwen3_5_samod.modeling_qwen3_5_samod import (
            _normalize_cu_seqlens,
            _padded_batch_to_varlen,
            _unflatten_mrope_position_ids,
            _varlen_to_padded_batch,
        )

        inputs_embeds = kwargs["inputs_embeds"]
        if inputs_embeds.ndim < 2 or inputs_embeds.shape[0] != 1:
            raise ValueError(
                "samod_cu_seqlens requires flat rmpad inputs with batch dim 1, "
                f"got inputs_embeds.shape={tuple(inputs_embeds.shape)}"
            )
        samod_cu_seqlens = _normalize_cu_seqlens(samod_cu_seqlens, int(inputs_embeds.shape[1]), inputs_embeds.device)
        inputs_embeds, varlen_valid_mask = _varlen_to_padded_batch(inputs_embeds, samod_cu_seqlens)
        kwargs["inputs_embeds"] = inputs_embeds
        attn = kwargs.get("attention_mask", None)
        if attn is None:
            kwargs["attention_mask"] = varlen_valid_mask.to(dtype=torch.long)
        else:
            kwargs["attention_mask"], _ = _varlen_to_padded_batch(attn, samod_cu_seqlens)
        position_ids = kwargs.get("position_ids", None)
        if position_ids is not None:
            kwargs["position_ids"] = _unflatten_mrope_position_ids(position_ids, samod_cu_seqlens)

    outputs = self.language_model(
        input_ids=None,
        **kwargs,
    )

    if samod_cu_seqlens is not None and varlen_valid_mask is not None:
        # ModelOutput / tuple: restore flat [1, nnz, H] for actor pad_input / response lm_head.
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            flat = _padded_batch_to_varlen(outputs.last_hidden_state, samod_cu_seqlens, varlen_valid_mask)
            outputs.last_hidden_state = flat
        elif isinstance(outputs, tuple) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
            flat = _padded_batch_to_varlen(outputs[0], samod_cu_seqlens, varlen_valid_mask)
            outputs = (flat,) + outputs[1:]
    return outputs


@dataclass
class Qwen3_5CausalLMOutputForPPO(Qwen3_5CausalLMOutputWithPast):
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    logits_are_response_window: Optional[bool] = None


def forward_with_normal_backend(
    self: "Qwen3_5ForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    # Named so transformers.generate kwargs validation accepts HFRollout's mm_token_type_ids.
    mm_token_type_ids: Optional[torch.IntTensor] = None,
    **kwargs,
) -> "Qwen3_5CausalLMOutputForPPO":
    skip_logits = bool(kwargs.pop("skip_logits", False))
    logits_response_length = kwargs.pop("logits_response_length", None)
    logits_response_start_idx = kwargs.pop("logits_response_start_idx", None)
    logits_rmpad_indices = kwargs.pop("logits_rmpad_indices", None)
    logits_batch_size = kwargs.pop("logits_batch_size", None)
    logits_seqlen = kwargs.pop("logits_seqlen", None)
    # Pop sparse-only distill kwargs so dense backbone does not see them.
    # Keep samod_cu_seqlens: dense qwen3_5_base_forward uses it to unflatten rmpad
    # [1,nnz] → [B,max_len] (Qwen3.5 linear_attention needs per-sample boundaries).
    kwargs.pop("distill_topk_indices", None)
    kwargs.pop("return_samod_trajectory", None)
    if mm_token_type_ids is not None:
        kwargs["mm_token_type_ids"] = mm_token_type_ids
    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]
    logits = None
    logits_are_response_window = None
    if not skip_logits:
        if logits_rmpad_indices is not None and logits_response_length is not None:
            from verl.models.transformers.qwen3_5_samod import _logits_from_rmpad_hidden

            if logits_batch_size is None or logits_seqlen is None:
                raise ValueError("logits_rmpad_indices requires logits_batch_size and logits_seqlen")
            logits = _logits_from_rmpad_hidden(
                hidden_states,
                self.lm_head,
                indices=logits_rmpad_indices,
                batch_size=int(logits_batch_size),
                seqlen=int(logits_seqlen),
                response_length=int(logits_response_length),
                response_start_idx=logits_response_start_idx,
            )
            logits_are_response_window = True
        elif logits_response_length is not None:
            from verl.models.transformers.qwen3_5_samod import _hidden_for_response_logits

            hidden_for_logits = _hidden_for_response_logits(
                hidden_states,
                response_length=int(logits_response_length),
                response_start_idx=logits_response_start_idx,
            )
            logits = self.lm_head(hidden_for_logits)
            logits_are_response_window = True
        else:
            logits = self.lm_head(hidden_states)
    return Qwen3_5CausalLMOutputForPPO(
        logits=logits,
        hidden_states=outputs.hidden_states,
        last_hidden_state=hidden_states,
        logits_are_response_window=logits_are_response_window,
    )


def forward_with_torch_backend(
    self: "Qwen3_5ForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    mm_token_type_ids: Optional[torch.IntTensor] = None,
    **kwargs,
) -> "Qwen3_5CausalLMOutputForPPO":
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    if mm_token_type_ids is not None:
        kwargs["mm_token_type_ids"] = mm_token_type_ids
    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]

    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")

    fused_linear_for_ppo = FusedLinearForPPO()
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
    )
    return Qwen3_5CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )


def forward_with_triton_backend(
    self: "Qwen3_5ForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    mm_token_type_ids: Optional[torch.IntTensor] = None,
    **kwargs,
) -> "Qwen3_5CausalLMOutputForPPO":
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    if mm_token_type_ids is not None:
        kwargs["mm_token_type_ids"] = mm_token_type_ids
    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]

    if labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_triton_backend, either labels or input_ids must be provided.")

    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        self.lm_head.weight,
        rolled_labels,
        temperature,
        "none",
    )
    return Qwen3_5CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )
