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

import functools
import logging
import os
from dataclasses import dataclass
from typing import Optional

import torch
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLForConditionalGeneration,
)

from verl.utils.transformers_compat import unpack_visual_output

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def get_rope_index(
    processor,
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Gets the position ids for Qwen3-VL, it should be generated before sharding the sequence.
    The batch dim has been removed and the input_ids should be a 1D tensor representing a single example.
    https://github.com/huggingface/transformers/blob/v4.57.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py#L916
    """
    spatial_merge_size = processor.image_processor.merge_size
    image_token_id = processor.image_token_id
    video_token_id = processor.video_token_id
    vision_start_token_id = processor.vision_start_token_id

    # Since we use timestamps to separate videos,
    # like <t1> <vision_start> <frame1> <vision_end> <t2> <vision_start> <frame2> <vision_end>,
    # the video_grid_thw should also be split
    if video_grid_thw is not None:
        video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
        video_grid_thw[:, 0] = 1

    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.ones(3, input_ids.shape[0], dtype=input_ids.dtype, device=input_ids.device)
        image_index, video_index = 0, 0
        attention_mask = attention_mask.to(input_ids.device)
        input_ids = input_ids[attention_mask == 1]
        image_nums, video_nums = 0, 0
        vision_start_indices = torch.argwhere(input_ids == vision_start_token_id)
        vision_tokens = input_ids[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        input_tokens = input_ids.tolist()
        llm_pos_ids_list: list = []
        st = 0
        remain_images, remain_videos = image_nums, video_nums
        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                video_index += 1
                remain_videos -= 1
                ed = ed_video

            llm_grid_t, llm_grid_h, llm_grid_w = (
                t.item(),
                h.item() // spatial_merge_size,
                w.item() // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

            # t_index is always 0 because llm_grid_t is always 1
            # (we use timestamps to encode the temporal information for videos)
            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w).flatten()
            h_index = torch.arange(llm_grid_h).view(1, -1, 1).expand(llm_grid_t, -1, llm_grid_w).flatten()
            w_index = torch.arange(llm_grid_w).view(1, 1, -1).expand(llm_grid_t, llm_grid_h, -1).flatten()
            llm_pos_ids_list.append(torch.stack([t_index, h_index, w_index]) + text_len + st_idx)
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx)

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        position_ids[..., attention_mask == 1] = llm_positions.to(position_ids.device)
    else:
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            position_ids = position_ids.unsqueeze(0).expand(3, -1).to(attention_mask.device)
        else:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1).expand(3, -1)

    return position_ids


def _get_input_embeds(
    model: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
):
    inputs_embeds = model.get_input_embeddings()(input_ids)
    image_mask, video_mask = None, None
    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.dtype)
        image_embeds, deepstack_image_embeds = unpack_visual_output(model.visual(pixel_values, grid_thw=image_grid_thw))
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
        pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
        video_embeds, deepstack_video_embeds = unpack_visual_output(
            model.visual(pixel_values_videos, grid_thw=video_grid_thw)
        )
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

    visual_pos_masks = None
    deepstack_visual_embeds = None
    if image_mask is not None and video_mask is not None:
        # aggregate visual_pos_masks and deepstack_visual_embeds
        image_mask = image_mask[..., 0]
        video_mask = video_mask[..., 0]
        visual_pos_masks = image_mask | video_mask
        deepstack_visual_embeds = []
        image_mask_joint = image_mask[visual_pos_masks]
        video_mask_joint = video_mask[visual_pos_masks]
        for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds, strict=False):
            embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
            embed_joint[image_mask_joint, :] = img_embed
            embed_joint[video_mask_joint, :] = vid_embed
            deepstack_visual_embeds.append(embed_joint)
    elif image_mask is not None:
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        deepstack_visual_embeds = deepstack_image_embeds
    elif video_mask is not None:
        video_mask = video_mask[..., 0]
        visual_pos_masks = video_mask
        deepstack_visual_embeds = deepstack_video_embeds

    if pixel_values is None and pixel_values_videos is None:
        config = model.config.vision_config
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        pixel_values = torch.zeros((16, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        image_embeds, dummy_deepstack_image_embeds = unpack_visual_output(
            model.visual(pixel_values, grid_thw=image_grid_thw)
        )
        inputs_embeds = inputs_embeds + 0.0 * image_embeds.mean()
        for emb in dummy_deepstack_image_embeds or []:
            inputs_embeds = inputs_embeds + 0.0 * emb.mean()

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "visual_pos_masks": visual_pos_masks,
        "deepstack_visual_embeds": deepstack_visual_embeds,
    }


@dataclass
class Qwen3VLCausalLMOutputForPPO(Qwen3VLCausalLMOutputWithPast):
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    logits_are_response_window: Optional[bool] = None


def qwen3_vl_base_forward(
    self: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    **kwargs,
):
    # Actor rmpad packs to [1, nnz]. Dense Qwen3-VL flash-attn would otherwise treat
    # the pack as one sequence and leak across samples, while the SAMoD student
    # unflattens via samod_cu_seqlens — that mismatch shows up as huge init vopd.
    samod_cu_seqlens = kwargs.pop("samod_cu_seqlens", None)
    input_kwargs = _get_input_embeds(
        self, input_ids, attention_mask, pixel_values, pixel_values_videos, image_grid_thw, video_grid_thw
    )  # avoid lora module having multiple keyword arguments
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
        visual_pos_masks = kwargs.get("visual_pos_masks", None)
        if visual_pos_masks is not None:
            visual_pos_masks, _ = _varlen_to_padded_batch(visual_pos_masks, samod_cu_seqlens)
            kwargs["visual_pos_masks"] = visual_pos_masks.to(dtype=torch.bool)

    for key in (
        "mm_token_type_ids",
        "rmpad_indices",
        "padded_seqlen",
        "padded_batch_size",
        "logits_rmpad_indices",
        "logits_batch_size",
        "logits_seqlen",
        "logits_response_length",
        "logits_response_start_idx",
        "return_samod_metrics",
        "return_samod_trajectory",
        "distill_topk_indices",
    ):
        kwargs.pop(key, None)

    outputs = self.language_model(
        input_ids=None,
        **kwargs,
    )

    if samod_cu_seqlens is not None and varlen_valid_mask is not None:
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            outputs.last_hidden_state = _padded_batch_to_varlen(
                outputs.last_hidden_state, samod_cu_seqlens, varlen_valid_mask
            )
        elif isinstance(outputs, tuple) and len(outputs) > 0 and torch.is_tensor(outputs[0]):
            flat = _padded_batch_to_varlen(outputs[0], samod_cu_seqlens, varlen_valid_mask)
            outputs = (flat,) + outputs[1:]
    return outputs


def forward_with_normal_backend(
    self: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    samod_cu_seqlens: Optional[torch.Tensor] = None,
    **kwargs,
) -> "Qwen3VLCausalLMOutputForPPO":
    # Pop sparse-only distill kwargs so dense backbone does not see them.
    # Keep samod_cu_seqlens: dense qwen3_vl_base_forward uses it to unflatten rmpad.
    kwargs.pop("distill_topk_indices", None)
    kwargs.pop("return_samod_trajectory", None)
    skip_logits = bool(kwargs.pop("skip_logits", False))
    logits_response_length = kwargs.pop("logits_response_length", None)
    logits_response_start_idx = kwargs.pop("logits_response_start_idx", None)
    logits_rmpad_indices = kwargs.pop("logits_rmpad_indices", None)
    logits_batch_size = kwargs.pop("logits_batch_size", None)
    logits_seqlen = kwargs.pop("logits_seqlen", None)
    if samod_cu_seqlens is not None:
        kwargs["samod_cu_seqlens"] = samod_cu_seqlens
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

    return Qwen3VLCausalLMOutputForPPO(
        logits=logits,
        hidden_states=outputs.hidden_states,
        last_hidden_state=hidden_states,
        logits_are_response_window=logits_are_response_window,
    )


def forward_with_torch_backend(
    self: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3VLCausalLMOutputForPPO":
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]

    # Loss calculations
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
    return Qwen3VLCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )


def forward_with_triton_backend(
    self: "Qwen3VLForConditionalGeneration",
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    **kwargs,
) -> "Qwen3VLCausalLMOutputForPPO":
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]

    # Loss calculations
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
    return Qwen3VLCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
    )


def patch_qwen3_vl_moe_sparse_moe_block_forward():
    """
    Monkey patch to fix a bug in transformers 4.57.3 where Qwen3VLMoeTextSparseMoeBlock.forward
    incorrectly uses torch.zeros_like(hidden_states) instead of torch.zeros_like(router_logits)
    when creating router_weights (line 148 in modeling_qwen3_vl_moe.py).

    This is a minimal fix that only changes the problematic line while keeping the rest of the
    original implementation intact.
    """
    try:
        from transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe import Qwen3VLMoeTextSparseMoeBlock
    except ImportError:
        # Model not available, skip patching
        return

    # Store the original forward method for reference
    original_forward = Qwen3VLMoeTextSparseMoeBlock.forward

    @functools.wraps(original_forward)
    def patched_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = hidden_states.shape[0]
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        router_logits = self.gate(hidden_states)
        routing_weights = torch.nn.functional.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, router_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        # BUG FIX: Original code incorrectly uses hidden_states here, should use router_logits
        routing_weights = routing_weights.to(router_logits.dtype)
        router_weights = torch.zeros_like(router_logits).scatter_(1, router_indices, routing_weights)
        hidden_states = hidden_states.reshape(batch_size, -1, self.hidden_size)
        routed_out = self.experts(hidden_states, router_weights, router_indices)
        return routed_out

    # Apply the patch
    Qwen3VLMoeTextSparseMoeBlock.forward = patched_forward
    logger.info("Monkey patched Qwen3VLMoeTextSparseMoeBlock.forward to fix router_weights bug")
