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

import math
import re
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, BaseModelOutputWithPooling, CausalLMOutputWithPast
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig, Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3_5GatedDeltaNet,
    Qwen3_5Model,
    Qwen3_5ModelOutputWithPast,
    Qwen3_5PreTrainedModel,
    Qwen3_5RMSNorm,
    Qwen3_5TextModel,
    Qwen3_5TextRotaryEmbedding,
    Qwen3_5VisionBlock,
    Qwen3_5VisionModel,
)
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs


def _get_text_config(config):
    return getattr(config, "text_config", config)


class _LayerCacheView:
    """Expose one attention layer's cache size to create_causal_mask."""

    def __init__(self, cache: Cache, layer_idx: int):
        self.cache = cache
        self.layer_idx = layer_idx

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.cache.get_seq_length(self.layer_idx)

    def get_mask_sizes(self, query_length: int, layer_idx: int = 0) -> tuple[int, int]:
        return self.cache.get_mask_sizes(query_length, self.layer_idx)

    def __getattr__(self, name):
        return getattr(self.cache, name)


class SAMoDRaggedCache(DynamicCache):
    """Single-sample SAMoD cache with compact per-layer prefill state.

    SAMoD only sparsifies vision tokens during prefill. Each full-attention layer
    therefore stores exactly its kept prompt tokens, while decode is dense and
    appends every generated token. No rectangular padding or ``layer_valid`` mask
    is needed for batch size one.
    """

    def __init__(self, config):
        super().__init__(config=config)
        self._samod_meta = {"logical_seen": 0, "layer_valid": {}}


def _ensure_samod_cache_meta(cache: Optional[Cache]) -> Optional[dict]:
    """Attach SAMoD sparse-KV metadata onto a DynamicCache (not registered as buffers).

    Fields:
      - logical_seen: original (unpacked) token count for RoPE / generate positions
      - layer_valid: layer_idx -> Bool[B, physical_len] valid entries in compact KV
    Prefill writes compact KV (kept vision ∪ text) with per-layer lengths; decode
    appends new text tokens and must mask with ``layer_valid``, not the logical
    prompt attention_mask (lengths differ when vision tokens were dropped).
    """
    if cache is None:
        return None
    meta = getattr(cache, "_samod_meta", None)
    if meta is None:
        meta = {"logical_seen": 0, "layer_valid": {}}
        setattr(cache, "_samod_meta", meta)
    return meta


def _samod_cache_is_empty(cache: Optional[Cache]) -> bool:
    if cache is None:
        return True
    try:
        return int(cache.get_seq_length()) == 0
    except (AttributeError, ValueError, TypeError):
        return True


def _attention_mask_has_padding(attention_mask: Optional[torch.Tensor]) -> bool:
    return attention_mask is not None and attention_mask.ndim == 2 and not bool(torch.all(attention_mask))


def _forced_keep_mask(vision_mask: torch.Tensor, vision_token_retention: float) -> torch.Tensor:
    """Fixed vision keep mask. The router still runs; only the keep decision is replaced."""
    mask = vision_mask.to(dtype=torch.bool)
    rate = float(vision_token_retention)
    if rate <= 0.0:
        return torch.zeros_like(mask)
    if rate >= 1.0:
        return mask
    counts = mask.sum(dim=-1)
    keep_count = torch.ceil(counts.to(dtype=torch.float32) * rate).to(dtype=torch.long)
    keep_count = torch.where(counts > 0, keep_count.clamp(min=1), keep_count)
    rank = mask.to(dtype=torch.long).cumsum(dim=-1)
    return mask & (rank <= keep_count.unsqueeze(-1))


def _ohisa_sequence_keep_mask(
    hidden_states: torch.Tensor,
    attention_mask_2d: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Every valid token in the sequence. Padding is excluded when a 2D mask exists."""
    if attention_mask_2d is not None:
        return attention_mask_2d.to(device=hidden_states.device, dtype=torch.bool)
    return torch.ones(hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device)


def _create_layer_causal_mask(
    config: Qwen3_5TextConfig,
    inputs_embeds: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache],
    position_ids: Optional[torch.Tensor],
    layer_idx: int,
):
    cache_for_mask = _LayerCacheView(past_key_values, layer_idx) if past_key_values is not None else None
    return create_causal_mask(
        config=config,
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        past_key_values=cache_for_mask,
        position_ids=position_ids,
    )


def _build_packed_indices(mask: torch.Tensor, max_length: int) -> torch.Tensor:
    batch_size, seq_length = mask.shape
    if max_length == 0:
        return torch.empty((batch_size, 0), dtype=torch.long, device=mask.device)

    ranks = mask.to(dtype=torch.long).cumsum(dim=1) - 1
    source_positions = torch.arange(seq_length, dtype=torch.long, device=mask.device).unsqueeze(0).expand(batch_size, -1)
    batch_positions = torch.arange(batch_size, dtype=torch.long, device=mask.device).unsqueeze(1).expand_as(mask)
    packed_indices = source_positions.new_zeros((batch_size, max_length))
    packed_indices[batch_positions[mask], ranks[mask]] = source_positions[mask]
    return packed_indices


def _build_vision_pack_scaffold(
    vision_token_mask: torch.Tensor,
    segment_ids: Optional[torch.Tensor] = None,
) -> dict:
    """Shared per-forward vision pack metadata (lengths / indices / pack_mask)."""
    mask = vision_token_mask.to(dtype=torch.bool)
    segment_max_id = None
    segment_reduction = None
    if segment_ids is not None:
        segment_ids = segment_ids.to(device=mask.device, dtype=torch.long).masked_fill(~mask, 0)
        unlabeled = mask & (segment_ids == 0)
        if bool(unlabeled.any()):
            fallback = _vision_segment_ids(unlabeled)
            segment_ids = torch.where(
                unlabeled,
                fallback + int(segment_ids.max().item()),
                segment_ids,
            )
        segment_max_id = int(segment_ids.max().item()) if segment_ids.numel() else 0
        if segment_max_id > 0:
            batch = int(segment_ids.shape[0])
            offsets = torch.arange(batch, device=mask.device).unsqueeze(1) * (segment_max_id + 1)
            flat_ids = (segment_ids + offsets).reshape(-1)
            valid = (segment_ids.reshape(-1) > 0).to(dtype=torch.float32)
            n_bins = batch * (segment_max_id + 1)
            counts = torch.zeros(n_bins, device=mask.device, dtype=torch.float32)
            counts.scatter_add_(0, flat_ids, valid)
            segment_reduction = {
                "flat_ids": flat_ids,
                "valid": valid,
                "counts_clamped": counts.clamp_min(1.0),
                "n_bins": n_bins,
            }
    batch = int(mask.shape[0])
    lengths = mask.sum(dim=1).to(dtype=torch.long)
    max_len = int(lengths.max().item()) if batch > 0 else 0
    if max_len == 0:
        return {
            "mask": mask,
            "lengths": lengths,
            "max_len": 0,
            "pack_mask": None,
            "indices": None,
            "batch_ids": None,
            "segment_ids": segment_ids,
            "segment_max_id": segment_max_id,
            "segment_reduction": segment_reduction,
        }
    pack_mask = torch.arange(max_len, device=mask.device).unsqueeze(0) < lengths.unsqueeze(1)
    indices = _build_packed_indices(mask, max_len)
    batch_ids = torch.arange(batch, device=mask.device).unsqueeze(1).expand_as(indices)
    return {
        "mask": mask,
        "lengths": lengths,
        "max_len": max_len,
        "pack_mask": pack_mask,
        "indices": indices,
        "batch_ids": batch_ids,
        "segment_ids": segment_ids,
        "segment_max_id": segment_max_id,
        "segment_reduction": segment_reduction,
    }


def _append_visual_grid_segments(
    segment_ids: torch.Tensor,
    modality_mask: torch.Tensor,
    grid_thw: Optional[torch.Tensor],
    spatial_merge_size: int,
) -> None:
    """Label each image, and each whole video, as one reduction segment."""
    if grid_thw is None:
        return
    mask = modality_mask[..., 0] if modality_mask.ndim == 3 else modality_mask
    flat_positions = mask.to(dtype=torch.bool).reshape(-1).nonzero(as_tuple=False).flatten()
    grids = grid_thw.detach().to(device="cpu", dtype=torch.long).reshape(-1, 3)
    merge_area = int(spatial_merge_size) ** 2
    label_parts: list[torch.Tensor] = []
    next_id = int(segment_ids.max().item()) + 1
    for t, h, w in grids.tolist():
        n_tokens = int(t) * (int(h * w) // merge_area)
        label_parts.append(torch.full((n_tokens,), next_id, dtype=torch.long))
        next_id += 1
    labels = torch.cat(label_parts) if label_parts else torch.empty(0, dtype=torch.long)
    if int(labels.numel()) != int(flat_positions.numel()):
        raise ValueError(
            "Visual placeholder/grid token count mismatch while building SAMoD segments: "
            f"mask={flat_positions.numel()}, grids={int(labels.numel())}"
        )
    segment_ids.reshape(-1)[flat_positions] = labels.to(device=segment_ids.device, dtype=torch.long)


def _expand_gather_indices(indices: torch.Tensor, trailing_ndim: int) -> torch.Tensor:
    gather_indices = indices
    for _ in range(trailing_ndim):
        gather_indices = gather_indices.unsqueeze(-1)
    return gather_indices


def _per_token_rms(
    hidden: torch.Tensor,
    member_mask: torch.Tensor,
    eps: float = 1e-12,
    *,
    assume_nonempty: bool = False,
) -> torch.Tensor:
    """Per-token RMS of ``member_mask`` only.

    Padding and every other token are a constant radicand, so they are not in
    the mean and get no gradient. An all-zero member is floored so ``sqrt(0)``
    cannot NaN the backward. A real token's mean square is far above the floor.
    """
    mask = member_mask.to(device=hidden.device, dtype=torch.bool)
    # Non-members stay 1. Their squares never enter the image mean.
    radicand = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.float32)
    if assume_nonempty or bool(mask.any()):
        radicand[mask] = hidden[mask].float().pow(2).mean(dim=-1)
    return radicand.clamp_min(eps).sqrt()


def _packed_vision_mean_rms_scale(
    packed_vision: torch.Tensor,
    hidden_states: torch.Tensor,
    vision_mask: torch.Tensor,
    indices: torch.Tensor,
    pack_mask: torch.Tensor,
    vision_pack: Optional[dict],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-pack-slot image/segment mean RMS (float32), matching pack-normalize.

    Single-image rows reduce on the pack only; multi-segment rows scatter onto the
    full sequence so ``scatter_add`` order stays bitwise-identical to
    ``_router_normalize_by_image``.
    """
    vis_f = packed_vision.float()
    rms_pack = vis_f.pow(2).mean(dim=-1).clamp_min(1e-12).sqrt()
    segment_ids = None if vision_pack is None else vision_pack.get("segment_ids")
    if segment_ids is None:
        valid = pack_mask.to(dtype=rms_pack.dtype)
        denom = valid.sum(dim=-1, keepdim=True).clamp_min(1)
        return ((rms_pack * valid).sum(dim=-1, keepdim=True) / denom).clamp_min(eps)

    batch_ids = None if vision_pack is None else vision_pack.get("batch_ids")
    if batch_ids is None:
        batch_ids = torch.arange(
            packed_vision.shape[0], device=packed_vision.device
        ).unsqueeze(1).expand_as(indices)
    flat = pack_mask
    token_rms = torch.ones(
        hidden_states.shape[:2], device=hidden_states.device, dtype=torch.float32
    )
    token_rms[batch_ids[flat], indices[flat]] = rms_pack[flat]
    mean_rms = _per_image_threshold(
        token_rms,
        vision_mask,
        segment_ids=segment_ids,
        segment_max_id=None if vision_pack is None else vision_pack.get("segment_max_id"),
        segment_reduction=None if vision_pack is None else vision_pack.get("segment_reduction"),
    ).clamp_min(eps)
    return _gather_by_indices(mean_rms, indices, pack_mask).clamp_min(eps)


def _normalize_packed_vision_tokens(
    packed_vision: torch.Tensor,
    hidden_states: torch.Tensor,
    vision_mask: torch.Tensor,
    indices: torch.Tensor,
    pack_mask: torch.Tensor,
    vision_pack: Optional[dict],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Image-RMS normalize already-packed vision tokens (eval / no-grad path)."""
    with torch.no_grad():
        scale = _packed_vision_mean_rms_scale(
            packed_vision,
            hidden_states,
            vision_mask,
            indices,
            pack_mask,
            vision_pack,
            eps=eps,
        )
        out = (packed_vision.float() / scale.unsqueeze(-1)).to(dtype=packed_vision.dtype)
        return out.masked_fill(~pack_mask.unsqueeze(-1), 0)


def _router_linear_packed_rms_normalized(
    packed_vision: torch.Tensor,
    hidden_states: torch.Tensor,
    vision_mask: torch.Tensor,
    indices: torch.Tensor,
    pack_mask: torch.Tensor,
    vision_pack: Optional[dict],
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    """Eval router on packed vision: ``(W @ h) / mean_rms + b`` (float32 matmul).

    Same ``mean_rms`` reduction as ``_normalize_packed_vision_tokens``. Equivalent
    to ``Linear(h / mean_rms)`` with bias applied after the division. Bitwise-stable
    across call sites via one shared float32 recipe (not via bf16 normalize→Linear,
    which is a different rounding association).
    """
    with torch.no_grad():
        scale = _packed_vision_mean_rms_scale(
            packed_vision,
            hidden_states,
            vision_mask,
            indices,
            pack_mask,
            vision_pack,
            eps=eps,
        )
        pre = F.linear(packed_vision.float(), weight.float(), None)
        out = pre / scale.unsqueeze(-1)
        if bias is not None:
            out = out + bias.float()
        out = out.to(dtype=packed_vision.dtype)
        return out.masked_fill(~pack_mask.unsqueeze(-1), 0)


def _router_normalize_by_image(
    hidden: torch.Tensor,
    vision_mask: torch.Tensor,
    document_breaks: Optional[torch.Tensor] = None,
    segment_ids: Optional[torch.Tensor] = None,
    segment_max_id: Optional[int] = None,
    segment_reduction: Optional[dict] = None,
    assume_nonempty: bool = False,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Divide vision tokens by their image's mean per-token RMS. Scale is detached.

    Text and padding stay unchanged, and the residual stream is not written.
    """
    mask = vision_mask.to(device=hidden.device, dtype=torch.bool)
    if not assume_nonempty and not bool(mask.any()):
        return hidden
    grad_enabled = torch.is_grad_enabled()
    vision_norm = None
    with torch.no_grad():
        if grad_enabled:
            token_rms = _per_token_rms(hidden, mask, assume_nonempty=assume_nonempty)
            vision_hidden = None
        else:
            vision_hidden = hidden[mask].float()
            token_rms = torch.ones(hidden.shape[:2], device=hidden.device, dtype=torch.float32)
            token_rms[mask] = vision_hidden.pow(2).mean(dim=-1).clamp_min(1e-12).sqrt()
        mean_rms = _per_image_threshold(
            token_rms,
            mask,
            document_breaks,
            segment_ids=segment_ids,
            segment_max_id=segment_max_id,
            segment_reduction=segment_reduction,
        ).clamp_min(eps)
        if vision_hidden is not None:
            safe_scale = mean_rms.masked_fill(~mask, 1)
            vision_norm = (vision_hidden / safe_scale[mask].unsqueeze(-1)).to(dtype=hidden.dtype)
    if vision_norm is not None:
        out = hidden.clone()
        out[mask] = vision_norm
        return out
    safe_scale = mean_rms.masked_fill(~mask, 1).unsqueeze(-1)
    normed = hidden.float() / safe_scale
    return torch.where(mask.unsqueeze(-1), normed.to(dtype=hidden.dtype), hidden)


def _gather_by_indices(
    values: torch.Tensor,
    indices: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if values.ndim == 1:
        gathered = values.gather(0, indices.reshape(-1)).view(indices.shape)
        return gathered.masked_fill(~valid_mask, 0) if valid_mask is not None else gathered

    trailing = values.shape[2:]
    gather_indices = _expand_gather_indices(indices, len(trailing)).expand(*indices.shape, *trailing)
    gathered = values.gather(dim=1, index=gather_indices)
    if valid_mask is None:
        return gathered

    mask = valid_mask
    while mask.ndim < gathered.ndim:
        mask = mask.unsqueeze(-1)
    return gathered.masked_fill(~mask, 0)


def _gather_many_by_indices(
    values_list: list,
    indices: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> list:
    """Gather multiple ``[B, S, ...]`` tensors with a shared index expansion when shapes match."""
    if not values_list:
        return []
    # Group by trailing shape so expand+gather index is built once per group.
    groups: dict[tuple, list[int]] = {}
    for i, values in enumerate(values_list):
        key = tuple(values.shape[2:])
        groups.setdefault(key, []).append(i)
    out = [None] * len(values_list)
    mask = valid_mask
    mask_expanded_cache: dict[int, torch.Tensor] = {}
    for trailing, idxs in groups.items():
        gather_indices = _expand_gather_indices(indices, len(trailing)).expand(*indices.shape, *trailing)
        mask_e = None
        if mask is not None:
            ndim = 2 + len(trailing)
            if ndim not in mask_expanded_cache:
                m = mask
                while m.ndim < ndim:
                    m = m.unsqueeze(-1)
                mask_expanded_cache[ndim] = m
            mask_e = mask_expanded_cache[ndim]
        for i in idxs:
            gathered = values_list[i].gather(dim=1, index=gather_indices)
            if mask_e is not None:
                gathered = gathered.masked_fill(~mask_e, 0)
            out[i] = gathered
    return out


def _pack_by_kept_mask(
    values: torch.Tensor,
    kept_mask: torch.Tensor,
    lengths: torch.Tensor,
    max_length: int,
) -> torch.Tensor:
    """p-MoD-style pack: boolean index → split → ``pad_sequence`` (bs=1 skips pad).

    ``values`` is ``[B, S]`` or ``[B, S, ...]``; ``kept_mask`` / ``lengths`` are per-sample.
    """
    batch_size = int(values.shape[0])
    trailing = values.shape[2:]
    if max_length <= 0:
        return values.new_zeros((batch_size, 0, *trailing) if trailing else (batch_size, 0))

    # TINY path (p-MoD): single sample — flat boolean select, no pad_sequence.
    if batch_size == 1:
        packed = values[kept_mask]
        return packed.unsqueeze(0)

    parts = torch.split(values[kept_mask], lengths.tolist(), dim=0)
    return torch.nn.utils.rnn.pad_sequence(parts, batch_first=True)


def _pack_many_by_kept_mask(
    values_list: list,
    kept_mask: torch.Tensor,
    lengths: torch.Tensor,
    max_length: int,
) -> list:
    return [_pack_by_kept_mask(v, kept_mask, lengths, max_length) for v in values_list]


def _prefix_pack_lengths(
    kept_tokens_mask: torch.Tensor,
    response_start_idx: torch.Tensor,
) -> torch.Tensor:
    """Number of packed tokens whose logical index is strictly before ``response_start_idx``."""
    seq_len = int(kept_tokens_mask.shape[1])
    cols = torch.arange(seq_len, device=kept_tokens_mask.device, dtype=torch.long).unsqueeze(0)
    rsi = response_start_idx.to(device=kept_tokens_mask.device, dtype=torch.long).unsqueeze(1)
    prefix_cols = cols < rsi
    return (kept_tokens_mask & prefix_cols).sum(dim=1).to(dtype=torch.long)


def _normalize_cu_seqlens(cu_seqlens: torch.Tensor, total_tokens: int, device: torch.device) -> torch.Tensor:
    cu = cu_seqlens.to(device=device, dtype=torch.int32).reshape(-1)
    if int(cu[0].item()) != 0:
        raise ValueError(f"samod_cu_seqlens must start at 0, got {cu.tolist()}")
    if int(cu[-1].item()) != int(total_tokens):
        raise ValueError(
            f"samod_cu_seqlens end {int(cu[-1].item())} != total_tokens {total_tokens}"
        )
    return cu


def _varlen_to_padded_batch(
    values: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert flat varlen tokens ``[T, ...]`` / ``[1, T, ...]`` to padded ``[B, max_len, ...]``.

    Returns ``(padded, valid_mask)`` with ``valid_mask`` shape ``[B, max_len]``.
    This restores per-sample batch boundaries for text SAMoD's existing ``[B, S]`` pack path
    (required for linear-attention isolation) while still allowing actor-side rmpad.
    """
    if values.ndim >= 2 and values.shape[0] == 1:
        values = values.squeeze(0)
    total = int(values.shape[0])
    cu = _normalize_cu_seqlens(cu_seqlens, total, values.device)
    lengths = (cu[1:] - cu[:-1]).to(dtype=torch.long)
    batch_size = int(lengths.numel())
    if batch_size == 0:
        empty = values.new_empty((0, 0, *values.shape[1:]))
        return empty, torch.zeros((0, 0), device=values.device, dtype=torch.bool)
    max_len = int(lengths.max().item())
    token_ids = torch.arange(max_len, device=values.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    valid = token_ids < lengths.unsqueeze(1)
    src = cu[:-1].to(dtype=torch.long).unsqueeze(1) + token_ids
    if total > 0:
        src_safe = torch.where(valid, src, torch.zeros_like(src))
        padded = values[src_safe]
    else:
        padded = values.new_zeros((batch_size, max_len, *values.shape[1:]))
    if padded.ndim > 2:
        padded = padded.masked_fill(~valid.view(batch_size, max_len, *([1] * (padded.ndim - 2))), 0)
    elif padded.ndim == 2:
        padded = padded.masked_fill(~valid, 0)
    else:
        padded = padded.masked_fill(~valid, 0)
    return padded, valid


def _padded_batch_to_varlen(
    values: torch.Tensor,
    cu_seqlens: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Inverse of ``_varlen_to_padded_batch``: ``[B, max_len, ...]`` → ``[1, T, ...]``."""
    cu = cu_seqlens.to(device=values.device, dtype=torch.int32).reshape(-1)
    lengths = (cu[1:] - cu[:-1]).to(dtype=torch.long)
    batch_size = int(lengths.numel())
    total = int(cu[-1].item()) if cu.numel() > 0 else 0
    if batch_size == 0 or total == 0:
        return values.new_empty((1, 0, *values.shape[2:]))

    max_len = int(values.shape[1])
    token_ids = torch.arange(max_len, device=values.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    if valid_mask is not None:
        valid = valid_mask.to(device=values.device, dtype=torch.bool)
        ranks = valid.to(dtype=torch.long).cumsum(dim=1) - 1
        dest = cu[:-1].to(dtype=torch.long).unsqueeze(1) + ranks
    else:
        valid = token_ids < lengths.unsqueeze(1)
        dest = cu[:-1].to(dtype=torch.long).unsqueeze(1) + token_ids

    flat = values.new_zeros((total, *values.shape[2:]))
    flat[dest[valid]] = values[valid]
    return flat.unsqueeze(0)


def _restore_rmpad_padded_batch(
    *,
    input_ids: torch.Tensor,
    rmpad_indices: torch.Tensor,
    padded_batch_size: int,
    padded_seqlen: int,
    attention_mask: Optional[torch.Tensor] = None,
    mm_token_type_ids: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Restore actor rmpad ``[1, nnz]`` (+ optional flat side tensors) to padded ``[B, S]`` via ``pad_input``."""
    from flash_attn.bert_padding import pad_input

    flat_ids = input_ids.squeeze(0)
    bsz = int(padded_batch_size)
    seqlen = int(padded_seqlen)
    pad_ids = pad_input(flat_ids.unsqueeze(-1), rmpad_indices, bsz, seqlen).squeeze(-1)
    if attention_mask is None:
        flat_valid = torch.ones(flat_ids.shape[0], 1, device=flat_ids.device, dtype=torch.float32)
        pad_attn = pad_input(flat_valid, rmpad_indices, bsz, seqlen).squeeze(-1).to(dtype=torch.long)
    else:
        flat_attn = attention_mask.squeeze(0).unsqueeze(-1).to(dtype=torch.float32)
        pad_attn = pad_input(flat_attn, rmpad_indices, bsz, seqlen).squeeze(-1).to(dtype=torch.long)
    pad_mm = None
    if mm_token_type_ids is not None:
        flat_mm = mm_token_type_ids.squeeze(0).unsqueeze(-1).to(dtype=torch.float32)
        pad_mm = pad_input(flat_mm, rmpad_indices, bsz, seqlen).squeeze(-1).to(dtype=mm_token_type_ids.dtype)
    pad_pos = None
    if position_ids is not None:
        if position_ids.ndim == 3 and int(position_ids.shape[1]) == 1:
            # [C, 1, T] → pad once as [T, C] then reshape to [C, B, S]
            flat_pos = position_ids.squeeze(1).transpose(0, 1).to(dtype=torch.float32)  # [T, C]
            pad_pos_bt = pad_input(flat_pos, rmpad_indices, bsz, seqlen)  # [B, S, C]
            pad_pos = pad_pos_bt.to(dtype=position_ids.dtype).permute(2, 0, 1).contiguous()
        else:
            flat_pos = position_ids.squeeze(0).unsqueeze(-1).to(dtype=torch.float32)
            pad_pos = pad_input(flat_pos, rmpad_indices, bsz, seqlen).squeeze(-1).to(dtype=position_ids.dtype)
    return pad_ids, pad_attn, pad_mm, pad_pos


def _unflatten_mrope_position_ids(
    position_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> torch.Tensor:
    """Unflatten rmpad MRoPE ids into padded batch layout.

    Accepts:
      - ``[C, 1, T]`` (C=3 or 4) → ``[C, B, max_len]``
      - ``[1, T]`` / ``[T]`` → ``[B, max_len]``
    """
    if position_ids.ndim == 3 and position_ids.shape[1] == 1:
        # [C, 1, T] — stack channels as trailing dim, pad once, restore [C, B, S].
        flat_pos = position_ids.squeeze(1).transpose(0, 1)  # [T, C]
        padded, _ = _varlen_to_padded_batch(flat_pos, cu_seqlens)  # [B, max_len, C]
        return padded.permute(2, 0, 1).contiguous()
    if position_ids.ndim == 2 and position_ids.shape[0] == 1:
        padded, _ = _varlen_to_padded_batch(position_ids, cu_seqlens)
        return padded
    if position_ids.ndim == 1:
        padded, _ = _varlen_to_padded_batch(position_ids, cu_seqlens)
        return padded
    return position_ids


def _layer_has_previous_state(
    past_key_values: Optional[Cache],
    layer_idx: int,
    layer_type: str,
) -> bool:
    if past_key_values is None:
        return False
    if layer_type == "linear_attention":
        try:
            return bool(past_key_values.has_previous_state(layer_idx))
        except (AttributeError, ValueError):
            return False
    try:
        return past_key_values.get_seq_length(layer_idx) > 0
    except (AttributeError, ValueError):
        return past_key_values.get_seq_length() > 0


def _get_samod_no_split_modules(config) -> list[str]:
    del config
    return ["Qwen3_5SAMoDDecoderLayer", "Qwen3_5VisionBlock"]


def _get_samod_margin(config) -> float:
    raw = getattr(config, "samod_margin", None)
    if raw is None:
        return 0.1
    return float(raw)


def _resolve_samod_schedule(config) -> tuple[list[int], dict[int, float]]:
    """SAMoD on every LLM layer. Vision configs have no ``num_hidden_layers``."""
    num_hidden_layers = getattr(config, "num_hidden_layers", None)
    if num_hidden_layers is None:
        return [], {}
    margin = _get_samod_margin(config)
    indices = list(range(int(num_hidden_layers)))
    return indices, {idx: margin for idx in indices}


def configure_samod(
    config,
    margin: float = 0.1,
    dreg_loss_weight: float = 0.001,
    sparsity_target: Optional[float] = 0.05,
    weight_norm: float = 0.2,
    ohisa_loss_weight: float = 0.0,
):
    """Attach the remaining SAMoD / OHiSA knobs to the LLM text config."""
    text_config = _get_text_config(config)
    text_config.samod_margin = float(margin)
    text_config.samod_dreg_loss_weight = float(dreg_loss_weight)
    text_config.samod_sparsity_target = None if sparsity_target is None else float(sparsity_target)
    text_config.samod_weight_norm = _branch_reweight_scale(weight_norm)
    text_config.ohisa_loss_weight = float(ohisa_loss_weight)
    return config


def _attach_samod_dreg_loss(output, score_loss: Optional[torch.Tensor]):
    """Attach unweighted ``samod_dreg_loss`` only (no separate unweighted field)."""
    if score_loss is not None:
        setattr(output, "samod_dreg_loss", score_loss)
    return output


def _attach_extra_metrics(output, extra_metrics: Optional[dict[str, torch.Tensor]]):
    if extra_metrics:
        setattr(output, "extra_metrics", extra_metrics)
    return output


def _attach_samod_trajectory(output, trajectory: Optional[dict]):
    if trajectory is not None:
        setattr(output, "samod_trajectory", trajectory)
    return output


def _get_samod_dreg_loss_weight(config) -> float:
    text_config = getattr(config, "text_config", config)
    return float(getattr(text_config, "samod_dreg_loss_weight", 1.0))


def _get_samod_sparsity_target(config) -> Optional[tuple[float, float]]:
    """Vision token retention gate ``(0, target)``, the old ``0.00-x`` range. ``None`` disables it."""
    text_config = getattr(config, "text_config", config)
    raw = getattr(text_config, "samod_sparsity_target", None)
    if raw is None:
        return None
    upper = float(raw)
    if not (0.0 <= upper <= 1.0):
        raise ValueError(f"samod_sparsity_target must be in [0, 1], got {upper}")
    return 0.0, upper


def _normalize_samod_sparsity_target(config) -> None:
    text_config = getattr(config, "text_config", config)
    bounds = _get_samod_sparsity_target(config)
    text_config.samod_sparsity_target = None if bounds is None else bounds[1]


def _build_samod_dreg_loss(
    score_stats: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    config,
    usage_stats: Optional[list[tuple[int, torch.Tensor, torch.Tensor]]] = None,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Return unweighted SAMoD score loss and its per-sample values.

    The scalar is what training logs. The second tensor is length ``B`` and is
    detached. ``samod_dreg_loss_weight`` is not applied to either.

    Training paths multiply by ``samod_dreg_loss_weight`` when adding to the total loss.

    If ``samod_sparsity_target`` is ``null``, return the per-sample mean as-is.
    Otherwise the target is a vision token retention upper bound ``x`` and the hard gate is:
        ``vision_token_retention > x``: ``+mean_keep``,
        else ``0``.
    Each sample's vision token retention ``(#kept vision) / (#vision)`` is aggregated
    across SAMoD layers without mixing samples. Per-sample losses are averaged
    only after gating.
    """
    if not score_stats:
        return None, None
    # Token/layer averages stay FP32 even if a producer leaked BF16 stats.
    keep_sum = torch.stack(
        [item[0].to(dtype=torch.float32) for item in score_stats]
    ).sum(dim=0)
    score_count = torch.stack(
        [item[2].to(device=keep_sum.device, dtype=torch.float32) for item in score_stats]
    ).sum(dim=0)
    valid_samples = score_count > 0
    if not bool(valid_samples.any()):
        return None, None
    mean_keep = keep_sum / score_count.clamp_min(1.0)

    def _finish(per_sample: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        full = torch.where(valid_samples, per_sample, torch.zeros_like(per_sample))
        return per_sample[valid_samples].mean(), full.detach()

    target_range = _get_samod_sparsity_target(config)
    if target_range is None:
        return _finish(mean_keep)

    if not usage_stats:
        return _finish(mean_keep)
    passed = torch.stack(
        [item[1].to(device=mean_keep.device, dtype=torch.float32) for item in usage_stats]
    ).sum(dim=0)
    total = torch.stack(
        [item[2].to(device=mean_keep.device, dtype=torch.float32) for item in usage_stats]
    ).sum(dim=0)
    vision_token_retention = passed / total.clamp_min(1.0)
    _lower, upper = target_range
    per_sample_loss = torch.where(
        vision_token_retention > upper,
        mean_keep,
        torch.zeros_like(mean_keep),
    )
    return _finish(per_sample_loss)


def _build_samod_extra_metrics(
    usage_stats: Optional[list[tuple[int, torch.Tensor, torch.Tensor]]],
    *,
    include_per_layer: bool = False,
) -> Optional[dict[str, torch.Tensor]]:
    if not usage_stats:
        return None
    used_tokens = torch.stack([item[1].float() for item in usage_stats]).sum()
    total_tokens = torch.stack(
        [item[2].to(device=used_tokens.device, dtype=torch.float32) for item in usage_stats]
    ).sum()
    metrics: dict[str, torch.Tensor] = {
        "samod_vision_token_retention_numerator": used_tokens,
        "samod_vision_token_retention_denominator": total_tokens,
        "samod_vision_token_retention": used_tokens / total_tokens.clamp_min(1.0),
    }
    if include_per_layer:
        for layer_idx, passed, total in usage_stats:
            passed_f = passed.to(device=used_tokens.device, dtype=torch.float32).sum()
            total_f = total.to(device=used_tokens.device, dtype=torch.float32).sum().clamp_min(1.0)
            metrics[f"samod_vision_token_retention_layer_{int(layer_idx)}"] = passed_f / total_f
            metrics[f"samod_vision_token_retention_layer_{int(layer_idx)}_numerator"] = passed_f
            metrics[f"samod_vision_token_retention_layer_{int(layer_idx)}_denominator"] = total_f
    return metrics


def _vision_segment_ids(
    vision_mask: torch.Tensor,
    document_breaks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Positive id for each contiguous vision run. Non-vision positions stay 0."""
    mask = vision_mask.to(dtype=torch.bool)
    prev = torch.zeros_like(mask)
    if int(mask.shape[-1]) > 1:
        prev[..., 1:] = mask[..., :-1]
    if document_breaks is not None:
        prev = prev & ~document_breaks.to(device=mask.device, dtype=torch.bool)
    starts = mask & ~prev
    return starts.to(dtype=torch.long).cumsum(dim=-1) * mask.to(dtype=torch.long)


def _broadcast_segment_mean(
    values: torch.Tensor,
    segment_ids: torch.Tensor,
    max_id: Optional[int] = None,
    reduction: Optional[dict] = None,
) -> torch.Tensor:
    """Replace each vision token with the mean of its segment. Others stay 0."""
    if values.numel() == 0:
        return torch.zeros_like(values)
    if max_id is None:
        max_id = int(segment_ids.max().item()) if segment_ids.numel() else 0
    if max_id == 0:
        return torch.zeros_like(values)
    if reduction is None:
        batch = int(values.shape[0])
        offsets = torch.arange(batch, device=values.device).unsqueeze(1) * (max_id + 1)
        flat_ids = (segment_ids + offsets).reshape(-1)
        valid = (segment_ids.reshape(-1) > 0).to(dtype=torch.float32)
        n_bins = batch * (max_id + 1)
        counts = torch.zeros(n_bins, device=values.device, dtype=torch.float32)
        counts.scatter_add_(0, flat_ids, valid)
    else:
        flat_ids = reduction["flat_ids"]
        valid = reduction["valid"]
        counts_clamped = reduction["counts_clamped"]
        n_bins = int(reduction["n_bins"])
    sums = torch.zeros(n_bins, device=values.device, dtype=torch.float32)
    # Add the zeros too. Dropping them changes CUDA scatter_add order by ~1 ulp.
    sums.scatter_add_(0, flat_ids, values.reshape(-1).float() * valid)
    if reduction is None:
        counts_clamped = counts.clamp_min(1.0)
    means = sums / counts_clamped
    out = means[flat_ids].reshape_as(values).to(dtype=values.dtype)
    return out.masked_fill(segment_ids == 0, 0)


_TANH_ONE = math.tanh(1.0)


def _vision_run_starts(
    vision_mask: torch.Tensor,
    document_breaks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """True on the first token of each contiguous vision run (one image)."""
    mask = vision_mask.to(dtype=torch.bool)
    prev = torch.zeros_like(mask)
    if int(mask.shape[-1]) > 1:
        prev[..., 1:] = mask[..., :-1]
    if document_breaks is not None:
        prev = prev & ~document_breaks.to(device=mask.device, dtype=torch.bool)
    return mask & ~prev


def _paramthresh_image_base_sums(
    scores: torch.Tensor,
    threshold: torch.Tensor,
    vision_mask: torch.Tensor,
    keep_arg: float,
    document_breaks: Optional[torch.Tensor] = None,
    segment_ids: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-sample sum and count of the NEO paramthresh score base.

    One image contributes once:
    ``stopgrad(mean_t relu(tanh((score + threshold) / keep_arg + 1) / tanh(1))) * threshold``.
    The outer ``stopgrad`` already stops both score and the threshold inside ``tanh``.
    The outer multiply still trains ``threshold``. Callers that share the live
    keep router pass a threshold recomputed on detached hidden states, so this
    term does not train the backbone.
    """
    if float(keep_arg) == 0.0:
        raise ValueError("paramthresh score base divides by samod_margin, which is 0")
    mask = vision_mask.to(device=scores.device, dtype=torch.bool)
    scores_f = scores.to(device=scores.device, dtype=torch.float32)
    thresh = threshold.to(device=scores.device, dtype=torch.float32)
    if thresh.ndim == 1:
        thresh = thresh.unsqueeze(-1)
    if thresh.shape != scores_f.shape:
        thresh = thresh.expand_as(scores_f)
    activated = torch.relu(torch.tanh((scores_f + thresh) / float(keep_arg) + 1.0) / _TANH_ONE)
    factor = _per_image_threshold(
        activated, mask, document_breaks, segment_ids=segment_ids
    ).detach()
    term = factor * thresh
    if segment_ids is None:
        starts = _vision_run_starts(mask, document_breaks)
    else:
        segments = segment_ids.to(device=mask.device, dtype=torch.long).masked_fill(~mask, 0)
        previous = torch.zeros_like(segments)
        previous[..., 1:] = segments[..., :-1]
        starts = (segments > 0) & (segments != previous)
    image_sum = (term * starts.to(dtype=term.dtype)).sum(dim=-1)
    n_images = starts.sum(dim=-1).to(dtype=torch.float32)
    return image_sum, n_images


def _vision_run_count(
    vision_mask: torch.Tensor,
    document_breaks: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Number of contiguous vision runs in each row."""
    mask = vision_mask.to(dtype=torch.bool)
    prev = torch.zeros_like(mask)
    if int(mask.shape[-1]) > 1:
        prev[..., 1:] = mask[..., :-1]
    if document_breaks is not None:
        prev = prev & ~document_breaks.to(device=mask.device, dtype=torch.bool)
    return (mask & ~prev).sum(dim=-1)


def _per_image_threshold(
    token_aux: torch.Tensor,
    vision_mask: torch.Tensor,
    document_breaks: Optional[torch.Tensor] = None,
    segment_ids: Optional[torch.Tensor] = None,
    segment_max_id: Optional[int] = None,
    segment_reduction: Optional[dict] = None,
) -> torch.Tensor:
    """Mean of per-token aux logits inside each image, broadcast onto that image.

    One image in the row is the previous sample mean: sum the vision tokens and
    divide by their count. Segment ids are only built when some row has more
    than one image.
    """
    mask = vision_mask.to(dtype=torch.bool)
    if segment_ids is not None:
        segments = (
            segment_ids
            if segment_reduction is not None
            else segment_ids.to(device=mask.device, dtype=torch.long).masked_fill(~mask, 0)
        )
        return _broadcast_segment_mean(
            token_aux,
            segments,
            max_id=segment_max_id,
            reduction=segment_reduction,
        )
    n_images = _vision_run_count(mask, document_breaks)
    if n_images.numel() == 0 or int(n_images.max().item()) <= 1:
        vis = mask.to(dtype=token_aux.dtype)
        denom = vis.sum(dim=-1, keepdim=True).clamp_min(1)
        mean = (token_aux * vis).sum(dim=-1, keepdim=True) / denom
        return mean.expand_as(token_aux) * vis
    return _broadcast_segment_mean(token_aux, _vision_segment_ids(mask, document_breaks))


class Qwen3_5SAMoDTokenRouter(nn.Module):
    """Vision-token router: ``Linear(H→2)``, dim1 averaged inside each image."""

    def __init__(self, config: Union[Qwen3_5TextConfig, Qwen3_5VisionConfig]):
        super().__init__()
        self.config = config
        self.is_paramthresh = True
        self.is_two_dim_router = True
        self.router = nn.Linear(config.hidden_size, 2, bias=True)
        self.router._is_samod_router = True
        # Image-RMS normalize (former normalize_hidden=true, not RMSNorm).
        self.normalize_hidden = True
        self.hidden_norm = None
        self.reset_parameters()

    def reset_parameters(self):
        _init_samod_linear(self.router, self.config)

    def _router_hidden(
        self,
        hidden_states: torch.Tensor,
        vision_mask: Optional[torch.Tensor] = None,
        *,
        normalize: bool = False,
    ) -> torch.Tensor:
        if not normalize:
            return hidden_states
        if vision_mask is None:
            vision_mask = torch.ones(
                hidden_states.shape[:2], dtype=torch.bool, device=hidden_states.device
            )
        return _router_normalize_by_image(hidden_states, vision_mask)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.router(self._router_hidden(hidden_states))[..., 0]

    def _forward_vision_router(
        self,
        hidden_states: torch.Tensor,
        vision_token_mask: torch.Tensor,
        *,
        return_param_threshold: bool = False,
        vision_pack: Optional[dict] = None,
        normalize: bool = False,
    ):
        """Vision-only router via per-sample pack + ``gather`` (not flat boolean index).

        On PPU/CUDA, ``hidden_states[mask]`` → Linear → ``scores[mask]=...`` is *not*
        gradient-equivalent to full-seq+mask (Adam drifts over steps; matches the
        late SFT curve divergence under rmpad unflatten). Per-sample pack using the
        same ``_build_packed_indices`` / ``_gather_by_indices`` path as early-prune
        matches full-seq+mask exactly while still skipping text/pad tokens in the
        kept pack length.
        """
        batch = int(hidden_states.shape[0])
        mask = vision_token_mask.to(device=hidden_states.device, dtype=torch.bool)
        scores = hidden_states.new_zeros(hidden_states.shape[:2])
        thresh = None
        if vision_pack is not None and vision_pack.get("indices") is not None:
            max_len = int(vision_pack["max_len"])
            pack_mask = vision_pack["pack_mask"]
            indices = vision_pack["indices"]
            batch_ids = vision_pack.get("batch_ids")
        else:
            lengths = mask.sum(dim=1)
            max_len = int(lengths.max().item()) if batch > 0 else 0
            if max_len == 0:
                return (scores, thresh) if return_param_threshold else scores
            pack_mask = torch.arange(max_len, device=hidden_states.device).unsqueeze(0) < lengths.unsqueeze(1)
            indices = _build_packed_indices(mask, max_len)
            batch_ids = None
        if max_len == 0:
            return (scores, thresh) if return_param_threshold else scores

        if batch_ids is None:
            batch_ids = torch.arange(batch, device=hidden_states.device).unsqueeze(1).expand_as(indices)
        flat = pack_mask

        vis = _gather_by_indices(hidden_states, indices, pack_mask)
        if normalize:
            # Eval: Linear on raw pack, then / mean_rms (+ bias).
            out = _router_linear_packed_rms_normalized(
                vis,
                hidden_states,
                mask,
                indices,
                pack_mask,
                vision_pack,
                self.router.weight,
                self.router.bias,
            )
        else:
            out = self.router(vis)

        scores[batch_ids[flat], indices[flat]] = out[..., 0][flat].to(dtype=scores.dtype)
        if not return_param_threshold:
            return scores
        token_aux = hidden_states.new_zeros(hidden_states.shape[:2])
        token_aux[batch_ids[flat], indices[flat]] = out[..., 1][flat].to(dtype=token_aux.dtype)
        thresh = _per_image_threshold(
            token_aux,
            mask,
            segment_ids=None if vision_pack is None else vision_pack.get("segment_ids"),
            segment_max_id=None if vision_pack is None else vision_pack.get("segment_max_id"),
            segment_reduction=None if vision_pack is None else vision_pack.get("segment_reduction"),
        )
        return scores, thresh

    def forward_score_threshold(
        self,
        hidden_states: torch.Tensor,
        vision_token_mask: torch.Tensor,
        *,
        vision_pack: Optional[dict] = None,
        normalize: bool = False,
    ):
        """Return score and threshold for a two-dimensional router rule."""
        if not self.is_two_dim_router:
            raise RuntimeError("forward_score_threshold requires a two-dimensional router rule")
        return self._forward_vision_router(
            hidden_states,
            vision_token_mask,
            return_param_threshold=True,
            vision_pack=vision_pack,
            normalize=normalize,
        )

    def forward_paramthresh(
        self,
        hidden_states: torch.Tensor,
        vision_token_mask: torch.Tensor,
        *,
        return_param_threshold: bool = True,
        vision_pack: Optional[dict] = None,
        normalize: bool = False,
    ):
        """Vision-only fused ``Linear(H→2)`` (packed gather). See ``_forward_vision_router``."""
        if not self.is_paramthresh:
            raise RuntimeError("forward_paramthresh requires a paramthresh-family keep rule")
        if return_param_threshold:
            return self.forward_score_threshold(
                hidden_states, vision_token_mask, vision_pack=vision_pack, normalize=normalize
            )
        return self._forward_vision_router(
            hidden_states, vision_token_mask, vision_pack=vision_pack, normalize=normalize
        )

def _init_samod_linear(module: nn.Linear, config) -> None:
    """Zero-init the fused H→2 router (former special_init, std 0)."""
    del config
    nn.init.zeros_(module.weight)
    if module.bias is not None:
        nn.init.zeros_(module.bias)


def _reset_samod_router_parameters(module: nn.Module):
    for submodule in module.modules():
        if isinstance(submodule, Qwen3_5SAMoDTokenRouter):
            submodule.reset_parameters()


def _samod_init_router_weights(model: nn.Module, module: nn.Module, super_init) -> None:
    del model
    if isinstance(module, nn.Linear) and getattr(module, "_is_samod_router", False):
        if not getattr(module.weight, "_is_hf_initialized", False):
            nn.init.zeros_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            return
    super_init(module)


def _branch_reweight_scale(weight_norm) -> float:
    """``w`` in ``1 + w * tanh``. ``w <= 0`` disables branch reweight."""
    if isinstance(weight_norm, bool) or not isinstance(weight_norm, (int, float, str)):
        raise ValueError(f"samod_weight_norm must be a number, got {weight_norm!r}")
    if isinstance(weight_norm, (int, float)):
        return float(weight_norm)
    text = str(weight_norm).strip()
    try:
        return float(text)
    except ValueError:
        pass
    match = re.fullmatch(r"tanh_([+-]?(?:\d+\.\d+|\d+))_1(?:\.0+)?", text)
    if match:
        return float(match.group(1))
    raise ValueError(f"samod_weight_norm must be a number, got {weight_norm!r}")


class Qwen3_5SAMoDDecoderLayer(Qwen3_5DecoderLayer):
    def __init__(self, config: Qwen3_5TextConfig, layer_idx: int, keep_arg: Optional[float] = None, threshold: Optional[float] = None):
        super().__init__(config=config, layer_idx=layer_idx)
        self.config = config
        self.layer_idx = layer_idx
        if keep_arg is None and threshold is not None:
            keep_arg = threshold
        self.samod_enabled = keep_arg is not None
        if self.samod_enabled:
            self.samod_margin = float(keep_arg)
            self.norm_scale = _branch_reweight_scale(
                getattr(config, "samod_weight_norm", 0.2)
            )
            self.reweight_skipped = True
            self.router = Qwen3_5SAMoDTokenRouter(config)

    def __call__(self, *args, **kwargs):
        # SAMoD packing length is data-dependent. HF GradientCheckpointingLayer wraps the
        # *whole* layer, so recompute can rebuild a different keep length → CheckpointError.
        # Bypass the outer layer checkpoint when SAMoD+GC; selective checkpoint of the packed
        # attn/MLP runs inside forward() after keep/pack are fixed (FSDP/DTensor-safe).
        if self.samod_enabled and self.gradient_checkpointing and self.training:
            return nn.Module.__call__(self, *args, **kwargs)
        return super().__call__(*args, **kwargs)

    def _branch_reweight(self, logits: torch.Tensor, vision_mask: torch.Tensor) -> torch.Tensor:
        """``1 + w * tanh(logit)`` on vision tokens. ``w <= 0`` leaves every weight at 1."""
        if self.norm_scale <= 0.0:
            return torch.ones_like(logits)
        signed = torch.tanh(logits)
        mask = vision_mask.to(device=logits.device, dtype=torch.bool)
        return torch.where(mask, 1.0 + self.norm_scale * signed, torch.ones_like(signed))

    def _build_keep_vision_mask(
        self,
        router_scores: torch.Tensor,
        visual_pos_masks: torch.Tensor,
        attention_mask_2d: Optional[torch.Tensor] = None,
        param_threshold: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        valid = visual_pos_masks.to(dtype=torch.bool)
        if attention_mask_2d is not None:
            valid = valid & attention_mask_2d.to(device=valid.device, dtype=torch.bool)
        if param_threshold is None:
            raise ValueError("paramthresh keep requires param_threshold")
        thresh = param_threshold.to(device=router_scores.device, dtype=router_scores.dtype)
        if thresh.ndim == 1:
            thresh = thresh.unsqueeze(-1)
        return valid & ((router_scores + thresh + float(self.samod_margin)) >= 0)

    def _compute_samod_score_stats(
        self,
        router_scores: torch.Tensor,
        vision_token_mask: torch.Tensor,
        kept_vision_mask: torch.Tensor,
        param_threshold: Optional[torch.Tensor] = None,
        segment_ids: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per-image PDF term, counted once per image.

        ``stopgrad(mean_t relu(tanh((score+threshold)/margin+1)/tanh(1))) * threshold``.
        """
        del kept_vision_mask
        zero = torch.zeros((router_scores.shape[0],), device=router_scores.device, dtype=torch.float32)
        if not bool(vision_token_mask.any()):
            return zero, zero, zero
        if param_threshold is None:
            raise ValueError("paramthresh score loss requires param_threshold")
        image_sum, n_images = _paramthresh_image_base_sums(
            router_scores,
            param_threshold,
            vision_token_mask,
            self.samod_margin,
            segment_ids=segment_ids,
        )
        return image_sum, image_sum, n_images

    def _forward_token_mixer(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> torch.Tensor:
        if self.layer_type == "linear_attention":
            return self.linear_attn(
                hidden_states=hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
            )
        if self.layer_type == "full_attention":
            hidden_states, _ = self.self_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            return hidden_states
        raise NotImplementedError(f"Unsupported Qwen3.5 layer_type: {self.layer_type}")

    def _forward_decoder(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._forward_token_mixer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states

    def forward_return_residual(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self._forward_token_mixer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual

    def _forward_without_mod(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        attention_mask_2d: Optional[torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[Cache],
        **kwargs,
    ) -> torch.Tensor:
        """Dense residual path (no router / pack). Also used by force-dense rollout."""
        if (
            past_key_values is not None
            and self.layer_type == "full_attention"
            and _layer_has_previous_state(past_key_values, self.layer_idx, self.layer_type)
        ):
            attention_mask = _create_layer_causal_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=attention_mask_2d,
                past_key_values=past_key_values,
                position_ids=position_ids,
                layer_idx=self.layer_idx,
            )
        return self._forward_decoder(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    def _forward_samod_cached_decode(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        attention_mask_2d: Optional[torch.Tensor],
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: Optional[Cache],
        **kwargs,
    ) -> torch.Tensor:
        """SAMoD decode after sparse prefill: reuse per-layer compact KV; keep all current tokens.

        Prefill already decided keep/skip and wrote packed KV (different length per layer).
        Generate steps only append non-vision tokens, which are always kept (branch weight 1).
        Attention must use compact ``layer_valid`` masks — never the logical prompt mask.

        ``layer_valid`` may store ``None`` as an all-valid sentinel (no padding in compact KV).
        """
        q_len = int(hidden_states.shape[1])
        bsz = int(hidden_states.shape[0])
        device = hidden_states.device
        meta = _ensure_samod_cache_meta(past_key_values)
        cur_valid = torch.ones((bsz, q_len), device=device, dtype=torch.bool)
        cur_has_pad = False
        next_layer_valid = None
        reuse_next_layer_valid = False
        if attention_mask_2d is not None and attention_mask_2d.shape[-1] >= q_len:
            # Decode usually passes the full logical mask; only the last q_len columns are current.
            cur_valid = attention_mask_2d[:, -q_len:].to(device=device, dtype=torch.bool)
            cur_has_pad = True  # caller provided a mask; treat as potentially padded

        if (
            past_key_values is not None
            and self.layer_type == "full_attention"
            and _layer_has_previous_state(past_key_values, self.layer_idx, self.layer_type)
        ):
            has_layer_valid = meta is not None and self.layer_idx in meta["layer_valid"]
            prev_valid = meta["layer_valid"][self.layer_idx] if has_layer_valid else None
            if has_layer_valid and prev_valid is None:
                # Explicit all-valid sentinel stored as None under the layer key.
                if not cur_has_pad:
                    attn_2d = None
                    reuse_next_layer_valid = True
                else:
                    try:
                        phys_total = int(past_key_values.get_seq_length(self.layer_idx))
                    except (AttributeError, ValueError, TypeError):
                        phys_total = q_len
                    phys_prev = max(phys_total - q_len, 0)
                    compact_2d = torch.cat(
                        [
                            torch.ones((bsz, phys_prev), device=device, dtype=torch.bool),
                            cur_valid,
                        ],
                        dim=-1,
                    )
                    attn_2d = compact_2d
                    next_layer_valid = compact_2d
                    reuse_next_layer_valid = True
            elif prev_valid is not None:
                # Prefill stored a real mask ⇒ compact KV had holes; always pass it.
                attn_2d = torch.cat([prev_valid.to(device=device, dtype=torch.bool), cur_valid], dim=-1)
                next_layer_valid = attn_2d
                reuse_next_layer_valid = True
            else:
                # Legacy cache without meta: rely on physical past length only (safe for bs=1 / no pad).
                attn_2d = None
            attention_mask = _create_layer_causal_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=attn_2d,
                past_key_values=past_key_values,
                position_ids=position_ids,
                layer_idx=self.layer_idx,
            )

        hidden_states = self._forward_decoder(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        if meta is not None and self.layer_type == "full_attention":
            layer_valid = meta["layer_valid"]
            if self.layer_idx in layer_valid:
                if reuse_next_layer_valid:
                    # This exact bool mask was already constructed for the causal mask.
                    layer_valid[self.layer_idx] = next_layer_valid
                else:
                    prev_valid = layer_valid[self.layer_idx]
                    if prev_valid is None and not cur_has_pad:
                        # Stay on all-valid sentinel — avoid growing bool masks.
                        layer_valid[self.layer_idx] = None
                    elif prev_valid is None:
                        try:
                            phys_total = int(past_key_values.get_seq_length(self.layer_idx))
                        except (AttributeError, ValueError, TypeError):
                            phys_total = q_len
                        phys_prev = max(phys_total - q_len, 0)
                        layer_valid[self.layer_idx] = torch.cat(
                            [
                                torch.ones((bsz, phys_prev), device=device, dtype=torch.bool),
                                cur_valid,
                            ],
                            dim=-1,
                        )
                    else:
                        layer_valid[self.layer_idx] = torch.cat(
                            [prev_valid.to(device=device, dtype=torch.bool), cur_valid], dim=-1
                        )
            else:
                # First decode step without prefill meta: mark full physical cache valid.
                try:
                    phys = int(past_key_values.get_seq_length(self.layer_idx))
                except (AttributeError, ValueError, TypeError):
                    phys = q_len
                if not cur_has_pad:
                    layer_valid[self.layer_idx] = None
                else:
                    layer_valid[self.layer_idx] = torch.ones(
                        (bsz, phys), device=device, dtype=torch.bool
                    )
        return hidden_states

    def _forward_samod_ragged_dense_decode(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: SAMoDRaggedCache,
        **kwargs,
    ) -> torch.Tensor:
        """Dense decode over a single-sample compact prefill cache."""
        attention_mask = None
        attn_impl = str(getattr(self.config, "_attn_implementation", "") or "")
        if (
            self.layer_type == "full_attention"
            and not attn_impl.startswith("flash_attention")
            and _layer_has_previous_state(past_key_values, self.layer_idx, self.layer_type)
        ):
            attention_mask = _create_layer_causal_mask(
                config=self.config,
                inputs_embeds=hidden_states,
                attention_mask=None,
                past_key_values=past_key_values,
                position_ids=position_ids,
                layer_idx=self.layer_idx,
            )
        return self._forward_decoder(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    def _forward_packed_prefix_suffix_incremental(
        self,
        kept_tokens: torch.Tensor,
        kept_cos: torch.Tensor,
        kept_sin: torch.Tensor,
        kept_attention_mask: Optional[torch.Tensor],
        kept_attention_mask_2d: torch.Tensor,
        kept_attention_mask_arg: Optional[torch.Tensor],
        kept_position_ids: Optional[torch.Tensor],
        prefix_pack_lens: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-sample prefix pack + token-wise suffix — matches generate(use_cache=True)."""
        bsz, max_kept, hidden = kept_tokens.shape
        mod_out = kept_tokens.new_zeros(bsz, max_kept, hidden)
        res_out = kept_tokens.new_zeros(bsz, max_kept, hidden)
        for b in range(bsz):
            total = int(kept_attention_mask_2d[b].sum().item())
            p_len = min(int(prefix_pack_lens[b].item()), total)
            layer_cache = DynamicCache(config=self.config)
            if p_len > 0:
                tokens_p = kept_tokens[b : b + 1, :p_len]
                if self.layer_type == "linear_attention":
                    attn_p = None
                else:
                    attn_p = _create_layer_causal_mask(
                        config=self.config,
                        inputs_embeds=tokens_p,
                        attention_mask=None,
                        past_key_values=None,
                        position_ids=None,
                        layer_idx=self.layer_idx,
                    )
                mod_p, res_p = self.forward_return_residual(
                    tokens_p,
                    attention_mask=attn_p,
                    position_ids=None,
                    past_key_values=layer_cache,
                    position_embeddings=(kept_cos[b : b + 1, :p_len], kept_sin[b : b + 1, :p_len]),
                    **kwargs,
                )
                mod_out[b, :p_len] = mod_p[0]
                res_out[b, :p_len] = res_p[0]
            for i in range(p_len, total):
                tokens_i = kept_tokens[b : b + 1, i : i + 1]
                if self.layer_type == "linear_attention":
                    attn_i = None
                else:
                    attn_i = _create_layer_causal_mask(
                        config=self.config,
                        inputs_embeds=tokens_i,
                        attention_mask=None,
                        past_key_values=layer_cache,
                        position_ids=None,
                        layer_idx=self.layer_idx,
                    )
                mod_i, res_i = self.forward_return_residual(
                    tokens_i,
                    attention_mask=attn_i,
                    position_ids=None,
                    past_key_values=layer_cache,
                    position_embeddings=(kept_cos[b : b + 1, i : i + 1], kept_sin[b : b + 1, i : i + 1]),
                    **kwargs,
                )
                mod_out[b, i] = mod_i[0, 0]
                res_out[b, i] = res_i[0, 0]
        return mod_out, res_out

    @staticmethod
    def _string_writeback(
        h_in: torch.Tensor,
        mod_outputs: torch.Tensor,
        residual: torch.Tensor,
        kept_router_weights: torch.Tensor,
        kept_tokens_mask: torch.Tensor,
        kept_attention_mask_2d: torch.Tensor,
        kept_attention_mask_arg: Optional[torch.Tensor],
        router_weights: Optional[torch.Tensor] = None,
        dropped_vision_mask: Optional[torch.Tensor] = None,
        reweight_skipped: bool = True,
    ) -> torch.Tensor:
        """Writeback: dropped get ``h_in * w`` only if ``reweight_skipped``; kept get ``residual + mlp * w``.

        ``kept_router_weights`` is already packed to kept length (text positions are 1).
        With ``reweight_skipped``, build a 2D scale (``1`` on kept/text, ``w`` on dropped)
        and multiply once, then overwrite kept — denser than boolean-index scaling and
        cheaper than ``where(dropped, h*w, h)`` which materializes a full ``h*w``.
        """
        dtype = mod_outputs.dtype
        h = h_in.to(dtype=dtype)
        if (
            reweight_skipped
            and dropped_vision_mask is not None
            and router_weights is not None
        ):
            scale = torch.ones(h.shape[:2], device=h.device, dtype=dtype)
            scale = torch.where(dropped_vision_mask, router_weights.to(dtype=dtype), scale)
            out = h * scale.unsqueeze(-1)
        else:
            out = h.clone()
        kept_w = kept_router_weights.to(dtype=dtype)

        if kept_attention_mask_arg is not None:
            w_kept = kept_w[kept_attention_mask_2d]
            combined = mod_outputs[kept_attention_mask_2d].mul(
                w_kept.unsqueeze(-1)
            ).add(residual[kept_attention_mask_2d])
            out[kept_tokens_mask] = combined
        else:
            flat_mlp = mod_outputs.reshape(-1, mod_outputs.shape[-1])
            flat_res = residual.reshape(-1, residual.shape[-1])
            flat_w = kept_w.reshape(-1, 1)
            out[kept_tokens_mask] = flat_mlp.mul(flat_w).add(flat_res)
        return out

    def _token_router_input(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_2d: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # SAMoD returns 6-tuple: hidden, score_sum, score_count, passed, total,
        # ohisa_keep_mask (all valid sequence tokens; response scope is applied in the loss).
        if (
            self.samod_enabled
            and isinstance(past_key_values, SAMoDRaggedCache)
            and _layer_has_previous_state(past_key_values, self.layer_idx, self.layer_type)
        ):
            hidden_states = self._forward_samod_ragged_dense_decode(
                hidden_states,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )
            return hidden_states, None, None, None, None, None

        if not self.samod_enabled:
            return self._forward_decoder(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        visual_pos_masks = (
            visual_pos_masks.to(device=hidden_states.device, dtype=torch.bool)
            if visual_pos_masks is not None
            else None
        )
        has_layer_past = _layer_has_previous_state(past_key_values, self.layer_idx, self.layer_type)
        if has_layer_past:
            zero = hidden_states.new_zeros(())
            hidden_states = self._forward_samod_cached_decode(
                hidden_states,
                attention_mask=attention_mask,
                attention_mask_2d=attention_mask_2d,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )
            keep = None
            if self.training:
                q_len = int(hidden_states.shape[1])
                bsz = int(hidden_states.shape[0])
                device = hidden_states.device
                cur_valid = torch.ones((bsz, q_len), device=device, dtype=torch.bool)
                if attention_mask_2d is not None and attention_mask_2d.shape[-1] >= q_len:
                    cur_valid = attention_mask_2d[:, -q_len:].to(device=device, dtype=torch.bool)
                keep = cur_valid.clone()
            return hidden_states, zero, zero, zero, zero, keep

        # No vision positions: plain residual path.
        if visual_pos_masks is None:
            hidden_states = self._forward_without_mod(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                attention_mask_2d=attention_mask_2d,
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )
            zero = hidden_states.new_zeros(())
            empty_keep = (
                torch.zeros(hidden_states.shape[:2], device=hidden_states.device, dtype=torch.bool)
                if self.training
                else None
            )
            return hidden_states, zero, zero, zero, zero, empty_keep

        vision_token_mask = visual_pos_masks
        valid_mask = None
        if attention_mask_2d is not None:
            valid_mask = attention_mask_2d.to(device=hidden_states.device, dtype=torch.bool)
            vision_token_mask = vision_token_mask & valid_mask

        vision_pack = kwargs.pop("samod_vision_pack", None)
        if vision_pack is None:
            vision_pack = _build_vision_pack_scaffold(vision_token_mask)

        h_in = hidden_states
        router_in = self._token_router_input(hidden_states)
        # Eval: fold image-RMS normalize into the packed router gather so we do not
        # materialize a full-sequence normalized clone and gather it again.
        # Training keeps the explicit normalize so score-loss can reuse ``router_in``.
        router_normalize = bool(
            getattr(getattr(self, "router", None), "normalize_hidden", False)
            and getattr(self.router, "hidden_norm", None) is None
        )
        fuse_pack_normalize = router_normalize and not self.training
        if router_normalize and not fuse_pack_normalize:
            router_in = _router_normalize_by_image(
                router_in,
                vision_token_mask,
                segment_ids=vision_pack.get("segment_ids"),
                segment_max_id=vision_pack.get("segment_max_id"),
                segment_reduction=vision_pack.get("segment_reduction"),
                assume_nonempty=int(vision_pack.get("max_len", 0)) > 0,
            )
        param_threshold = None
        keep_scores = None
        keep_threshold = None

        def _tanh_reweight(logit: torch.Tensor, *, mask_non_vision: bool) -> torch.Tensor:
            src = logit.masked_fill(~visual_pos_masks, float("-inf")) if mask_non_vision else logit
            return self._branch_reweight(src, vision_token_mask)

        router_raw, param_threshold = self.router.forward_paramthresh(
            router_in,
            vision_token_mask,
            return_param_threshold=True,
            vision_pack=vision_pack,
            normalize=fuse_pack_normalize,
        )
        thresh_for_add = param_threshold
        if thresh_for_add.ndim == 1:
            thresh_for_add = thresh_for_add.unsqueeze(-1)
        thresh_for_add = thresh_for_add.to(device=router_raw.device, dtype=router_raw.dtype)
        effective_scores = router_raw + thresh_for_add
        router_weights = _tanh_reweight(effective_scores, mask_non_vision=True)
        keep_scores, keep_threshold = router_raw, param_threshold
        kept_mod_target_mask = self._build_keep_vision_mask(
            router_scores=keep_scores,
            visual_pos_masks=vision_token_mask,
            param_threshold=keep_threshold,
        )

        forced_vision_token_retention = getattr(self, "_profile_vision_token_retention", None)
        if forced_vision_token_retention is not None and not self.training:
            kept_mod_target_mask = _forced_keep_mask(vision_token_mask, float(forced_vision_token_retention))

        if self.training and _get_samod_dreg_loss_weight(self.config) != 0.0:
            _, score_threshold = self.router.forward_paramthresh(
                router_in.detach(),
                vision_token_mask,
                return_param_threshold=True,
                vision_pack=vision_pack,
            )
            samod_score_keep_sum, samod_score_drop_sum, samod_score_count = self._compute_samod_score_stats(
                keep_scores,
                vision_token_mask,
                kept_mod_target_mask,
                param_threshold=score_threshold,
                segment_ids=vision_pack.get("segment_ids"),
            )
        else:
            samod_score_keep_sum = hidden_states.new_zeros(())
            samod_score_drop_sum = hidden_states.new_zeros(())
            samod_score_count = hidden_states.new_zeros(())
        self._samod_score_drop_sum = samod_score_drop_sum
        samod_score_sum = samod_score_keep_sum
        # Keep per-sample counts so score-loss gates never couple samples.
        samod_passed_vision = kept_mod_target_mask.sum(dim=1).to(dtype=torch.float32)
        samod_total_vision = vision_token_mask.sum(dim=1).to(dtype=torch.float32)

        dense_attn_kwargs = kwargs

        def _dense_weighted_branch(*, weight_mask: torch.Tensor, ohisa_km: torch.Tensor):
            def _full_attn_mlp(tokens: torch.Tensor):
                return self.forward_return_residual(
                    tokens,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    position_embeddings=position_embeddings,
                    **dense_attn_kwargs,
                )

            if self.gradient_checkpointing and self.training:
                mlp_out, residual = torch_checkpoint(_full_attn_mlp, hidden_states, use_reentrant=False)
            else:
                mlp_out, residual = _full_attn_mlp(hidden_states)
            # residual + w*mlp on weight_mask, residual+mlp elsewhere (no clone / boolean write).
            w_seq = torch.ones(router_weights.shape, device=router_weights.device, dtype=mlp_out.dtype)
            w_seq = torch.where(
                weight_mask, router_weights.to(dtype=mlp_out.dtype), w_seq
            )
            out = residual + mlp_out * w_seq.unsqueeze(-1)
            # Prefill with identity pack: record all-valid sentinel for decode.
            if (
                past_key_values is not None
                and not isinstance(past_key_values, SAMoDRaggedCache)
                and self.layer_type == "full_attention"
            ):
                meta = _ensure_samod_cache_meta(past_key_values)
                if meta is not None:
                    meta["layer_valid"][self.layer_idx] = None
            return (
                out,
                samod_score_sum,
                samod_score_count,
                samod_passed_vision,
                samod_total_vision,
                ohisa_km,
            )

        # Packing mask = kept vision ∪ all non-vision valid tokens (exclude dropped vision / pad).
        if valid_mask is not None:
            kept_tokens_mask = kept_mod_target_mask | (
                (~visual_pos_masks) & valid_mask
            )
        else:
            kept_tokens_mask = kept_mod_target_mask | (~visual_pos_masks)

        # OHiSA keep mask is the whole valid sequence. Inference does not build it.
        ohisa_keep_mask = (
            _ohisa_sequence_keep_mask(hidden_states, attention_mask_2d)
            if self.training
            else None
        )

        kept_tokens_lengths = kept_tokens_mask.sum(dim=1).int()
        max_kept_length = int(kept_tokens_lengths.max().item())
        if max_kept_length == 0:
            # All vision dropped: scale ``h_in * w`` only when reweight_skipped.
            out = h_in
            if self.reweight_skipped:
                w_full = router_weights.to(dtype=h_in.dtype).unsqueeze(-1)
                out = torch.where(vision_token_mask.unsqueeze(-1), h_in * w_full, h_in)
            return (
                out,
                samod_score_sum,
                samod_score_count,
                samod_passed_vision,
                samod_total_vision,
                ohisa_keep_mask,
            )

        # Identity pack (no vision drop, no pad holes): skip gather/scatter — dense weighted.
        # weight_mask = all vision (≡ kept when nothing dropped).
        if max_kept_length == int(hidden_states.shape[1]):
            return _dense_weighted_branch(
                weight_mask=vision_token_mask, ohisa_km=ohisa_keep_mask
            )

        kept_attention_mask_2d = (
            torch.arange(max_kept_length, device=hidden_states.device).unsqueeze(0) < kept_tokens_lengths.unsqueeze(1)
        )

        # Pack like p-MoD: boolean index + pad_sequence (bs=1 TINY path inside helper).
        # Pack branch weights too (text→1, vision→w) so writeback avoids boolean gather.
        cos, sin = position_embeddings
        w_full = torch.ones_like(router_weights)
        w_full = torch.where(vision_token_mask, router_weights, w_full)
        pack_srcs = [hidden_states, cos, sin, w_full]
        if position_ids is not None:
            pack_srcs.append(position_ids)
        packed = _pack_many_by_kept_mask(
            pack_srcs, kept_tokens_mask, kept_tokens_lengths, max_kept_length
        )
        kept_tokens, kept_cos, kept_sin, kept_w = packed[0], packed[1], packed[2], packed[3]
        kept_position_ids = packed[4] if position_ids is not None else None
        packed_attn_kwargs = kwargs

        if kept_tokens_mask.shape[0] == 1:
            kept_attention_mask_arg = None
        else:
            # Equal pack lengths ⇒ all-True mask; skip FA padding mask (matches prior torch.all).
            equal_lens = bool((kept_tokens_lengths == max_kept_length).all().item())
            kept_attention_mask_arg = None if equal_lens else kept_attention_mask_2d

        # When attention_mask is None (typically bs==1), FlashAttention may treat gapped
        # text position_ids (gaps from dropped vision tokens) as a packed sequence and
        # then crash if no position_id==0 boundary exists. RoPE already uses gathered cos/sin.
        if kept_attention_mask_arg is None:
            kept_position_ids = None

        if self.layer_type == "linear_attention":
            # GDN uses padding mask + its own cache; never call full-attn causal mask here.
            kept_attention_mask = kept_attention_mask_arg
        else:
            kept_attention_mask = _create_layer_causal_mask(
                config=self.config,
                inputs_embeds=kept_tokens,
                attention_mask=kept_attention_mask_arg,
                past_key_values=past_key_values,
                position_ids=kept_position_ids,
                layer_idx=self.layer_idx,
            )

        # Keep/pack is outside the checkpoint region. Only packed attn+MLP is recomputed, so
        # packing length cannot drift (and router stays on the normal FSDP forward path).
        def _packed_attn_mlp(packed_tokens: torch.Tensor, packed_cos: torch.Tensor, packed_sin: torch.Tensor):
            return self.forward_return_residual(
                packed_tokens,
                attention_mask=kept_attention_mask,
                position_ids=kept_position_ids,
                past_key_values=past_key_values,
                position_embeddings=(packed_cos, packed_sin),
                **packed_attn_kwargs,
            )

        response_start = kwargs.get("samod_response_start_idx")
        use_packed_incremental = (
            response_start is not None
            and not has_layer_past
        )
        if use_packed_incremental:
            prefix_pack_lens = _prefix_pack_lengths(kept_tokens_mask, response_start)
            has_suffix = bool((prefix_pack_lens < kept_tokens_lengths).any().item())
        else:
            has_suffix = False

        if self.gradient_checkpointing and self.training:
            mod_outputs, residual = torch_checkpoint(
                _packed_attn_mlp,
                kept_tokens,
                kept_cos,
                kept_sin,
                use_reentrant=False,
            )
        elif has_suffix:
            mod_outputs, residual = self._forward_packed_prefix_suffix_incremental(
                kept_tokens,
                kept_cos,
                kept_sin,
                kept_attention_mask,
                kept_attention_mask_2d,
                kept_attention_mask_arg,
                kept_position_ids,
                prefix_pack_lens,
                **packed_attn_kwargs,
            )
        else:
            mod_outputs, residual = _packed_attn_mlp(kept_tokens, kept_cos, kept_sin)

        # Record compact KV validity for later cached decode (physical length == max_kept).
        if (
            past_key_values is not None
            and not isinstance(past_key_values, SAMoDRaggedCache)
            and self.layer_type == "full_attention"
        ):
            meta = _ensure_samod_cache_meta(past_key_values)
            if meta is not None:
                if kept_attention_mask_arg is None:
                    meta["layer_valid"][self.layer_idx] = None
                else:
                    meta["layer_valid"][self.layer_idx] = kept_attention_mask_2d.detach().clone()

        hidden_states = self._string_writeback(
            h_in,
            mod_outputs,
            residual,
            kept_w,
            kept_tokens_mask,
            kept_attention_mask_2d,
            kept_attention_mask_arg,
            router_weights=router_weights,
            dropped_vision_mask=vision_token_mask & ~kept_mod_target_mask,
            reweight_skipped=self.reweight_skipped,
        )

        return (
            hidden_states,
            samod_score_sum,
            samod_score_count,
            samod_passed_vision,
            samod_total_vision,
            ohisa_keep_mask,
        )


class Qwen3_5SAMoDTextModel(Qwen3_5TextModel):
    _no_split_modules = ["Qwen3_5SAMoDDecoderLayer"]

    def __init__(self, config: Qwen3_5TextConfig):
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)

        target_indices, keep_args = _resolve_samod_schedule(config)
        config.samod_target_layer_indices = target_indices
        config.samod_margin = _get_samod_margin(config)
        _normalize_samod_sparsity_target(config)
        target_index_set = set(target_indices)

        layers = []
        for layer_idx in range(config.num_hidden_layers):
            keep_arg = keep_args[layer_idx] if layer_idx in target_index_set else None
            layers.append(Qwen3_5SAMoDDecoderLayer(config, layer_idx, keep_arg=keep_arg))

        self.layers = nn.ModuleList(layers)
        self.norm = Qwen3_5RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3_5TextRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()
        _reset_samod_router_parameters(self)

    def _init_weights(self, module):
        _samod_init_router_weights(self, module, super()._init_weights)


    def forward_layer(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        attention_mask_2d: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        past_key_values: Optional[Cache] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run a single dense decoder layer transition F_l(H_l)."""
        decoder_layer = self.layers[layer_idx]
        if position_embeddings is None:
            position_embeddings = getattr(self, "_samod_last_position_embeddings", None)
        if position_ids is None:
            position_ids = getattr(self, "_samod_last_text_position_ids", None)
        if attention_mask is None:
            if getattr(self.config, "layer_types", None) is not None and self.config.layer_types[layer_idx] == "linear_attention":
                attention_mask = getattr(self, "_samod_last_linear_attn_mask", None)
            else:
                attention_mask = getattr(self, "_samod_last_causal_mask", None)
        if attention_mask_2d is None:
            attention_mask_2d = getattr(self, "_samod_last_attention_mask_2d", None)
        # Dense teacher layers are stock Qwen3_5DecoderLayer (no SAMoD packing).
        return decoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        visual_segment_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Actor rmpad packs samples to ``[1, nnz]`` + ``samod_cu_seqlens``. Text SAMoD's
        # existing pack path isolates samples via batch dim ``[B, max_len]`` (required for
        # linear_attention). Unflatten here so the non-varlen path stays bit-identical, then
        # flatten outputs back for ``pad_input``.
        # Model may already unflatten (``samod_varlen_cu_seqlens`` + valid mask) to avoid a
        # second pad round-trip for M-RoPE.
        samod_cu_seqlens = kwargs.pop("samod_cu_seqlens", None)
        samod_varlen_cu = kwargs.pop("samod_varlen_cu_seqlens", None)
        samod_varlen_valid = kwargs.pop("samod_varlen_valid_mask", None)
        varlen_valid_mask: Optional[torch.Tensor] = None
        if samod_varlen_cu is not None:
            samod_cu_seqlens = samod_varlen_cu
            varlen_valid_mask = samod_varlen_valid
            if varlen_valid_mask is None and attention_mask is not None and attention_mask.ndim == 2:
                varlen_valid_mask = attention_mask.to(dtype=torch.bool)
        elif samod_cu_seqlens is not None:
            if inputs_embeds.ndim < 2 or inputs_embeds.shape[0] != 1:
                raise ValueError(
                    "samod_cu_seqlens requires flat rmpad inputs with batch dim 1, "
                    f"got inputs_embeds.shape={tuple(inputs_embeds.shape)}"
                )
            total_tokens = int(inputs_embeds.shape[1])
            samod_cu_seqlens = _normalize_cu_seqlens(samod_cu_seqlens, total_tokens, inputs_embeds.device)
            inputs_embeds, varlen_valid_mask = _varlen_to_padded_batch(inputs_embeds, samod_cu_seqlens)
            if attention_mask is None:
                attention_mask = varlen_valid_mask.to(dtype=torch.long)
            else:
                attention_mask, _ = _varlen_to_padded_batch(attention_mask, samod_cu_seqlens)
            if position_ids is not None:
                position_ids = _unflatten_mrope_position_ids(position_ids, samod_cu_seqlens)
            if visual_pos_masks is not None:
                visual_pos_masks, _ = _varlen_to_padded_batch(visual_pos_masks, samod_cu_seqlens)
                visual_pos_masks = visual_pos_masks.to(dtype=torch.bool)
            if visual_segment_ids is not None:
                visual_segment_ids, _ = _varlen_to_padded_batch(visual_segment_ids, samod_cu_seqlens)
                visual_segment_ids = visual_segment_ids.to(dtype=torch.long)

        use_single_ragged_cache = int(inputs_embeds.shape[0]) == 1
        if use_cache and use_single_ragged_cache and (
            past_key_values is None
            or (
                isinstance(past_key_values, DynamicCache)
                and not isinstance(past_key_values, SAMoDRaggedCache)
                and _samod_cache_is_empty(past_key_values)
            )
        ):
            # HF generate may pre-create and pass an empty DynamicCache. Replace
            # it before prefill so the real generation path uses compact
            # single-sample cache semantics too.
            past_key_values = SAMoDRaggedCache(config=self.config)
        elif use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        samod_meta = _ensure_samod_cache_meta(past_key_values) if use_cache else None
        is_samod_prefill = _samod_cache_is_empty(past_key_values)

        if position_ids is None:
            # Use logical (unpacked) length for RoPE — compact KV length differs when vision is dropped.
            if samod_meta is not None and int(samod_meta.get("logical_seen", 0)) > 0:
                past_seen_tokens = int(samod_meta["logical_seen"])
            else:
                past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            position_ids = position_ids[1:]
        else:
            text_position_ids = None

        attention_mask_2d = attention_mask if _attention_mask_has_padding(attention_mask) else None
        return_samod_trajectory = bool(kwargs.pop("return_samod_trajectory", False))
        return_samod_metrics = bool(
            kwargs.pop(
                "return_samod_metrics",
                getattr(self.config, "samod_return_metrics", True),
            )
        )
        # Shared vision pack scaffold for all SAMoD layers / routers in this forward.
        samod_vision_pack = None
        if visual_pos_masks is not None:
            vision_for_pack = visual_pos_masks.to(dtype=torch.bool)
            if attention_mask_2d is not None:
                vision_for_pack = vision_for_pack & attention_mask_2d.to(
                    device=vision_for_pack.device, dtype=torch.bool
                )
            samod_vision_pack = _build_vision_pack_scaffold(
                vision_for_pack,
                segment_ids=visual_segment_ids,
            )

        # Skip materializing a dense causal mask only when every layer will rebuild its own
        # packed/compact mask (sparse SAMoD eval/generate). No-vision paths fall through to
        # stock decoder layers and must keep padding-aware causal masks, otherwise
        # multi-sample full_attention leaks across the batch.
        all_samod = all(
            bool(getattr(layer, "samod_enabled", False))
            for layer in self.layers[: self.config.num_hidden_layers]
        )
        has_vision = samod_vision_pack is not None and int(samod_vision_pack["max_len"]) > 0
        sparse_samod_active = all_samod and has_vision
        is_single_ragged_decode = (
            isinstance(past_key_values, SAMoDRaggedCache)
            and not _samod_cache_is_empty(past_key_values)
        )
        # Masked SAMoD builds per-layer masks internally; still need padding-aware causal
        # mask when batch has left/right pad (multi-sample rollout / training).
        if is_single_ragged_decode or (
            sparse_samod_active and not self.training and attention_mask_2d is None
        ):
            causal_mask = None
        else:
            causal_mask = create_causal_mask(
                config=self.config,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_ids=text_position_ids,
            )
        linear_attn_mask = (
            None
            if is_single_ragged_decode
            else self._update_linear_attn_mask(attention_mask, past_key_values)
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        # Cache for teacher on-policy layer calls (dense transition F_l^T) — training only.
        if self.training:
            self._samod_last_position_embeddings = position_embeddings
            self._samod_last_text_position_ids = text_position_ids
            self._samod_last_attention_mask_2d = attention_mask_2d
            self._samod_last_causal_mask = causal_mask
            self._samod_last_linear_attn_mask = linear_attn_mask
        log_vision_token_retention = bool(getattr(self.config, "samod_log_vision_token_retention", True))
        compute_samod_score = self.training and _get_samod_dreg_loss_weight(self.config) != 0.0
        samod_score_stats: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        samod_usage_stats: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        traj_layer_indices: list[int] = []
        traj_h_in: list[torch.Tensor] = []
        traj_h_out: list[torch.Tensor] = []
        traj_keep_masks: list[torch.Tensor] = []

        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            layer_mask = linear_attn_mask if self.config.layer_types[layer_idx] == "linear_attention" else causal_mask
            layer_kwargs = dict(
                attention_mask=layer_mask,
                position_ids=text_position_ids,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )
            if is_single_ragged_decode and decoder_layer.samod_enabled:
                # Same decode used with or without OHiSA. Recording the
                # inputs and outputs must not switch this path.
                h_in = hidden_states if return_samod_trajectory else None
                hidden_states = decoder_layer._forward_samod_ragged_dense_decode(
                    hidden_states,
                    position_embeddings=position_embeddings,
                    position_ids=text_position_ids,
                    past_key_values=past_key_values,
                    **kwargs,
                )
                if return_samod_trajectory:
                    traj_layer_indices.append(layer_idx)
                    traj_h_in.append(h_in)
                    traj_h_out.append(hidden_states)
                    traj_keep_masks.append(_ohisa_sequence_keep_mask(h_in, attention_mask_2d))
                continue
            h_in = hidden_states
            if decoder_layer.samod_enabled:
                layer_kwargs["attention_mask_2d"] = attention_mask_2d
                layer_kwargs["visual_pos_masks"] = visual_pos_masks
                if samod_vision_pack is not None:
                    layer_kwargs["samod_vision_pack"] = samod_vision_pack
                (
                    hidden_states,
                    score_sum,
                    score_count,
                    passed_vision,
                    total_vision,
                    keep_mask,
                ) = decoder_layer(hidden_states, **layer_kwargs)
                if compute_samod_score:
                    drop_sum = getattr(
                        decoder_layer,
                        "_samod_score_drop_sum",
                        score_sum.new_zeros(()),
                    )
                    samod_score_stats.append((score_sum, drop_sum, score_count))
                # Vision token retention is prefill / training-forward only; decode returns placeholders.
                if is_samod_prefill and passed_vision is not None and total_vision is not None:
                    samod_usage_stats.append((layer_idx, passed_vision.detach(), total_vision.detach()))
                if return_samod_trajectory:
                    traj_layer_indices.append(layer_idx)
                    traj_h_in.append(h_in)
                    # Align full layer outputs (post-MLP / branch weight), same as original OHiSA.
                    traj_h_out.append(hidden_states)
                    traj_keep_masks.append(keep_mask)
            else:
                hidden_states = decoder_layer(hidden_states, **layer_kwargs)

        hidden_states = self.norm(hidden_states)

        # Track logical sequence length for decode RoPE (independent of compact KV lengths).
        if samod_meta is not None:
            # Prefer true nnz when we entered via rmpad; otherwise padded max_len.
            if samod_cu_seqlens is not None:
                q_len = int(samod_cu_seqlens[-1].item())
            else:
                q_len = int(inputs_embeds.shape[1])
            if is_samod_prefill:
                samod_meta["logical_seen"] = q_len
            else:
                samod_meta["logical_seen"] = int(samod_meta.get("logical_seen", 0)) + q_len

        if samod_cu_seqlens is not None and varlen_valid_mask is not None:
            # Logits still need the flat ``[1, nnz]`` stream. OHiSA trajectory
            # stays on the padded ``[B, max_len]`` layout so layer hiddens, RoPE,
            # and the linear-attention padding mask share one sequence length.
            hidden_states = _padded_batch_to_varlen(hidden_states, samod_cu_seqlens, varlen_valid_mask)

        output = Qwen3_5ModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
        samod_dreg_loss = None
        samod_dreg_loss_per_sample = None
        if compute_samod_score:
            samod_dreg_loss, samod_dreg_loss_per_sample = _build_samod_dreg_loss(
                samod_score_stats, self.config, samod_usage_stats
            )
        output = _attach_samod_dreg_loss(output, samod_dreg_loss)
        if samod_dreg_loss_per_sample is not None:
            setattr(output, "samod_dreg_loss_per_sample", samod_dreg_loss_per_sample)
        if return_samod_metrics and log_vision_token_retention and is_samod_prefill:
            output = _attach_extra_metrics(
                output,
                _build_samod_extra_metrics(
                    samod_usage_stats,
                    include_per_layer=not self.training,
                ),
            )
        if return_samod_trajectory:
            output = _attach_samod_trajectory(
                output,
                {
                    "layer_indices": traj_layer_indices,
                    "hidden_states_in": traj_h_in,
                    "hidden_states_out": traj_h_out,
                    "keep_masks": traj_keep_masks,
                    "position_embeddings": position_embeddings,
                    "text_position_ids": text_position_ids,
                    "attention_mask_2d": attention_mask_2d,
                    "causal_mask": causal_mask,
                    "linear_attn_mask": linear_attn_mask,
                    "visual_pos_masks": visual_pos_masks,
                },
            )
        return output


def _sequence_dim(tensor: torch.Tensor, length: int, other: Optional[torch.Tensor] = None, other_length: Optional[int] = None) -> int:
    matches = [axis for axis, size in enumerate(tensor.shape) if int(size) == int(length)]
    if other is not None and other_length is not None:
        matches = [axis for axis in matches if int(other.shape[axis]) == int(other_length)]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one sequence axis of length {length} in shape {tuple(tensor.shape)}, found {matches}"
        )
    return matches[0]


def _pad_sequence_dim(tensor: torch.Tensor, dim: int, total: int) -> torch.Tensor:
    current = int(tensor.shape[dim])
    if current == total:
        return tensor
    if current > total:
        raise RuntimeError(f"Cannot pad sequence {current} down to {total}")
    pad_shape = list(tensor.shape)
    pad_shape[dim] = total - current
    if tensor.dtype == torch.bool:
        filler = torch.zeros(pad_shape, dtype=torch.bool, device=tensor.device)
    else:
        filler = tensor.new_zeros(pad_shape)
    return torch.cat((tensor, filler), dim=dim)


def _align_keep_to_hidden(keep: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    """Decode masks can cover the cached prefix while hidden is only the new token."""
    keep = keep.to(device=hidden.device, dtype=torch.bool)
    if keep.shape == hidden.shape[:2]:
        return keep
    if keep.ndim == 2 and keep.shape[0] == hidden.shape[0] and keep.shape[1] > hidden.shape[1]:
        return keep[:, -hidden.shape[1] :]
    raise RuntimeError(
        f"OHiSA keep shape {tuple(keep.shape)} does not match hidden {tuple(hidden.shape)}"
    )


def _stitch_response_window_trajectory(
    prefill: dict,
    decodes: list,
    *,
    seqlen: int,
    attention_mask: Optional[torch.Tensor],
    position_ids: Optional[torch.Tensor],
    visual_pos_masks: Optional[torch.Tensor],
    language_model: nn.Module,
) -> dict:
    """Place prompt-prefill and token-decode layer states on one ``[B, S]`` layout.

    The student forward itself is unchanged. Uncomputed tail positions (the last
    response token is not run, because logits only need the previous hidden)
    stay zero and are marked not-kept, so the response MSE ignores them.
    """
    pieces = [prefill, *decodes]
    layer_indices = list(prefill["layer_indices"])
    for part in decodes:
        if list(part["layer_indices"]) != layer_indices:
            raise RuntimeError("OHiSA decode trajectory layers do not match the prefill")
    hidden_in = []
    hidden_out = []
    keep_masks = []
    for layer_i in range(len(layer_indices)):
        inputs = []
        outputs = []
        keeps = []
        for part in pieces:
            incoming = part["hidden_states_in"][layer_i]
            inputs.append(incoming)
            outputs.append(part["hidden_states_out"][layer_i])
            keeps.append(_align_keep_to_hidden(part["keep_masks"][layer_i], incoming))
        hidden_in.append(_pad_sequence_dim(torch.cat(inputs, dim=1), 1, seqlen))
        hidden_out.append(_pad_sequence_dim(torch.cat(outputs, dim=1), 1, seqlen))
        keep_masks.append(_pad_sequence_dim(torch.cat(keeps, dim=1), 1, seqlen))

    position_embeddings = None
    if prefill.get("position_embeddings") is not None:
        stitched = []
        for comp_i in range(len(prefill["position_embeddings"])):
            parts = [part["position_embeddings"][comp_i] for part in pieces]
            other = parts[1] if len(parts) > 1 else None
            other_length = pieces[1]["hidden_states_in"][0].shape[1] if len(parts) > 1 else None
            seq_dim = _sequence_dim(
                parts[0],
                pieces[0]["hidden_states_in"][0].shape[1],
                other,
                other_length,
            )
            for part, tensor in zip(pieces[1:], parts[1:]):
                if int(tensor.shape[seq_dim]) != int(part["hidden_states_in"][0].shape[1]):
                    raise RuntimeError("OHiSA position embeddings use different sequence axes")
            stitched.append(_pad_sequence_dim(torch.cat(parts, dim=seq_dim), seq_dim, seqlen))
        position_embeddings = tuple(stitched)

    text_position_ids = None
    if position_ids is not None:
        if position_ids.ndim == 3 and int(position_ids.shape[0]) == 4:
            text_position_ids = position_ids[0]
        elif position_ids.ndim == 2:
            text_position_ids = position_ids
        elif position_ids.ndim == 3:
            text_position_ids = position_ids[0]

    attention_mask_2d = None
    if attention_mask is not None and attention_mask.ndim == 2 and not bool(torch.all(attention_mask)):
        attention_mask_2d = attention_mask.to(dtype=torch.bool)

    causal_mask = create_causal_mask(
        config=language_model.config,
        inputs_embeds=hidden_in[0],
        attention_mask=attention_mask,
        past_key_values=None,
        position_ids=text_position_ids,
    )
    linear_attn_mask = language_model._update_linear_attn_mask(attention_mask, None)
    return {
        "layer_indices": layer_indices,
        "hidden_states_in": hidden_in,
        "hidden_states_out": hidden_out,
        "keep_masks": keep_masks,
        "position_embeddings": position_embeddings,
        "text_position_ids": text_position_ids,
        "attention_mask_2d": attention_mask_2d,
        "causal_mask": causal_mask,
        "linear_attn_mask": linear_attn_mask,
        "visual_pos_masks": visual_pos_masks,
    }


class Qwen3_5SAMoDModel(Qwen3_5Model):
    _no_split_modules = ["Qwen3_5SAMoDDecoderLayer", "Qwen3_5VisionBlock"]

    def __init__(self, config):
        Qwen3_5PreTrainedModel.__init__(self, config)
        self._no_split_modules = _get_samod_no_split_modules(config)
        self.visual = Qwen3_5VisionModel._from_config(config.vision_config)
        self.language_model = Qwen3_5SAMoDTextModel._from_config(config.text_config)
        self.rope_deltas = None
        self.post_init()
        _reset_samod_router_parameters(self)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        **kwargs,
    ) -> Union[tuple, Qwen3_5ModelOutputWithPast]:
        response_length = kwargs.pop("response_length", None)
        response_start_idx = kwargs.pop("response_start_idx", None)
        rmpad_indices = kwargs.pop("rmpad_indices", None)
        padded_seqlen = kwargs.pop("padded_seqlen", None)
        padded_batch_size = kwargs.pop("padded_batch_size", None)
        samod_cu_seqlens = kwargs.pop("samod_cu_seqlens", None)

        if (
            response_length is not None
            and response_start_idx is not None
            and past_key_values is None
        ):
            if input_ids is None:
                raise ValueError("response logits forward requires input_ids")
            if (
                samod_cu_seqlens is not None
                and input_ids.ndim == 2
                and int(input_ids.shape[0]) == 1
                and rmpad_indices is not None
                and padded_seqlen is not None
                and padded_batch_size is not None
            ):
                input_ids, attention_mask, mm_token_type_ids, position_ids = _restore_rmpad_padded_batch(
                    input_ids=input_ids,
                    rmpad_indices=rmpad_indices,
                    padded_batch_size=int(padded_batch_size),
                    padded_seqlen=int(padded_seqlen),
                    attention_mask=attention_mask,
                    mm_token_type_ids=mm_token_type_ids,
                    position_ids=position_ids,
                )
            elif attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)

            if (input_ids is None) ^ (inputs_embeds is not None):
                raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings()(input_ids)

            visual_pos_masks = None
            visual_segment_ids = torch.zeros(
                inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long
            )
            has_visual_segments = False
            if mm_token_type_ids is not None:
                visual_pos_masks = mm_token_type_ids.to(device=inputs_embeds.device) != 0

            if pixel_values is not None:
                image_outputs: BaseModelOutputWithPooling = self.get_image_features(
                    pixel_values, image_grid_thw, return_dict=True
                )
                image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                image_mask, _ = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds
                )
                visual_pos_masks = (
                    image_mask[..., 0] if visual_pos_masks is None else (visual_pos_masks | image_mask[..., 0])
                )
                _append_visual_grid_segments(
                    visual_segment_ids,
                    image_mask,
                    image_grid_thw,
                    self.config.vision_config.spatial_merge_size,
                )
                has_visual_segments = image_grid_thw is not None or has_visual_segments
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_outputs: BaseModelOutputWithPooling = self.get_video_features(
                    pixel_values_videos, video_grid_thw, return_dict=True
                )
                video_embeds = torch.cat(video_outputs.pooler_output, dim=0).to(
                    inputs_embeds.device, inputs_embeds.dtype
                )
                _, video_mask = self.get_placeholder_mask(
                    input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds
                )
                visual_pos_masks = (
                    video_mask[..., 0] if visual_pos_masks is None else (visual_pos_masks | video_mask[..., 0])
                )
                _append_visual_grid_segments(
                    visual_segment_ids,
                    video_mask,
                    video_grid_thw,
                    self.config.vision_config.spatial_merge_size,
                )
                has_visual_segments = video_grid_thw is not None or has_visual_segments
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if not has_visual_segments:
                visual_segment_ids = None

            if position_ids is None:
                position_ids = self.compute_3d_position_ids(
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=None,
                    mm_token_type_ids=mm_token_type_ids,
                )

            bsz, seqlen = input_ids.shape
            r_len = int(response_length)
            rsi = response_start_idx.to(device=input_ids.device, dtype=torch.long)
            if rsi.shape != (bsz,):
                raise ValueError(f"response_start_idx must be shape ({bsz},), got {tuple(rsi.shape)}")
            if (rsi < 1).any():
                raise ValueError("response_start_idx must be >= 1")

            w_max = int(rsi.max().item())
            if w_max + r_len > seqlen:
                raise ValueError(f"response window out of bounds: w_max={w_max}, r_len={r_len}, seqlen={seqlen}")

            prompt_embeds = inputs_embeds[:, :w_max]
            prompt_attn = attention_mask[:, :w_max]
            prompt_pos = position_ids[..., :w_max] if position_ids.ndim == 3 else position_ids[:, :w_max]
            prompt_vmask = visual_pos_masks[:, :w_max] if visual_pos_masks is not None else None
            prompt_segment_ids = (
                visual_segment_ids[:, :w_max] if visual_segment_ids is not None else None
            )
            response_ids = input_ids[:, w_max : w_max + r_len]

            lm_kwargs = dict(kwargs)
            want_traj = bool(lm_kwargs.get("return_samod_trajectory", False))
            for key in (
                "samod_response_start_idx",
                "output_hidden_states",
                "use_cache",
            ):
                lm_kwargs.pop(key, None)
            if not want_traj:
                # Keep the backbone call identical to the non-OHiSA rollout.
                lm_kwargs.pop("return_samod_trajectory", None)

            prefill_out = self.language_model(
                input_ids=None,
                inputs_embeds=prompt_embeds,
                attention_mask=prompt_attn,
                position_ids=prompt_pos,
                visual_pos_masks=prompt_vmask,
                visual_segment_ids=prompt_segment_ids,
                use_cache=True,
                past_key_values=None,
                **lm_kwargs,
            )
            past_key_values = prefill_out.past_key_values
            gather_idx = (rsi - 1).view(bsz, 1, 1).expand(-1, 1, prefill_out.last_hidden_state.shape[-1])
            response_hiddens = [torch.gather(prefill_out.last_hidden_state, dim=1, index=gather_idx).squeeze(1)]
            embed_tokens = self.get_input_embeddings()
            decode_outs = [] if want_traj else None
            for step in range(1, r_len):
                tok_embed = embed_tokens(response_ids[:, step - 1 : step])
                pos_col = w_max + step - 1
                step_pos = (
                    position_ids[..., pos_col : pos_col + 1]
                    if position_ids.ndim == 3
                    else position_ids[:, pos_col : pos_col + 1]
                )
                decode_out = self.language_model(
                    input_ids=None,
                    inputs_embeds=tok_embed,
                    attention_mask=attention_mask[:, : w_max + step],
                    position_ids=step_pos,
                    visual_pos_masks=None,
                    use_cache=True,
                    past_key_values=past_key_values,
                    **lm_kwargs,
                )
                past_key_values = decode_out.past_key_values
                response_hiddens.append(decode_out.last_hidden_state[:, -1, :])
                if decode_outs is not None:
                    decode_outs.append(decode_out)

            window_out = Qwen3_5ModelOutputWithPast(
                last_hidden_state=torch.stack(response_hiddens, dim=1),
                past_key_values=None,
            )
            if want_traj:
                prefill_traj = getattr(prefill_out, "samod_trajectory", None)
                if prefill_traj is None:
                    raise RuntimeError("OHiSA response-window prefill did not return samod_trajectory")
                decode_trajs = []
                for decode_out in decode_outs:
                    decode_traj = getattr(decode_out, "samod_trajectory", None)
                    if decode_traj is None:
                        raise RuntimeError("OHiSA response-window decode did not return samod_trajectory")
                    decode_trajs.append(decode_traj)
                window_out = _attach_samod_trajectory(
                    window_out,
                    _stitch_response_window_trajectory(
                        prefill_traj,
                        decode_trajs,
                        seqlen=seqlen,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        visual_pos_masks=visual_pos_masks,
                        language_model=self.language_model,
                    ),
                )
            return window_out

        if samod_cu_seqlens is not None:
            kwargs["samod_cu_seqlens"] = samod_cu_seqlens

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        visual_pos_masks = None
        visual_segment_ids = torch.zeros(
            inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long
        )
        has_visual_segments = False
        if mm_token_type_ids is not None:
            visual_pos_masks = mm_token_type_ids.to(device=inputs_embeds.device) != 0

        if pixel_values is not None:
            image_outputs: BaseModelOutputWithPooling = self.get_image_features(
                pixel_values,
                image_grid_thw,
                return_dict=True,
            )
            image_embeds = image_outputs.pooler_output
            image_embeds = torch.cat(image_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            image_mask, _ = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                image_features=image_embeds,
            )
            visual_pos_masks = image_mask[..., 0] if visual_pos_masks is None else (visual_pos_masks | image_mask[..., 0])
            _append_visual_grid_segments(
                visual_segment_ids,
                image_mask,
                image_grid_thw,
                self.config.vision_config.spatial_merge_size,
            )
            has_visual_segments = image_grid_thw is not None or has_visual_segments
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_outputs: BaseModelOutputWithPooling = self.get_video_features(
                pixel_values_videos,
                video_grid_thw,
                return_dict=True,
            )
            video_embeds = video_outputs.pooler_output
            video_embeds = torch.cat(video_embeds, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            _, video_mask = self.get_placeholder_mask(
                input_ids,
                inputs_embeds=inputs_embeds,
                video_features=video_embeds,
            )
            visual_pos_masks = video_mask[..., 0] if visual_pos_masks is None else (visual_pos_masks | video_mask[..., 0])
            _append_visual_grid_segments(
                visual_segment_ids,
                video_mask,
                video_grid_thw,
                self.config.vision_config.spatial_merge_size,
            )
            has_visual_segments = video_grid_thw is not None or has_visual_segments
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        if not has_visual_segments:
            visual_segment_ids = None

        # Flat rmpad ``[1, nnz]`` + cu_seqlens: unflatten once here (M-RoPE + TextModel),
        # so TextModel does not pad again.
        samod_cu_seqlens = kwargs.pop("samod_cu_seqlens", None)
        if (
            samod_cu_seqlens is not None
            and inputs_embeds is not None
            and inputs_embeds.ndim >= 2
            and int(inputs_embeds.shape[0]) == 1
            and int(samod_cu_seqlens.reshape(-1).numel()) > 2
        ):
            cu = _normalize_cu_seqlens(samod_cu_seqlens, int(inputs_embeds.shape[1]), inputs_embeds.device)
            inputs_embeds, pad_valid = _varlen_to_padded_batch(inputs_embeds, cu)
            if attention_mask is None:
                attention_mask = pad_valid.to(dtype=torch.long)
            else:
                attention_mask, _ = _varlen_to_padded_batch(attention_mask, cu)
            if visual_pos_masks is not None:
                visual_pos_masks, _ = _varlen_to_padded_batch(visual_pos_masks, cu)
                visual_pos_masks = visual_pos_masks.to(dtype=torch.bool)
            if visual_segment_ids is not None:
                visual_segment_ids, _ = _varlen_to_padded_batch(visual_segment_ids, cu)
            if mm_token_type_ids is not None and mm_token_type_ids.ndim == 2 and int(mm_token_type_ids.shape[0]) == 1:
                mm_token_type_ids, _ = _varlen_to_padded_batch(mm_token_type_ids, cu)
                mm_token_type_ids = mm_token_type_ids.to(dtype=mm_token_type_ids.dtype)
            if position_ids is None:
                pad_ids = None
                if input_ids is not None and input_ids.ndim == 2 and int(input_ids.shape[0]) == 1:
                    pad_ids, _ = _varlen_to_padded_batch(input_ids, cu)
                else:
                    # Fallback: embed-only position computation on padded embeds.
                    pad_ids = None
                position_ids = self.compute_3d_position_ids(
                    input_ids=pad_ids if pad_ids is not None else input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    mm_token_type_ids=mm_token_type_ids,
                )
            elif position_ids.ndim == 3 and int(position_ids.shape[1]) == 1:
                position_ids = _unflatten_mrope_position_ids(position_ids, cu)
            kwargs["samod_varlen_cu_seqlens"] = cu
            kwargs["samod_varlen_valid_mask"] = pad_valid
        elif position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            visual_segment_ids=visual_segment_ids,
            **kwargs,
        )

        output = Qwen3_5ModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
        )
        output = _attach_samod_dreg_loss(output, getattr(outputs, "samod_dreg_loss", None))
        per_sample_score = getattr(outputs, "samod_dreg_loss_per_sample", None)
        if per_sample_score is not None:
            setattr(output, "samod_dreg_loss_per_sample", per_sample_score)
        output = _attach_extra_metrics(output, getattr(outputs, "extra_metrics", None))
        output = _attach_samod_trajectory(output, getattr(outputs, "samod_trajectory", None))
        return output


class Qwen3_5SAMoDForConditionalGeneration(Qwen3_5ForConditionalGeneration):
    _no_split_modules = ["Qwen3_5SAMoDDecoderLayer", "Qwen3_5VisionBlock"]
    _supports_attention_backend = True
    _checkpoint_conversion_mapping = {}
    _tied_weights_keys = {"lm_head.weight": "model.language_model.embed_tokens.weight"}
    accepts_loss_kwargs = False

    def __init__(self, config):
        Qwen3_5PreTrainedModel.__init__(self, config)
        self._no_split_modules = _get_samod_no_split_modules(config)
        self.model = Qwen3_5SAMoDModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()
        _reset_samod_router_parameters(self)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        mm_token_type_ids: Optional[torch.IntTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[tuple, Qwen3_5CausalLMOutputWithPast]:
        # Distill path can skip materializing full-vocab logits (major activation peak).
        # ``distill_topk_indices`` computes JS top-k log-probs inside this FSDP forward so
        # ``lm_head`` stays on-graph without ever allocating logits[..., V] / grad_logits.
        distill_topk_indices = kwargs.pop("distill_topk_indices", None)
        skip_logits = bool(kwargs.pop("skip_logits", False)) or distill_topk_indices is not None
        logits_response_length = kwargs.pop("logits_response_length", None)
        logits_response_start_idx = kwargs.pop("logits_response_start_idx", None)
        logits_rmpad_indices = kwargs.pop("logits_rmpad_indices", None)
        logits_batch_size = kwargs.pop("logits_batch_size", None)
        logits_seqlen = kwargs.pop("logits_seqlen", None)
        samod_cu_seqlens = kwargs.pop("samod_cu_seqlens", None)
        response_logits = logits_response_length is not None and logits_response_start_idx is not None
        # OHiSA records layer states inside the same response-window rollout
        # the actor uses when the flag is off. Do not switch to a full-sequence pass.
        model_response_length = logits_response_length
        model_response_start_idx = logits_response_start_idx
        if not response_logits and logits_response_start_idx is not None:
            kwargs["samod_response_start_idx"] = logits_response_start_idx

        outputs = self.model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            mm_token_type_ids=mm_token_type_ids,
            samod_cu_seqlens=samod_cu_seqlens,
            response_length=model_response_length,
            response_start_idx=model_response_start_idx,
            rmpad_indices=logits_rmpad_indices,
            padded_seqlen=logits_seqlen,
            padded_batch_size=logits_batch_size,
            **kwargs,
        )

        hidden_states = outputs[0]
        topk_log_probs = None
        logits_are_response_window = False
        if distill_topk_indices is not None:
            from verl.workers.utils.ohisa import extract_topk_log_probs_from_hidden

            topk_log_probs, _ = extract_topk_log_probs_from_hidden(
                hidden_states[:, :-1, :],
                self.lm_head,
                topk=int(distill_topk_indices.shape[-1]),
                indices=distill_topk_indices,
            )
            logits = None
        elif skip_logits:
            logits = None
        elif response_logits:
            # Response-window backbone already returned response-only hiddens,
            # including when OHiSA asked for the layer trajectory.
            logits = self.lm_head(hidden_states)
            logits_are_response_window = True
        else:
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        samod_dreg_loss = getattr(outputs, "samod_dreg_loss", None)
        loss = None
        if labels is not None:
            if logits is None:
                raise ValueError("labels require logits; unset skip_logits / distill_topk_indices when passing labels")
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.text_config.vocab_size)
            if samod_dreg_loss is not None:
                loss = loss + samod_dreg_loss * _get_samod_dreg_loss_weight(self.config)

        output = Qwen3_5CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )
        # Always expose last backbone states for memory-efficient distill (lm_head outside).
        setattr(output, "last_hidden_state", hidden_states)
        setattr(output, "logits_are_response_window", logits_are_response_window)
        if topk_log_probs is not None:
            setattr(output, "topk_log_probs", topk_log_probs)
        output = _attach_samod_dreg_loss(output, samod_dreg_loss)
        per_sample_score = getattr(outputs, "samod_dreg_loss_per_sample", None)
        if per_sample_score is not None:
            setattr(output, "samod_dreg_loss_per_sample", per_sample_score)
        output = _attach_extra_metrics(output, getattr(outputs, "extra_metrics", None))
        output = _attach_samod_trajectory(output, getattr(outputs, "samod_trajectory", None))
        return output

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        is_first_iteration=False,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            use_cache=use_cache,
            is_first_iteration=is_first_iteration,
            **kwargs,
        )
        return model_inputs


class Qwen3_5SAMoDForCausalLM(Qwen3_5ForCausalLM):
    _no_split_modules = ["Qwen3_5SAMoDDecoderLayer"]
    _supports_attention_backend = True
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}
    _keys_to_ignore_on_load_unexpected = [r"^model.visual.*"]

    def __init__(self, config):
        Qwen3_5PreTrainedModel.__init__(self, config)
        self.model = Qwen3_5SAMoDTextModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
        _reset_samod_router_parameters(self)

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        skip_logits = bool(kwargs.pop("skip_logits", False))
        distill_topk_indices = kwargs.pop("distill_topk_indices", None)
        skip_logits = skip_logits or distill_topk_indices is not None
        logits_response_length = kwargs.pop("logits_response_length", None)
        logits_response_start_idx = kwargs.pop("logits_response_start_idx", None)
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        topk_log_probs = None
        logits_are_response_window = False
        if distill_topk_indices is not None:
            from verl.workers.utils.ohisa import extract_topk_log_probs_from_hidden

            topk_log_probs, _ = extract_topk_log_probs_from_hidden(
                hidden_states[:, :-1, :],
                self.lm_head,
                topk=int(distill_topk_indices.shape[-1]),
                indices=distill_topk_indices,
            )
            logits = None
        elif skip_logits:
            logits = None
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
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

        samod_dreg_loss = getattr(outputs, "samod_dreg_loss", None)
        loss = None
        if labels is not None:
            if logits is None:
                raise ValueError("labels require logits; unset skip_logits / distill_topk_indices when passing labels")
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            if samod_dreg_loss is not None:
                loss = loss + samod_dreg_loss * _get_samod_dreg_loss_weight(self.config)

        output = CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        setattr(output, "last_hidden_state", hidden_states)
        setattr(output, "logits_are_response_window", logits_are_response_window)
        if topk_log_probs is not None:
            setattr(output, "topk_log_probs", topk_log_probs)
        output = _attach_samod_dreg_loss(output, samod_dreg_loss)
        per_sample_score = getattr(outputs, "samod_dreg_loss_per_sample", None)
        if per_sample_score is not None:
            setattr(output, "samod_dreg_loss_per_sample", per_sample_score)
        output = _attach_extra_metrics(output, getattr(outputs, "extra_metrics", None))
        output = _attach_samod_trajectory(output, getattr(outputs, "samod_trajectory", None))
        return output


__all__ = [
    "SAMoDRaggedCache",
    "Qwen3_5SAMoDForConditionalGeneration",
    "Qwen3_5SAMoDForCausalLM",
    "Qwen3_5SAMoDModel",
    "Qwen3_5SAMoDTextModel",
    "Qwen3_5SAMoDDecoderLayer",
    "Qwen3_5SAMoDTokenRouter",
    "configure_samod",
]
