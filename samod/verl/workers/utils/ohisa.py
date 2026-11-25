# Copyright 2025 Bytedance Ltd. and/or its affiliates
"""Hidden Trajectory OPD helpers for off-policy SFT distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import compute_self_distillation_loss
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.torch_functional import masked_mean, masked_sum

TASK_LOSS_FORMS = ("sft", "off_distill", "on_distill")


@dataclass
class SelfDistillationCfg:
    full_logit_distillation: bool = True
    divergence_type: str = "jsd"
    distillation_topk: Optional[int] = 100
    distillation_add_tail: bool = True
    alpha: float = 0.5
    is_clip: Optional[float] = None


@dataclass
class OnDistillCfg:
    """Controls actor HF generate for ``task_loss_form=on_distill``."""

    max_new_tokens: int = 512
    temperature: float = 1.0
    top_p: float = 1.0
    do_sample: bool = True
    # If True, ban image/video placeholder tokens during generation.
    suppress_mm_tokens: bool = True


def unwrap_model(module: torch.nn.Module) -> torch.nn.Module:
    """Unwrap common distributed wrappers."""
    while True:
        if hasattr(module, "module"):
            module = module.module
            continue
        break
    return module


@dataclass(frozen=True)
class OhisaRuntimeCfg:
    """Resolved on-policy OHiSA settings from student HF ``text_config``."""

    loss_weight: float = 0.0
    # True when the HF config exposes ``ohisa_loss_weight`` (even if 0).
    configured: bool = False

    @property
    def enabled(self) -> bool:
        """Whether λ scales L_hidden into the optim objective."""
        return float(self.loss_weight) != 0.0

    @property
    def track(self) -> bool:
        """Collect trajectory / compute L_hidden for logging (independent of λ)."""
        return bool(self.configured) or self.enabled

    @property
    def mode(self) -> str:
        return "on_only" if self.track else "disabled"


def resolve_ohisa_from_text_config(text_config: Any) -> OhisaRuntimeCfg:
    """Read OHiSA weight from HF ``text_config`` (or a plain mapping).

    The hidden term is always on-policy: ``MSE(h_a^{l+1}, F_l^T(h_a^l))``.
    Config exposes ``ohisa_loss_weight`` → track and log. Weight 0 still logs
    and does not enter the optimizer.
    """
    if text_config is None:
        return OhisaRuntimeCfg()

    def _get(name: str, default: Any = None) -> Any:
        if isinstance(text_config, dict):
            return text_config.get(name, default)
        return getattr(text_config, name, default)

    configured = _ohisa_config_has_weight(text_config)
    weight = float(_get("ohisa_loss_weight", 0.0) or 0.0)
    return OhisaRuntimeCfg(loss_weight=weight, configured=configured)


def _ohisa_config_has_weight(sub: Any) -> bool:
    if sub is None:
        return False
    if isinstance(sub, dict):
        return "ohisa_loss_weight" in sub
    return hasattr(sub, "ohisa_loss_weight")


def resolve_ohisa_from_module(module: torch.nn.Module) -> OhisaRuntimeCfg:
    """Resolve OHiSA settings from an (possibly wrapped) HF module.

    Qwen stores the knobs on ``text_config``. NEO stores them on ``llm_config``.
    A nested config that actually has ``ohisa_loss_weight`` wins; otherwise
    the reader falls through to the root config and the weight stays 0.
    """
    root = unwrap_model(module)
    cfg = getattr(root, "config", None)
    if cfg is None:
        return OhisaRuntimeCfg()
    for name in ("text_config", "llm_config"):
        sub = getattr(cfg, name, None)
        if _ohisa_config_has_weight(sub):
            return resolve_ohisa_from_text_config(sub)
    return resolve_ohisa_from_text_config(cfg)


def _decoder_with_layers(module: torch.nn.Module) -> Optional[torch.nn.Module]:
    """Prefer a module that owns ``.layers``; NEO keeps them on ``language_model.model``."""
    if hasattr(module, "layers"):
        return module
    inner = getattr(module, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner
    return None


def get_text_model(module: torch.nn.Module) -> torch.nn.Module:
    """Return the text decoder stack (``language_model`` or ``model``)."""
    root = unwrap_model(module)
    if hasattr(root, "model") and hasattr(root.model, "language_model"):
        found = _decoder_with_layers(root.model.language_model)
        if found is not None:
            return found
    if hasattr(root, "language_model"):
        found = _decoder_with_layers(root.language_model)
        if found is not None:
            return found
    if hasattr(root, "model") and hasattr(root.model, "layers"):
        return root.model
    if hasattr(root, "layers"):
        return root
    raise AttributeError(f"Cannot locate text decoder layers on {type(root)}")


def teacher_forward_layer(
    teacher_module: torch.nn.Module,
    layer_idx: int,
    hidden_states: torch.Tensor,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    **kwargs,
) -> torch.Tensor:
    """Compute dense one-step transition ``H_{l+1} = F_l^T(H_l)`` under no_grad caller."""
    text_model = get_text_model(teacher_module)
    layer = text_model.layers[layer_idx]
    layer_kwargs = dict(kwargs)
    if position_embeddings is not None:
        layer_kwargs["position_embeddings"] = position_embeddings
    if position_ids is not None:
        layer_kwargs["position_ids"] = position_ids
    if attention_mask is not None:
        layer_kwargs["attention_mask"] = attention_mask
    out = layer(hidden_states, **layer_kwargs)
    if isinstance(out, tuple):
        return out[0]
    return out


def _flatten_loss_mask(data: TensorDict) -> tuple[torch.Tensor, bool]:
    """Return a 1D/2D loss mask aligned to next-token logits (``logits[:, :-1]``).

    Dataset ``loss_mask`` marks target tokens (assistant). For next-token JS/CE we
    supervise position ``i`` when predicting token ``i+1``, so the mask is
    ``loss_mask[..., 1:]`` (length ``S-1``), matching ``logits[:, :-1]``.
    """
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    pad_mode_str = str(getattr(pad_mode, "value", pad_mode))
    loss_mask = data["loss_mask"]
    if pad_mode_str in {"no_padding", "DatasetPadMode.NO_PADDING"} and hasattr(loss_mask, "is_nested") and loss_mask.is_nested:
        # Packed stream: values()[1:] aligns with logits[:, :-1] on the same packing.
        loss_mask_flat = loss_mask.values()[1:]
        return loss_mask_flat.to(dtype=torch.float32), True
    if "response_mask" in data.keys():
        mask = data["response_mask"].to(dtype=torch.float32)
        return (mask[..., 1:] if mask.shape[-1] > 0 else mask), False
    mask = loss_mask.to(dtype=torch.float32)
    if hasattr(mask, "is_nested") and mask.is_nested:
        return mask.values()[1:].to(dtype=torch.float32), True
    return (mask[..., 1:] if mask.ndim >= 1 and mask.shape[-1] > 0 else mask), False


def _align_mask_to_seq(loss_mask: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Ensure ``loss_mask`` trailing length equals next-token ``seq_len`` (``S-1``)."""
    mask = loss_mask.to(dtype=torch.float32)
    if mask.ndim == 0:
        raise ValueError("loss_mask must be at least 1D")
    if mask.shape[-1] == seq_len + 1:
        return mask[..., 1:]
    if mask.shape[-1] == seq_len:
        return mask
    raise ValueError(
        f"loss_mask length {mask.shape[-1]} incompatible with next-token seq_len {seq_len} "
        f"(expected {seq_len} or {seq_len + 1})"
    )


def extract_topk_log_probs(
    logits: torch.Tensor,
    topk: int,
    *,
    indices: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(topk_log_probs, topk_indices)`` with shape ``[..., K]``.

    Prefer :func:`extract_topk_log_probs_from_hidden` for training — full-vocab
    ``logits`` keep a ``[..., V]`` tensor on the autograd graph and allocate a
    matching ``grad_logits`` on backward (common OOM source vs SFT).
    """
    if indices is None:
        topk = min(int(topk), logits.shape[-1])
        topk_logits, indices = torch.topk(logits, k=topk, dim=-1)
    else:
        topk_logits = torch.gather(logits, dim=-1, index=indices)
    log_z = torch.logsumexp(logits, dim=-1, keepdim=True)
    topk_log_probs = topk_logits - log_z
    return topk_log_probs, indices


def _gather_linear_logits(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    indices: torch.Tensor,
    *,
    token_chunk_size: int = 2048,
) -> torch.Tensor:
    """``topk_logits[..., k] = hidden[..., :] @ weight[indices[..., k], :].T`` without full V."""
    *lead, hidden_size = hidden.shape
    indices = indices.long()
    hidden_f = hidden.reshape(-1, hidden_size)
    indices_f = indices.reshape(-1, indices.shape[-1])
    n_tokens, topk = indices_f.shape
    chunks: list[torch.Tensor] = []
    for start in range(0, n_tokens, token_chunk_size):
        end = min(start + token_chunk_size, n_tokens)
        h = hidden_f[start:end]
        idx = indices_f[start:end]
        # [C, K, H] — token-chunked to bound peak memory.
        selected = weight[idx]
        chunks.append((h.unsqueeze(1) * selected).sum(dim=-1))
    out = torch.cat(chunks, dim=0)
    return out.reshape(*lead, topk)


def _chunked_logsumexp_linear(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    vocab_chunk_size: int = 4096,
    token_chunk_size: int = 2048,
) -> torch.Tensor:
    """``logsumexp(hidden @ weight.T, dim=-1)`` with vocab/token chunking (no full logits)."""
    *lead, hidden_size = hidden.shape
    hidden_f = hidden.reshape(-1, hidden_size)
    n_tokens = hidden_f.shape[0]
    vocab = int(weight.shape[0])
    out = hidden_f.new_empty(n_tokens)
    for t0 in range(0, n_tokens, token_chunk_size):
        t1 = min(t0 + token_chunk_size, n_tokens)
        h = hidden_f[t0:t1]
        # Online logsumexp over vocab chunks.
        m = None
        s = None
        for v0 in range(0, vocab, vocab_chunk_size):
            v1 = min(v0 + vocab_chunk_size, vocab)
            logits_c = h @ weight[v0:v1].to(dtype=h.dtype).T
            logits_c = logits_c.float()
            c_max = logits_c.max(dim=-1).values
            if m is None:
                m = c_max
                s = torch.exp(logits_c - c_max.unsqueeze(-1)).sum(dim=-1)
            else:
                new_m = torch.maximum(m, c_max)
                s = s * torch.exp(m - new_m) + torch.exp(logits_c - new_m.unsqueeze(-1)).sum(dim=-1)
                m = new_m
        assert m is not None and s is not None
        out[t0:t1] = m + torch.log(s.clamp_min(1e-20))
    return out.reshape(*lead)


def extract_topk_log_probs_from_hidden(
    hidden: torch.Tensor,
    lm_head: torch.nn.Module,
    topk: int,
    *,
    indices: Optional[torch.Tensor] = None,
    vocab_chunk_size: int = 4096,
    token_chunk_size: int = 2048,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Memory-efficient top-k log-probs for distillation training.

    Args:
        hidden: Last hidden states aligned to next-token positions ``[..., H]``.
        lm_head: ``nn.Linear`` (bias-free) projecting to vocab.
        topk: Used only when ``indices is None`` (teacher path).
        indices: Optional precomputed top-k token ids (student gathers teacher idx).

    Notes:
        ``log_z`` is stop-grad so the graph never retains full-vocab logits; backward
        only flows through gathered top-k rows of ``lm_head.weight`` (no ``grad_logits[..., V]``).
    """
    weight = lm_head.weight
    bias = getattr(lm_head, "bias", None)
    if bias is not None:
        raise NotImplementedError("extract_topk_log_probs_from_hidden requires bias-free lm_head")

    if indices is None:
        # Teacher / first pass: need scores for all vocab — still chunked, no autograd.
        *lead, hidden_size = hidden.shape
        hidden_f = hidden.reshape(-1, hidden_size)
        n_tokens = hidden_f.shape[0]
        topk = min(int(topk), int(weight.shape[0]))
        best_scores = hidden_f.new_full((n_tokens, topk), float("-inf"))
        best_index = torch.zeros((n_tokens, topk), device=hidden.device, dtype=torch.long)
        vocab = int(weight.shape[0])
        for t0 in range(0, n_tokens, token_chunk_size):
            t1 = min(t0 + token_chunk_size, n_tokens)
            h = hidden_f[t0:t1]
            local_best_s = best_scores[t0:t1]
            local_best_i = best_index[t0:t1]
            for v0 in range(0, vocab, vocab_chunk_size):
                v1 = min(v0 + vocab_chunk_size, vocab)
                logits_c = h @ weight[v0:v1].to(dtype=h.dtype).T
                c_topk = min(topk, v1 - v0)
                scores_c, idx_c = torch.topk(logits_c, k=c_topk, dim=-1)
                idx_c = idx_c + v0
                merged_s = torch.cat([local_best_s, scores_c], dim=-1)
                merged_i = torch.cat([local_best_i, idx_c], dim=-1)
                local_best_s, sel = torch.topk(merged_s, k=topk, dim=-1)
                local_best_i = torch.gather(merged_i, dim=-1, index=sel)
            best_scores[t0:t1] = local_best_s
            best_index[t0:t1] = local_best_i
        indices = best_index.reshape(*lead, topk)
        log_z = _chunked_logsumexp_linear(
            hidden, weight, vocab_chunk_size=vocab_chunk_size, token_chunk_size=token_chunk_size
        )
        topk_logits = _gather_linear_logits(
            hidden, weight, indices, token_chunk_size=token_chunk_size
        )
        topk_log_probs = topk_logits - log_z.to(dtype=topk_logits.dtype).unsqueeze(-1)
        return topk_log_probs, indices

    # Student path: stop-grad on partition; grads only through gathered top-k logits.
    with torch.no_grad():
        log_z = _chunked_logsumexp_linear(
            hidden.detach(),
            weight.detach(),
            vocab_chunk_size=vocab_chunk_size,
            token_chunk_size=token_chunk_size,
        )
    topk_logits = _gather_linear_logits(
        hidden, weight, indices.long(), token_chunk_size=token_chunk_size
    )
    topk_log_probs = topk_logits - log_z.to(dtype=topk_logits.dtype).unsqueeze(-1)
    return topk_log_probs, indices


def _empty_ohisa_metrics() -> dict[str, float]:
    return {
        "ohisa/on_mse": 0.0,
        "ohisa/num_tokens": 0.0,
        "ohisa/num_layers": 0.0,
    }


def project_token_mask_to_valid_span(
    token_mask: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Left-align ``token_mask`` onto the valid tokens of each row.

    ``token_mask`` and ``attention_mask`` are ``[B, S]`` on the collated batch.
    The result is ``[B, max_valid]``, matching rmpad's padded expansion: valid
    tokens are packed to the left and the row is right-padded.
    """
    if token_mask.shape != attention_mask.shape:
        raise ValueError(
            f"token_mask shape {tuple(token_mask.shape)} != attention_mask {tuple(attention_mask.shape)}"
        )
    valid = attention_mask.to(dtype=torch.bool)
    lengths = valid.sum(dim=1)
    max_len = int(lengths.max().item()) if lengths.numel() else 0
    batch = int(token_mask.shape[0])
    out = torch.zeros(batch, max_len, dtype=torch.bool, device=token_mask.device)
    if max_len == 0:
        return out
    flags = token_mask.to(dtype=torch.bool) & valid
    for row in range(batch):
        idx = valid[row].nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            continue
        out[row, : idx.numel()] = flags[row, idx]
    return out


def build_response_token_mask(
    seq_len: int,
    response_mask: torch.Tensor,
    response_start_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Boolean mask ``[B, S]`` that is true only on valid response-token positions.

    ``response_mask`` is ``[B, R]``. ``response_start_idx`` is the index of the first
    response token in the full sequence. When it is omitted, the response occupies
    the last ``R`` positions.
    """
    if response_mask.ndim != 2:
        raise ValueError(f"response_mask must be [B, R], got {tuple(response_mask.shape)}")
    batch_size, resp_len = int(response_mask.shape[0]), int(response_mask.shape[1])
    if seq_len < 1:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=response_mask.device)
    valid = response_mask.to(dtype=torch.bool)
    if response_start_idx is None:
        if resp_len > seq_len:
            raise ValueError(f"response length {resp_len} exceeds seq_len {seq_len}")
        mask[:, seq_len - resp_len : seq_len] = valid
        return mask
    starts = response_start_idx.to(device=response_mask.device, dtype=torch.long).view(batch_size)
    if starts.shape != (batch_size,):
        raise ValueError(
            f"response_start_idx shape must be ({batch_size},), got {tuple(response_start_idx.shape)}"
        )
    cols = torch.arange(resp_len, device=response_mask.device, dtype=torch.long).view(1, resp_len)
    pos = starts.view(batch_size, 1) + cols
    in_range = valid & (pos >= 0) & (pos < seq_len)
    mask.scatter_(1, pos.clamp(0, seq_len - 1), in_range)
    return mask


def compute_hidden_trajectory_loss(
    actor_traj: dict[str, Any],
    teacher_module: torch.nn.Module,
    *,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.Tensor] = None,
    response_token_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """On-policy hidden MSE: ``MSE(h_a^{l+1}, F_l^T(h_a^l))`` on the keep mask.

    Each SAMoD layer's keep mask covers every valid sequence token. When
    ``response_token_mask`` is set, the loss is the intersection with response
    positions. The teacher transition is stop-grad. Gradients flow through the
    actor hidden ``h_a^{l+1}`` recorded on the student trajectory.
    """
    layer_indices: list[int] = list(actor_traj["layer_indices"])
    h_in_list: list[torch.Tensor] = actor_traj["hidden_states_in"]
    h_out_list: list[torch.Tensor] = actor_traj["hidden_states_out"]
    keep_masks: list[torch.Tensor] = actor_traj["keep_masks"]
    position_embeddings = actor_traj.get("position_embeddings")
    causal_mask = actor_traj.get("causal_mask")
    linear_attn_mask = actor_traj.get("linear_attn_mask")
    layer_extra: dict = {}
    if actor_traj.get("indexes") is not None:
        layer_extra["indexes"] = actor_traj["indexes"]
    if actor_traj.get("padlen"):
        layer_extra["padlen"] = actor_traj["padlen"]

    if not layer_indices:
        if h_in_list:
            zero = h_in_list[0].new_zeros(())
            empty = _empty_ohisa_metrics()
            empty["_per_sample_ohisa_loss"] = [0.0] * int(h_in_list[0].shape[0])
            return zero, empty
        return torch.zeros(()), _empty_ohisa_metrics()

    text_model = get_text_model(teacher_module)
    layer_types = getattr(getattr(text_model, "config", None), "layer_types", None)

    total_loss = None
    total_tokens = 0.0
    total_layers = 0.0
    per_sample_sum = None
    per_sample_count = None
    on_sum = 0.0
    on_layers = 0.0

    for layer_idx, h_a_l, h_a_next, keep_mask in zip(
        layer_indices, h_in_list, h_out_list, keep_masks
    ):
        if layer_types is not None and layer_types[layer_idx] == "linear_attention":
            layer_attn_mask = linear_attn_mask
        else:
            layer_attn_mask = causal_mask if causal_mask is not None else attention_mask
        layer_position_ids = position_ids if position_ids is not None else actor_traj.get("text_position_ids")
        # Run the teacher transition on every rank. Skipping an empty keep mask
        # desynchronizes FSDP collectives.
        with torch.no_grad():
            h_t_next = teacher_forward_layer(
                teacher_module,
                layer_idx,
                h_a_l.detach(),
                attention_mask=layer_attn_mask,
                position_ids=layer_position_ids,
                position_embeddings=position_embeddings,
                **layer_extra,
            ).detach()

        keep = keep_mask.to(dtype=torch.bool)
        if response_token_mask is not None:
            resp = response_token_mask.to(device=keep.device, dtype=torch.bool)
            if resp.shape != keep.shape:
                raise ValueError(
                    f"response_token_mask shape {tuple(resp.shape)} != keep mask {tuple(keep.shape)}"
                )
            keep = keep & resp
        if not bool(keep.any()):
            continue

        hidden_size = float(h_a_next.shape[-1])
        per_token = ((h_a_next - h_t_next).to(dtype=torch.float32) ** 2).sum(dim=-1) / hidden_size
        if per_sample_sum is None:
            per_sample_sum = torch.zeros(per_token.shape[0], device=per_token.device, dtype=torch.float32)
            per_sample_count = torch.zeros_like(per_sample_sum)
        per_sample_sum = per_sample_sum + (per_token.detach().float() * keep).sum(dim=-1)
        per_sample_count = per_sample_count + keep.sum(dim=-1).to(dtype=torch.float32)
        kept = per_token[keep]
        if kept.numel() == 0:
            continue
        layer_loss = kept.mean()
        total_loss = layer_loss if total_loss is None else (total_loss + layer_loss)
        total_layers += 1.0
        total_tokens += float(kept.numel())
        on_sum += float(layer_loss.detach().item())
        on_layers += 1.0

    if total_loss is None:
        device = h_in_list[0].device if h_in_list else torch.device("cpu")
        empty = _empty_ohisa_metrics()
        if h_in_list:
            empty["_per_sample_ohisa_loss"] = [0.0] * int(h_in_list[0].shape[0])
        return torch.zeros((), device=device), empty

    total_loss = total_loss / max(total_layers, 1.0)
    per_sample = per_sample_sum / per_sample_count.clamp_min(1.0)
    per_sample = torch.where(per_sample_count > 0, per_sample, torch.zeros_like(per_sample))
    return total_loss, {
        "ohisa/on_mse": on_sum / max(on_layers, 1.0),
        "_per_sample_ohisa_loss": per_sample.detach().cpu().tolist(),
        "ohisa/num_tokens": total_tokens,
        "ohisa/num_layers": total_layers,
    }


def compute_js_task_loss(
    student_topk_log_probs: torch.Tensor,
    teacher_topk_log_probs: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    cfg: SelfDistillationCfg,
    batch_num_tokens: Optional[float] = None,
    dp_size: int = 1,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """JS (or FKL/RKL) on assistant tokens using OPD's ``compute_self_distillation_loss``."""
    # Shapes: [B, T, K] or [T, K], with T = S-1 from logits[:, :-1].
    if student_topk_log_probs.ndim == 2:
        seq_len = int(student_topk_log_probs.shape[0])
        mask = _align_mask_to_seq(loss_mask.reshape(-1), seq_len)
        response_mask = mask.reshape(1, -1)
        student_lp = student_topk_log_probs.unsqueeze(0)
        teacher_lp = teacher_topk_log_probs.unsqueeze(0)
    else:
        seq_len = int(student_topk_log_probs.shape[1])
        if loss_mask.ndim == 1:
            mask = _align_mask_to_seq(loss_mask, seq_len)
            response_mask = mask.reshape(student_topk_log_probs.shape[0], -1)
            if response_mask.shape[0] == 1 and student_topk_log_probs.shape[0] > 1:
                response_mask = response_mask.expand(student_topk_log_probs.shape[0], -1)
        else:
            response_mask = _align_mask_to_seq(loss_mask, seq_len)
        student_lp = student_topk_log_probs
        teacher_lp = teacher_topk_log_probs

    if tuple(response_mask.shape[:2]) != tuple(student_lp.shape[:2]):
        raise ValueError(
            f"JS mask shape {tuple(response_mask.shape)} != student topk {tuple(student_lp.shape[:2])} "
            "(next-token logits vs loss_mask alignment failed)"
        )

    # Dummy per-token logprob (unused when full_logit_distillation=True).
    dummy = student_lp[..., 0]
    loss, metrics = compute_self_distillation_loss(
        student_log_probs=dummy,
        teacher_log_probs=dummy,
        response_mask=response_mask,
        self_distillation_config=cfg,
        student_topk_log_probs=student_lp,
        teacher_topk_log_probs=teacher_lp,
        loss_agg_mode="token-mean",
        batch_num_tokens=batch_num_tokens,
    )
    metrics.pop("_per_sample_vopd_loss", None)
    # Match SFT scaling convention: mean token loss * dp_size when using global batch_num_tokens.
    if batch_num_tokens is not None:
        # compute_self_distillation_loss already token-means; keep as-is for DP AVG later.
        pass
    return loss * float(dp_size) / max(dp_size, 1), metrics


def compute_ce_task_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    response_mask: torch.Tensor,
    *,
    batch_num_tokens,
    dp_size: int = 1,
) -> torch.Tensor:
    """Standard next-token CE on labeled (assistant) positions."""
    # logits/labels/mask: [B, S-1, V] / [B, S-1] / [B, S-1]
    log_probs = F.log_softmax(logits, dim=-1)
    nll = -log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    mask = response_mask.to(dtype=nll.dtype)
    denom = batch_num_tokens if batch_num_tokens is not None else mask.sum().clamp(min=1.0)
    return masked_sum(nll, mask) / denom * float(dp_size)


def _attach_samod_and_hidden(
    *,
    task_loss: torch.Tensor,
    actor_output: dict[str, Any],
    teacher_module: torch.nn.Module,
    ohisa_loss_weight: float,
    metrics: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    traj = actor_output.get("samod_trajectory")
    if traj is not None:
        weight = float(ohisa_loss_weight)
        # When λ==0 still compute for logging, but keep it off the optim graph.
        ctx = torch.enable_grad() if weight != 0.0 else torch.no_grad()
        with ctx:
            hidden_loss, hidden_metrics = compute_hidden_trajectory_loss(
                traj,
                teacher_module,
            )
        hidden_metrics.pop("_per_sample_ohisa_loss", None)
        metrics.update(hidden_metrics)
        # Log unweighted L_hidden (before λ); optim uses λ * L_hidden when λ!=0.
        metrics["ohisa/loss"] = float(hidden_loss.detach().item())
        loss = task_loss + weight * hidden_loss if weight != 0.0 else task_loss
    else:
        loss = task_loss
        metrics["ohisa/loss"] = 0.0
        metrics.update(_empty_ohisa_metrics())

    samod_dreg_loss = actor_output.get("samod_dreg_loss")
    if samod_dreg_loss is not None:
        weight = float(actor_output.get("samod_dreg_loss_weight", 1.0))
        loss = loss + samod_dreg_loss * weight
        metrics["samod_dreg_loss"] = float(samod_dreg_loss.detach().item())
    else:
        metrics["samod_dreg_loss"] = 0.0

    extra_metrics = actor_output.get("extra_metrics")
    if extra_metrics:
        for key, value in extra_metrics.items():
            if isinstance(key, str) and (
                key.startswith("samod_vision_token_retention_layer_")
                or key.endswith("_numerator")
                or key.endswith("_denominator")
                or key == "samod_vision_token_retention"
            ):
                continue
            if isinstance(value, torch.Tensor):
                metrics[key] = value.detach().item() if value.numel() == 1 else value.detach().cpu().tolist()
            else:
                metrics[key] = value
        if "samod_vision_token_retention" not in metrics and "samod_vision_token_retention" in extra_metrics:
            v = extra_metrics["samod_vision_token_retention"]
            metrics["samod_vision_token_retention"] = (
                float(v.detach().item()) if isinstance(v, torch.Tensor) else float(v)
            )
    return loss, metrics


def ohisa_sft_loss(
    *,
    actor_output: dict[str, Any],
    teacher_output: dict[str, Any],
    data: TensorDict,
    teacher_module: torch.nn.Module,
    self_distillation_cfg: SelfDistillationCfg,
    ohisa_loss_weight: float,
    task_loss_form: str = "off_distill",
    dp_group=None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Total loss: task(sft|JS) + λ * on-policy L_hidden + samod_dreg_loss.

    ``task_loss_form``:
      - ``sft``: CE on labeled assistant tokens (teacher-independent task loss)
      - ``off_distill`` / ``on_distill``: JS on the tokens provided by the trainer
        (labeled for off-policy; actor-generated for on-policy)

    The hidden term is ``MSE(h_a^{l+1}, F_l^T(h_a^l))`` on the actor keep mask.
    """
    form = str(task_loss_form)
    if form not in TASK_LOSS_FORMS:
        raise ValueError(f"Unsupported task_loss_form={task_loss_form!r}; expected one of {TASK_LOSS_FORMS}")

    batch_num_tokens = data["batch_num_tokens"]
    dp_size = int(data["dp_size"])
    metrics: dict[str, Any] = {"task_loss_form": form}

    if form == "sft":
        response_mask = data["response_mask"].to(dtype=torch.float32)
        logits = actor_output["logits"]
        labels = actor_output["labels"]
        task_loss = compute_ce_task_loss(
            logits,
            labels,
            response_mask,
            batch_num_tokens=batch_num_tokens,
            dp_size=dp_size,
        )
    else:
        loss_mask_flat, is_flat = _flatten_loss_mask(data)
        student_topk = actor_output["topk_log_probs"]
        teacher_topk = teacher_output["topk_log_probs"]
        if is_flat:
            task_loss, js_metrics = compute_js_task_loss(
                student_topk,
                teacher_topk,
                loss_mask_flat,
                cfg=self_distillation_cfg,
                batch_num_tokens=batch_num_tokens,
                dp_size=dp_size,
            )
        else:
            response_mask = data["response_mask"].to(dtype=torch.float32)
            task_loss, js_metrics = compute_js_task_loss(
                student_topk,
                teacher_topk,
                response_mask,
                cfg=self_distillation_cfg,
                batch_num_tokens=batch_num_tokens,
                dp_size=dp_size,
            )
        metrics.update(
            {f"self_distillation/{k.split('/')[-1]}" if "/" in k else k: v for k, v in js_metrics.items()}
        )

    loss, metrics = _attach_samod_and_hidden(
        task_loss=task_loss,
        actor_output=actor_output,
        teacher_module=teacher_module,
        ohisa_loss_weight=ohisa_loss_weight,
        metrics=metrics,
    )
    metrics["_ar_loss"] = float(task_loss.detach().item())
    return loss, metrics
