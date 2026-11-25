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

import argparse
import ast
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5CausalLMOutputWithPast

from verl.models.transformers.qwen3_5 import fast_pos_embed_interpolate

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

MODEL_AUTO_CLASS = "AutoModelForImageTextToText"


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
    labels: list[int] = []
    next_id = int(segment_ids.max().item()) + 1
    for t, h, w in grids.tolist():
        n_tokens = int(t) * (int(h * w) // merge_area)
        labels.extend([next_id] * n_tokens)
        next_id += 1
    if len(labels) != int(flat_positions.numel()):
        raise ValueError(
            "Visual placeholder/grid token count mismatch while building SAMoD segments: "
            f"mask={flat_positions.numel()}, grids={len(labels)}"
        )
    segment_ids.reshape(-1)[flat_positions] = torch.tensor(
        labels, device=segment_ids.device, dtype=torch.long
    )


def _default_project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve_repo_path(path_value: str, *, project_root: Path, config_dir: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    project_path = project_root / path
    if project_path.exists():
        return project_path
    return config_dir / path


def _deep_update(target: dict, overrides: dict) -> dict:
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _deep_update(target[key], value)
        else:
            target[key] = value
    return target


_PREPROCESSOR_CONFIG_FILES = ("preprocessor_config.json", "video_preprocessor_config.json")


def _remove_path(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        shutil.rmtree(path)


def _ensure_symlink(src: Path, target: Path) -> None:
    src = src.resolve()
    if target.is_symlink() and target.resolve() == src:
        return
    if target.exists() or target.is_symlink():
        _remove_path(target)
    target.symlink_to(src, target_is_directory=src.is_dir())


def _symlink_base_assets(base: Path, student: Path, modeling_file: Path) -> None:
    student.mkdir(parents=True, exist_ok=True)
    skip_names = {"config.json", modeling_file.name, *_PREPROCESSOR_CONFIG_FILES}
    for item in base.iterdir():
        target = student / item.name
        if item.name in skip_names:
            continue
        _ensure_symlink(item, target)


def _resolve_preprocessor_overrides(
    spec: dict,
    *,
    max_pixels: int | None = None,
    min_pixels: int | None = None,
) -> dict:
    overrides = dict(spec.get("preprocessor_overrides", {}))
    size_overrides = dict(overrides.get("size", {}))
    if max_pixels is not None:
        size_overrides["longest_edge"] = max_pixels
    if min_pixels is not None:
        size_overrides["shortest_edge"] = min_pixels
    if size_overrides:
        overrides["size"] = size_overrides
    return overrides


def _collect_relative_python_modules(root_file: Path) -> list[Path]:
    """Collect sibling .py modules reachable via relative imports from root_file."""
    source_dir = root_file.parent
    collected: set[Path] = {root_file.resolve()}
    pending = [root_file.resolve()]

    while pending:
        current = pending.pop()
        tree = ast.parse(current.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.ImportFrom) or node.level < 1:
                continue
            if not node.module:
                continue
            module_path = source_dir / f"{node.module.split('.')[0]}.py"
            if not module_path.is_file():
                continue
            resolved = module_path.resolve()
            if resolved in collected:
                continue
            collected.add(resolved)
            pending.append(resolved)

    return sorted(path for path in collected if path != root_file.resolve())


def _copy_modeling_companion_modules(modeling_file: Path, student: Path) -> None:
    for companion in _collect_relative_python_modules(modeling_file):
        shutil.copy2(companion, student / companion.name)


def _write_preprocessor_configs(base: Path, student: Path, preprocessor_overrides: dict) -> None:
    for filename in _PREPROCESSOR_CONFIG_FILES:
        src = base / filename
        if not src.is_file():
            continue
        target = student / filename
        if not preprocessor_overrides:
            _ensure_symlink(src, target)
            continue

        with src.open(encoding="utf-8") as f:
            cfg = json.load(f)
        _deep_update(cfg, preprocessor_overrides)
        _remove_path(target)
        with target.open("w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
            f.write("\n")


def build_runtime_model_repo(
    *,
    base_model_path: str,
    runtime_model_path: str,
    max_pixels: int | None = None,
    min_pixels: int | None = None,
) -> Path:
    """Build a runtime HF repo that keeps the original model intact.

    Weights/config are symlinked from ``base_model_path``; only preprocessor
    configs are rewritten so ``max_pixels`` / ``min_pixels`` take effect.
    """
    base = Path(base_model_path).resolve()
    runtime = Path(runtime_model_path).resolve()
    if base == runtime:
        raise ValueError("runtime_model_path must differ from base_model_path")
    if not base.is_dir():
        raise FileNotFoundError(f"base model path does not exist: {base}")

    runtime.mkdir(parents=True, exist_ok=True)
    for item in base.iterdir():
        if item.name in _PREPROCESSOR_CONFIG_FILES:
            continue
        _ensure_symlink(item, runtime / item.name)

    preprocessor_overrides = _resolve_preprocessor_overrides(
        {},
        max_pixels=max_pixels,
        min_pixels=min_pixels,
    )
    _write_preprocessor_configs(base, runtime, preprocessor_overrides)

    print(f"Built runtime model repo: {runtime}")
    print(f"  base model: {base}")
    if preprocessor_overrides:
        size = preprocessor_overrides.get("size", {})
        print(
            "  preprocessor size: "
            f"longest_edge={size.get('longest_edge')}, "
            f"shortest_edge={size.get('shortest_edge')}"
        )
    return runtime


def build_student_model_repo(
    *,
    base_model_path: str,
    student_model_path: str,
    student_config_path: str,
    project_root: str | None = None,
    max_pixels: int | None = None,
    min_pixels: int | None = None,
) -> Path:
    project_root_path = Path(project_root).resolve() if project_root else _default_project_root()
    student_config = Path(student_config_path).resolve()
    config_dir = student_config.parent
    with student_config.open(encoding="utf-8") as f:
        spec = json.load(f)

    base = Path(base_model_path).resolve()
    student = Path(student_model_path).resolve()
    modeling_file = _resolve_repo_path(spec["modeling_file"], project_root=project_root_path, config_dir=config_dir)
    modeling_module = spec.get("modeling_module") or modeling_file.stem
    model_class = spec["model_class"]

    if base == student:
        raise ValueError("student_model_path must differ from base_model_path")
    if not base.is_dir():
        raise FileNotFoundError(f"base model path does not exist: {base}")
    if not student_config.is_file():
        raise FileNotFoundError(f"student config does not exist: {student_config}")
    if not modeling_file.is_file():
        raise FileNotFoundError(f"modeling file does not exist: {modeling_file}")

    _symlink_base_assets(base, student, modeling_file)
    shutil.copy2(modeling_file, student / modeling_file.name)
    _copy_modeling_companion_modules(modeling_file, student)

    with (base / "config.json").open(encoding="utf-8") as f:
        hf_config = json.load(f)

    # Older multimodal checkpoints (notably Qwen2.5-VL) store the text
    # configuration at the top level.  Adding a partial ``text_config`` from
    # the student overrides would make current Transformers treat it as a
    # complete sub-config and fill all omitted fields with model defaults.
    # Normalize first so the original architecture is preserved, then apply
    # the sparse-model overrides below.
    config_overrides = dict(spec.get("config_overrides", {}))
    if "text_config" in config_overrides and "text_config" not in hf_config:
        hf_config = AutoConfig.from_pretrained(base, trust_remote_code=True).to_dict()

    # SAMoD knobs apply to the LLM backbone. They are not nested under
    # text_config / vision_config in the student spec.
    text_overrides = {
        key: config_overrides.pop(key)
        for key in list(config_overrides)
        if key.startswith(("samod_", "ohisa_"))
    }
    if text_overrides:
        text_config = hf_config.setdefault("text_config", {})
        if not isinstance(text_config, dict):
            raise TypeError("text_config must be a dict to attach SAMoD overrides")
        text_config.update(text_overrides)

    hf_config["architectures"] = [model_class]
    auto_map = dict(hf_config.get("auto_map", {}))
    auto_map[MODEL_AUTO_CLASS] = f"{modeling_module}.{model_class}"
    hf_config["auto_map"] = auto_map
    _deep_update(hf_config, config_overrides)

    with (student / "config.json").open("w", encoding="utf-8") as f:
        json.dump(hf_config, f, indent=2, ensure_ascii=False)
        f.write("\n")

    preprocessor_overrides = _resolve_preprocessor_overrides(
        spec,
        max_pixels=max_pixels,
        min_pixels=min_pixels,
    )
    _write_preprocessor_configs(base, student, preprocessor_overrides)

    print(f"Built student model repo: {student}")
    print(f"  spec: {student_config}")
    print(f"  base model: {base}")
    print(f"  modeling: {modeling_file}")
    print(f"  class: {model_class}")
    print(f"  auto class: {MODEL_AUTO_CLASS}")
    if preprocessor_overrides:
        size = preprocessor_overrides.get("size", {})
        print(f"  preprocessor size: longest_edge={size.get('longest_edge')}, shortest_edge={size.get('shortest_edge')}")
    return student


def _get_input_embeds(
    model,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    mm_token_type_ids: Optional[torch.IntTensor] = None,
):
    if inputs_embeds is None:
        inputs_embeds = model.get_input_embeddings()(input_ids)

    visual_pos_masks = None
    visual_segment_ids = torch.zeros(
        inputs_embeds.shape[:2], device=inputs_embeds.device, dtype=torch.long
    )
    has_visual_segments = False
    if mm_token_type_ids is not None:
        visual_pos_masks = mm_token_type_ids.to(device=inputs_embeds.device) != 0

    if pixel_values is not None:
        pixel_values = pixel_values.type(model.visual.dtype)
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw).pooler_output
        n_image_tokens = (input_ids == model.config.image_token_id).sum().item()
        n_image_features = image_embeds.shape[0]
        if n_image_tokens != n_image_features:
            raise ValueError(
                f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
            )

        image_mask = input_ids == model.config.image_token_id
        image_mask_expanded = image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        visual_pos_masks = image_mask if visual_pos_masks is None else (visual_pos_masks | image_mask)
        _append_visual_grid_segments(
            visual_segment_ids,
            image_mask,
            image_grid_thw,
            model.config.vision_config.spatial_merge_size,
        )
        has_visual_segments = image_grid_thw is not None or has_visual_segments
        inputs_embeds = inputs_embeds.masked_scatter(image_mask_expanded, image_embeds)

    if pixel_values_videos is not None:
        pixel_values_videos = pixel_values_videos.type(model.visual.dtype)
        video_embeds = model.visual(pixel_values_videos, grid_thw=video_grid_thw).pooler_output
        n_video_tokens = (input_ids == model.config.video_token_id).sum().item()
        n_video_features = video_embeds.shape[0]
        if n_video_tokens != n_video_features:
            raise ValueError(
                f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
            )

        video_mask = input_ids == model.config.video_token_id
        video_mask_expanded = video_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
        video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
        visual_pos_masks = video_mask if visual_pos_masks is None else (visual_pos_masks | video_mask)
        _append_visual_grid_segments(
            visual_segment_ids,
            video_mask,
            video_grid_thw,
            model.config.vision_config.spatial_merge_size,
        )
        has_visual_segments = video_grid_thw is not None or has_visual_segments
        inputs_embeds = inputs_embeds.masked_scatter(video_mask_expanded, video_embeds)

    if pixel_values is None and pixel_values_videos is None:
        config = model.config.vision_config
        patch_dim = config.in_channels * config.temporal_patch_size * config.patch_size**2
        pixel_values = torch.zeros((16, patch_dim), dtype=inputs_embeds.dtype, device=inputs_embeds.device)
        image_grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device=inputs_embeds.device)
        image_embeds = model.visual(pixel_values, grid_thw=image_grid_thw).pooler_output
        inputs_embeds = inputs_embeds + 0.0 * image_embeds.mean()

    if attention_mask is not None:
        attention_mask = attention_mask.to(inputs_embeds.device)
    if visual_pos_masks is not None:
        visual_pos_masks = visual_pos_masks.to(device=inputs_embeds.device, dtype=torch.bool)

    return {
        "inputs_embeds": inputs_embeds,
        "attention_mask": attention_mask,
        "visual_pos_masks": visual_pos_masks,
        "visual_segment_ids": visual_segment_ids if has_visual_segments else None,
    }


def qwen3_5_samod_base_forward(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[object] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    pixel_values: Optional[torch.FloatTensor] = None,
    pixel_values_videos: Optional[torch.FloatTensor] = None,
    image_grid_thw: Optional[torch.LongTensor] = None,
    video_grid_thw: Optional[torch.LongTensor] = None,
    mm_token_type_ids: Optional[torch.IntTensor] = None,
    **kwargs,
):
    input_kwargs = _get_input_embeds(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        inputs_embeds=inputs_embeds,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        mm_token_type_ids=mm_token_type_ids,
    )
    kwargs.update(input_kwargs)
    return self.language_model(
        input_ids=None,
        position_ids=position_ids,
        past_key_values=past_key_values,
        **kwargs,
    )


@dataclass
class Qwen3_5SparseCausalLMOutputForPPO(Qwen3_5CausalLMOutputWithPast):
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None
    samod_dreg_loss: Optional[torch.FloatTensor] = None
    samod_dreg_loss_per_sample: Optional[torch.FloatTensor] = None
    extra_metrics: Optional[dict[str, torch.Tensor]] = None
    samod_trajectory: Optional[dict] = None
    # Memory-efficient JS distill (monkey-patched forward must return these).
    topk_log_probs: Optional[torch.FloatTensor] = None
    last_hidden_state: Optional[torch.FloatTensor] = None
    # When True, logits are already [B, response_length, V]. Must default to None
    # (HF ModelOutput requires all non-first field defaults to be None).
    logits_are_response_window: Optional[bool] = None


def _hidden_for_response_logits(
    hidden_states: torch.Tensor,
    response_length: int,
    response_start_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Select hidden states that produce next-token logits for the response window.

    Matches ``DataParallelPPOActor._select_response_positions``:
    default ``[:, -response_length-1:-1]``; otherwise gather by ``response_start_idx``.
    """
    if response_start_idx is None:
        return hidden_states[:, -response_length - 1 : -1, :]

    if response_start_idx.dim() != 1:
        raise ValueError(f"response_start_idx must be rank-1, got shape {tuple(response_start_idx.shape)}")
    if (response_start_idx < 1).any():
        raise ValueError("response_start_idx must be >= 1 so response logits have a preceding context token.")

    batch_size, seqlen = int(hidden_states.shape[0]), int(hidden_states.shape[1])
    if response_start_idx.shape != (batch_size,):
        raise ValueError(
            f"response_start_idx shape must be ({batch_size},), got {tuple(response_start_idx.shape)}"
        )
    offsets = torch.arange(response_length, device=response_start_idx.device, dtype=response_start_idx.dtype)
    response_positions = response_start_idx.unsqueeze(1) - 1 + offsets.unsqueeze(0)
    if response_positions.numel() > 0:
        if int(response_positions.min().item()) < 0 or int(response_positions.max().item()) >= seqlen:
            raise ValueError(
                f"Response positions out of bounds for seqlen={seqlen}: "
                f"min={int(response_positions.min().item())}, max={int(response_positions.max().item())}"
            )
    gather_index = response_positions.to(dtype=torch.long).view(batch_size, response_length, 1).expand(
        batch_size, response_length, hidden_states.shape[-1]
    )
    return torch.gather(hidden_states, dim=1, index=gather_index)


def _logits_from_rmpad_hidden(
    hidden_states: torch.Tensor,
    lm_head: torch.nn.Module,
    *,
    indices: torch.Tensor,
    batch_size: int,
    seqlen: int,
    response_length: int,
    response_start_idx: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Pad flat rmpad hiddens ``[1, nnz, H]`` / ``[nnz, H]`` back to ``[B, S, H]``, then lm_head on response only."""
    from flash_attn.bert_padding import pad_input

    if hidden_states.ndim == 3 and hidden_states.shape[0] == 1:
        hidden_states = hidden_states.squeeze(0)
    if hidden_states.ndim != 2:
        raise ValueError(f"rmpad hidden must be [nnz, H] or [1, nnz, H], got {tuple(hidden_states.shape)}")
    full_hidden = pad_input(hidden_states, indices=indices, batch=batch_size, seqlen=seqlen)
    hidden_for_logits = _hidden_for_response_logits(
        full_hidden,
        response_length=int(response_length),
        response_start_idx=response_start_idx,
    )
    return lm_head(hidden_for_logits)


def forward_with_normal_backend(
    self,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    # Named so transformers.generate kwargs validation accepts HFRollout's mm_token_type_ids
    # (prepare_inputs_for_generation only has **kwargs; validation unions forward params).
    mm_token_type_ids: Optional[torch.IntTensor] = None,
    **kwargs,
) -> Qwen3_5SparseCausalLMOutputForPPO:
    # Pop before backbone: language_model does not accept these kwargs.
    distill_topk_indices = kwargs.pop("distill_topk_indices", None)
    skip_logits = bool(kwargs.pop("skip_logits", False)) or distill_topk_indices is not None
    logits_response_length = kwargs.pop("logits_response_length", None)
    logits_response_start_idx = kwargs.pop("logits_response_start_idx", None)
    logits_rmpad_indices = kwargs.pop("logits_rmpad_indices", None)
    logits_batch_size = kwargs.pop("logits_batch_size", None)
    logits_seqlen = kwargs.pop("logits_seqlen", None)
    if mm_token_type_ids is not None:
        kwargs["mm_token_type_ids"] = mm_token_type_ids

    outputs = self.model(input_ids, **kwargs)
    hidden_states = outputs[0]

    topk_log_probs = None
    logits = None
    logits_are_response_window = None
    if distill_topk_indices is not None:
        # Avoid full-vocab logits / grad_logits (OOM vs SFT on long sequences).
        from verl.workers.utils.ohisa import extract_topk_log_probs_from_hidden

        topk_log_probs, _ = extract_topk_log_probs_from_hidden(
            hidden_states[:, :-1, :],
            self.lm_head,
            topk=int(distill_topk_indices.shape[-1]),
            indices=distill_topk_indices,
        )
    elif not skip_logits:
        if logits_rmpad_indices is not None and logits_response_length is not None:
            # Rmpad [1, nnz, H] → pad → response window → lm_head (avoids [nnz, V] OOM).
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
            # Only project response-window hiddens through lm_head (avoids [B,S,V] OOM).
            hidden_for_logits = _hidden_for_response_logits(
                hidden_states,
                response_length=int(logits_response_length),
                response_start_idx=logits_response_start_idx,
            )
            logits = self.lm_head(hidden_for_logits)
            logits_are_response_window = True
        else:
            logits = self.lm_head(hidden_states)

    return Qwen3_5SparseCausalLMOutputForPPO(
        logits=logits,
        hidden_states=outputs.hidden_states,
        samod_dreg_loss=getattr(outputs, "samod_dreg_loss", None),
        samod_dreg_loss_per_sample=getattr(outputs, "samod_dreg_loss_per_sample", None),
        extra_metrics=getattr(outputs, "extra_metrics", None),
        samod_trajectory=getattr(outputs, "samod_trajectory", None),
        topk_log_probs=topk_log_probs,
        last_hidden_state=hidden_states,
        logits_are_response_window=logits_are_response_window,
    )


def forward_with_torch_backend(
    self,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    mm_token_type_ids: Optional[torch.IntTensor] = None,
    **kwargs,
) -> Qwen3_5SparseCausalLMOutputForPPO:
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    kwargs.pop("distill_topk_indices", None)
    kwargs.pop("skip_logits", None)
    kwargs.pop("logits_response_length", None)
    kwargs.pop("logits_response_start_idx", None)
    kwargs.pop("logits_rmpad_indices", None)
    kwargs.pop("logits_batch_size", None)
    kwargs.pop("logits_seqlen", None)
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
    return Qwen3_5SparseCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
        samod_dreg_loss=getattr(outputs, "samod_dreg_loss", None),
        samod_dreg_loss_per_sample=getattr(outputs, "samod_dreg_loss_per_sample", None),
        extra_metrics=getattr(outputs, "extra_metrics", None),
        samod_trajectory=getattr(outputs, "samod_trajectory", None),
    )


def forward_with_triton_backend(
    self,
    input_ids: torch.LongTensor = None,
    labels: Optional[torch.LongTensor] = None,
    temperature: float = 1.0,
    mm_token_type_ids: Optional[torch.IntTensor] = None,
    **kwargs,
) -> Qwen3_5SparseCausalLMOutputForPPO:
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    kwargs.pop("distill_topk_indices", None)
    kwargs.pop("skip_logits", None)
    kwargs.pop("logits_response_length", None)
    kwargs.pop("logits_response_start_idx", None)
    kwargs.pop("logits_rmpad_indices", None)
    kwargs.pop("logits_batch_size", None)
    kwargs.pop("logits_seqlen", None)
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
    return Qwen3_5SparseCausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        hidden_states=outputs.hidden_states,
        samod_dreg_loss=getattr(outputs, "samod_dreg_loss", None),
        samod_dreg_loss_per_sample=getattr(outputs, "samod_dreg_loss_per_sample", None),
        extra_metrics=getattr(outputs, "extra_metrics", None),
        samod_trajectory=getattr(outputs, "samod_trajectory", None),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen3.5 sparse adapter utilities.")
    parser.add_argument("--build-student-repo", action="store_true")
    parser.add_argument("--build-runtime-repo", action="store_true")
    parser.add_argument("--student-config", default=None)
    parser.add_argument("--base-model-path", default=None)
    parser.add_argument("--student-model-path", default=None)
    parser.add_argument("--runtime-model-path", default=None)
    parser.add_argument("--project-root", default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--min-pixels", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.build_runtime_repo:
        missing = [
            name
            for name in ("base_model_path", "runtime_model_path")
            if getattr(args, name) is None
        ]
        if missing:
            raise SystemExit(f"Missing required arguments for --build-runtime-repo: {', '.join(missing)}")
        build_runtime_model_repo(
            base_model_path=args.base_model_path,
            runtime_model_path=args.runtime_model_path,
            max_pixels=args.max_pixels,
            min_pixels=args.min_pixels,
        )
        return
    if not args.build_student_repo:
        raise SystemExit(
            "Nothing to do. Pass --build-student-repo or --build-runtime-repo."
        )
    missing = [
        name
        for name in ("student_config", "base_model_path", "student_model_path")
        if getattr(args, name) is None
    ]
    if missing:
        raise SystemExit(f"Missing required arguments for --build-student-repo: {', '.join(missing)}")
    build_student_model_repo(
        base_model_path=args.base_model_path,
        student_model_path=args.student_model_path,
        student_config_path=args.student_config,
        project_root=args.project_root,
        max_pixels=args.max_pixels,
        min_pixels=args.min_pixels,
    )


if __name__ == "__main__":
    main()
