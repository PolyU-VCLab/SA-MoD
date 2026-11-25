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

"""Qwen3.5 SAMoD runtime helpers for train / eval loading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import torch
from transformers import AutoConfig, AutoModelForImageTextToText

from .modeling_qwen3_5_samod import Qwen3_5SAMoDForConditionalGeneration, configure_samod

ConfigLike = Any
ModelLike = torch.nn.Module
PathLike = Union[str, Path]

VISION_ATTN_IMPLEMENTATION = "flash_attention_2"


def _get_text_config(config: ConfigLike) -> ConfigLike:
    return getattr(config, "text_config", config)


def _config_json_path(config_path: Optional[PathLike]) -> Optional[Path]:
    if config_path is None:
        return None
    path = Path(config_path)
    if path.is_file():
        return path if path.name == "config.json" else None
    candidate = path / "config.json"
    return candidate if candidate.is_file() else None


def _read_attn_from_json(config_path: Optional[PathLike]) -> tuple[Optional[str], Optional[str]]:
    json_path = _config_json_path(config_path)
    if json_path is None:
        return None, None
    with json_path.open(encoding="utf-8") as handle:
        raw = json.load(handle)
    vision = raw.get("vision_config", {}).get("_attn_implementation")
    text = raw.get("text_config", {}).get("_attn_implementation")
    return vision, text


def _infer_config_path(config: ConfigLike) -> Optional[str]:
    for attr in ("_name_or_path", "name_or_path"):
        value = getattr(config, attr, None)
        if value:
            return str(value)
    return None


def configure_hf_config_attn(
    config: ConfigLike,
    *,
    config_path: Optional[PathLike] = None,
    text_attn: Optional[str] = None,
    vision_attn: Optional[str] = None,
) -> ConfigLike:
    """Apply vision and text ``_attn_implementation`` from ``config.json``."""
    vision_cfg = getattr(config, "vision_config", None)
    text_cfg = _get_text_config(config)
    vision = getattr(vision_cfg, "_attn_implementation", None) if vision_cfg is not None else None
    text = getattr(text_cfg, "_attn_implementation", None)

    json_path = config_path or _infer_config_path(config)
    json_vision, json_text = _read_attn_from_json(json_path)
    if json_vision is not None:
        vision = json_vision
    if json_text is not None:
        text = json_text

    vision_impl = vision_attn if vision_attn is not None else (vision or VISION_ATTN_IMPLEMENTATION)
    text_impl = text_attn if text_attn is not None else (text or VISION_ATTN_IMPLEMENTATION)

    config._attn_implementation = vision_impl
    if vision_cfg is not None:
        vision_cfg._attn_implementation = vision_impl
    text_cfg._attn_implementation = text_impl
    return config


# eval_cfg / text_config field name -> configure_samod argument.
_CONFIG_FIELD_TO_CONFIGURE = {
    "samod_margin": "margin",
    "samod_dreg_loss_weight": "dreg_loss_weight",
    "samod_sparsity_target": "sparsity_target",
    "samod_weight_norm": "weight_norm",
    "ohisa_loss_weight": "ohisa_loss_weight",
    "margin": "margin",
    "dreg_loss_weight": "dreg_loss_weight",
    "sparsity_target": "sparsity_target",
    "weight_norm": "weight_norm",
}


def _filter_samod_configure_kwargs(samod_kwargs: Optional[dict[str, Any]]) -> dict[str, Any]:
    if not samod_kwargs:
        return {}
    filtered = {}
    for key, value in samod_kwargs.items():
        if value is None:
            continue
        configure_key = _CONFIG_FIELD_TO_CONFIGURE.get(key)
        if configure_key is not None:
            filtered[configure_key] = value
    return filtered


def load_samod_model(
    model_path: PathLike,
    *,
    torch_dtype: Optional[torch.dtype] = torch.bfloat16,
    device_map: Optional[str] = "auto",
    trust_remote_code: bool = True,
    configure_from_args: bool = False,
    samod_kwargs: Optional[dict[str, Any]] = None,
    attn_implementation: Optional[str] = VISION_ATTN_IMPLEMENTATION,
    **from_pretrained_kwargs: Any,
) -> ModelLike:
    """Load a Qwen3.5 SAMoD checkpoint.

    Args:
        configure_from_args: If True, apply ``samod_kwargs`` via ``configure_samod``
            (from-scratch / override). If False, reuse hyperparameters already in
            the checkpoint ``config.json``.
    """
    model_path = str(model_path)
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    )
    if configure_from_args:
        raw_kwargs = dict(samod_kwargs or {})
        log_retention = raw_kwargs.pop("samod_log_vision_token_retention", True)
        unknown = sorted(key for key in raw_kwargs if key not in _CONFIG_FIELD_TO_CONFIGURE)
        if unknown:
            known = ", ".join(sorted({*_CONFIG_FIELD_TO_CONFIGURE, "samod_log_vision_token_retention"}))
            raise ValueError(
                "Unknown SAMoD eval config fields: "
                + ", ".join(unknown)
                + f". Expected text_config names: {known}."
            )
        configure_samod(config, **_filter_samod_configure_kwargs(raw_kwargs))
        # Ensure HF auto_map / architectures point at SAMoD when building from a dense base.
        config.architectures = ["Qwen3_5SAMoDForConditionalGeneration"]
        text_cfg = getattr(config, "text_config", None)
        if text_cfg is not None:
            setattr(text_cfg, "samod_log_vision_token_retention", bool(log_retention))

    load_kwargs = {
        "config": config,
        "torch_dtype": torch_dtype,
        "trust_remote_code": trust_remote_code,
        **from_pretrained_kwargs,
    }
    if attn_implementation is not None:
        load_kwargs["attn_implementation"] = attn_implementation
    if device_map is not None:
        load_kwargs["device_map"] = device_map

    try:
        model = Qwen3_5SAMoDForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
    except (ValueError, OSError):
        model = AutoModelForImageTextToText.from_pretrained(model_path, **load_kwargs)
    return model
