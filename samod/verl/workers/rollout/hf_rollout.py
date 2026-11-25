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
"""
Rollout with HuggingFace models (sync, in-process).

Structure:
  1) Generic HF generate (micro-batches, pad, masks, optional rollout_log_probs)
  2) VLM path: processor/tokenizer + raw_prompt tokenization + multimodal kwargs

TODO: refactor this class. Currently, it will hang when using FSDP HybridShard. We should actually create a single
GPU model. Then, get full state_dict and bind the state_dict to the single GPU model. Then, use the single GPU model
to perform generation.
"""

import contextlib

import numpy as np
import torch
import torch.distributed
from tensordict import TensorDict
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

from verl import DataProto
from verl.utils.device import get_device_name, get_torch_device


def _seed_from_rollout_seeds(seeds) -> int:
    """Mix per-row rollout seeds into one integer.

    This model's generate() rejects a ``generator`` argument, so sampling uses
    the process RNG. Copies of the same prompt carry different seeds and
    therefore draw different tokens on different ranks.
    """
    acc = 0
    for seed in np.asarray(seeds).reshape(-1):
        acc = (acc * 1_000_003 + int(seed)) % (2**31 - 1)
    return acc if acc != 0 else 1
from verl.utils.model import compute_position_id_with_mask
from verl.utils.torch_functional import get_response_mask, logprobs_from_logits

from .base import BaseRollout

__all__ = ["HFRollout"]


class _FiniteLogitsProcessor(LogitsProcessor):
    """Sanitize logits before sampling so multinomial never sees NaN/Inf rows."""

    def __init__(self, fallback_token_id: int | None = None):
        self.fallback_token_id = fallback_token_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        finite = torch.isfinite(scores)
        if bool(finite.all()):
            return scores

        valid_rows = finite.any(dim=-1)
        min_value = torch.finfo(scores.dtype).min
        scores = torch.where(finite, scores, torch.full_like(scores, min_value))

        fallback_token_id = self.fallback_token_id
        if fallback_token_id is not None and 0 <= fallback_token_id < scores.shape[-1]:
            invalid_rows = ~valid_rows
            if bool(invalid_rows.any()):
                scores[invalid_rows] = min_value
                scores[invalid_rows, fallback_token_id] = 0

        return scores


class _SuppressTokenIdsProcessor(LogitsProcessor):
    """Ban tokens that are valid in prompts but invalid in generated responses."""

    def __init__(self, token_ids: set[int]):
        self.token_ids = sorted(token_ids)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if self.token_ids:
            valid_token_ids = [token_id for token_id in self.token_ids if 0 <= token_id < scores.shape[-1]]
            if valid_token_ids:
                scores[:, valid_token_ids] = torch.finfo(scores.dtype).min
        return scores


class HFRollout(BaseRollout):
    _MULTI_MODAL_GENERATE_KEYS = {
        "pixel_values",
        "grid_hw",
        "pixel_values_videos",
        "image_grid_thw",
        "video_grid_thw",
        "mm_token_type_ids",
        "vision_group_ids",
    }

    def __init__(
        self,
        module: nn.Module,
        config,
        processor=None,
        tokenizer=None,
        apply_chat_template_kwargs=None,
    ):
        self.config = config
        self.module = module
        # processor/tokenizer are required for VLM rollouts whose dataset only emits
        # `raw_prompt` messages (tokenization normally happens inside the AgentLoop).
        self.processor = processor
        self.tokenizer = tokenizer
        self.apply_chat_template_kwargs = dict(apply_chat_template_kwargs or {})
        self._invalid_text_token_ids_cache: set[int] | None = None

    async def resume(self, tags: list[str]):
        return None

    async def update_weights(self, weights, **kwargs):
        return None

    async def release(self):
        return None

    def _supports_mm_token_type_ids(self) -> bool:
        model_type = getattr(getattr(self.module, "config", None), "model_type", None)
        return model_type in {"qwen3_5", "qwen3_5_moe"}

    def _resolve_eos_pad_token_ids(self, meta_info: dict) -> tuple[object, int]:
        """Resolve eos/pad for generate + post-padding.

        Qwen generation_config often has pad_token_id=None even when the tokenizer
        defines one; HFRollout must not propagate None into tensor math.
        """
        eos_token_id = meta_info.get("eos_token_id")
        pad_token_id = meta_info.get("pad_token_id")

        tokenizer = self.tokenizer
        if tokenizer is None and self.processor is not None:
            tokenizer = getattr(self.processor, "tokenizer", None)

        if eos_token_id is None and tokenizer is not None:
            eos_token_id = tokenizer.eos_token_id
        if pad_token_id is None and tokenizer is not None:
            pad_token_id = tokenizer.pad_token_id

        if pad_token_id is None:
            if isinstance(eos_token_id, (list, tuple)):
                pad_token_id = eos_token_id[0] if eos_token_id else 0
            else:
                pad_token_id = eos_token_id if eos_token_id is not None else 0
        if eos_token_id is None:
            eos_token_id = pad_token_id

        return eos_token_id, int(pad_token_id)

    def _build_multi_modal_kwargs(self, prompts: DataProto, device: torch.device) -> dict[str, torch.Tensor]:
        multi_modal_inputs = prompts.non_tensor_batch.get("multi_modal_inputs")
        if multi_modal_inputs is None:
            return {}

        if isinstance(multi_modal_inputs, dict):
            items = [multi_modal_inputs]
        else:
            items = list(multi_modal_inputs.reshape(-1))

        kwargs = {}
        for key in self._MULTI_MODAL_GENERATE_KEYS:
            values = [
                item[key]
                for item in items
                if isinstance(item, dict) and key in item and item[key] is not None
            ]
            if not values:
                continue

            if all(isinstance(value, torch.Tensor) for value in values):
                tensors = [value.to(device=device) for value in values]
                try:
                    if key == "mm_token_type_ids" and all(tensor.ndim == 1 for tensor in tensors):
                        kwargs[key] = torch.stack(tensors, dim=0)
                    else:
                        kwargs[key] = torch.cat(tensors, dim=0)
                except RuntimeError:
                    kwargs[key] = torch.stack(tensors, dim=0)
            else:
                kwargs[key] = values
        return kwargs

    @staticmethod
    def _template_variables(processing_class) -> set[str]:
        chat_template = getattr(processing_class, "chat_template", None)
        if isinstance(chat_template, dict):
            chat_template = chat_template.get("default")
        if not chat_template:
            return set()
        try:
            from jinja2 import Environment, meta

            return set(meta.find_undeclared_variables(Environment().parse(chat_template)))
        except Exception:
            return set()

    def _filter_chat_template_kwargs(self, processing_class, kwargs: dict) -> dict:
        template_variables = self._template_variables(processing_class)
        if not template_variables:
            return dict(kwargs)
        return {key: value for key, value in kwargs.items() if key in template_variables}

    def _visual_special_token_ids(self) -> set[int]:
        token_ids: set[int] = set()
        token_names = {
            "image_token_id": "<|image_pad|>",
            "video_token_id": "<|video_pad|>",
            "vision_start_token_id": "<|vision_start|>",
            "vision_end_token_id": "<|vision_end|>",
        }
        sources = [
            self.processor,
            getattr(self.processor, "tokenizer", None),
            getattr(self.module, "config", None),
            getattr(getattr(self.module, "module", None), "config", None),
        ]
        for attr_name, token_text in token_names.items():
            for source in sources:
                if source is None:
                    continue
                token_id = getattr(source, attr_name, None)
                if token_id is not None:
                    token_ids.add(int(token_id))
                    break
            else:
                tokenizer = getattr(self.processor, "tokenizer", self.tokenizer)
                if tokenizer is not None and hasattr(tokenizer, "convert_tokens_to_ids"):
                    token_id = tokenizer.convert_tokens_to_ids(token_text)
                    if token_id is not None and token_id != getattr(tokenizer, "unk_token_id", None):
                        token_ids.add(int(token_id))
        return token_ids

    def _invalid_text_token_ids(self) -> set[int]:
        """Ban byte/control tokens that should never appear in normal assistant text."""
        if self._invalid_text_token_ids_cache is not None:
            return self._invalid_text_token_ids_cache

        tokenizer = getattr(self.processor, "tokenizer", None) or self.tokenizer
        if tokenizer is None:
            self._invalid_text_token_ids_cache = set()
            return self._invalid_text_token_ids_cache

        vocab_size = getattr(tokenizer, "vocab_size", None)
        if vocab_size is None:
            vocab = getattr(tokenizer, "get_vocab", lambda: {})()
            vocab_size = max(vocab.values()) + 1 if vocab else 0

        allowed_controls = {"\n", "\r", "\t"}
        special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])
        token_ids: set[int] = set()
        for token_id in range(int(vocab_size)):
            if token_id in special_ids:
                continue
            try:
                text = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
            except Exception:
                continue
            if any((ord(ch) < 32 and ch not in allowed_controls) or ord(ch) == 127 for ch in text):
                token_ids.add(token_id)

        self._invalid_text_token_ids_cache = token_ids
        return token_ids

    def _ensure_images_seqlens(self, multi_modal_inputs: dict, input_ids: torch.Tensor) -> dict:
        """Keep HFRollout multimodal metadata compatible with downstream metrics."""
        if "image_grid_thw" not in multi_modal_inputs or "images_seqlens" in multi_modal_inputs:
            return multi_modal_inputs

        image_token_id = getattr(self.processor, "image_token_id", None)
        if image_token_id is None:
            multi_modal_inputs["images_seqlens"] = torch.empty(0, dtype=torch.long)
            return multi_modal_inputs

        num_image_tokens = int((input_ids == image_token_id).sum().item())
        image_grid_thw = multi_modal_inputs.get("image_grid_thw")
        num_images = int(image_grid_thw.shape[0]) if isinstance(image_grid_thw, torch.Tensor) else 0
        if num_images <= 0:
            multi_modal_inputs["images_seqlens"] = torch.empty(0, dtype=torch.long)
            return multi_modal_inputs

        # For Qwen-VL processors, image pad tokens are contiguous per image. If
        # exact per-image spans are unavailable, split the total feature-token
        # count across images deterministically for metrics-only consumers.
        base = num_image_tokens // num_images
        remainder = num_image_tokens % num_images
        seqlens = torch.full((num_images,), base, dtype=torch.long)
        if remainder > 0:
            seqlens[:remainder] += 1
        multi_modal_inputs["images_seqlens"] = seqlens
        return multi_modal_inputs

    def _compute_position_ids(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, multi_modal_inputs: dict):
        """Compute (1, seq) text or (1, 4, seq) mrope position ids for a single sample.

        Mirrors the AgentLoop preprocessing so HF-rollout produces position ids in the
        same layout the rest of the pipeline (vLLM path) consumes.
        """
        if self.processor is None:
            return compute_position_id_with_mask(attention_mask)  # (1, seq_len)

        get_rope_index = getattr(self.processor, "get_rope_index", None)
        if get_rope_index is None:
            return compute_position_id_with_mask(attention_mask)

        multi_modal_kwargs = {
            "image_grid_thw": multi_modal_inputs.get("image_grid_thw"),
            "video_grid_thw": multi_modal_inputs.get("video_grid_thw"),
        }
        # For transformers>=5.3.0 Qwen3.5 uses mm_token_type_ids for multimodal rope.
        if multi_modal_inputs.get("mm_token_type_ids") is not None:
            mm_token_type_ids = torch.zeros_like(input_ids)
            mm_token_type_ids[0][input_ids[0] == self.processor.image_token_id] = 1
            video_token_id = getattr(self.processor, "video_token_id", None)
            if video_token_id is not None:
                mm_token_type_ids[0][input_ids[0] == video_token_id] = 2
            multi_modal_kwargs["mm_token_type_ids"] = mm_token_type_ids

        vision_position_ids, _ = get_rope_index(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **multi_modal_kwargs,
        )
        vision_position_ids = vision_position_ids.transpose(0, 1)  # (3, 1, seq) -> (1, 3, seq)

        valid_mask = attention_mask[0].bool()
        text_position_ids = torch.ones((1, input_ids.shape[-1]), dtype=torch.long, device=input_ids.device)
        text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item(), device=input_ids.device)
        text_position_ids = text_position_ids.unsqueeze(0)
        position_ids = torch.cat((text_position_ids, vision_position_ids.to(input_ids.device)), dim=1)  # (1, 4, seq)
        return position_ids

    def _resolve_chat_template_kwargs(self, prompts: DataProto | None = None) -> dict:
        kwargs = dict(self.apply_chat_template_kwargs)
        if prompts is not None:
            kwargs.update(prompts.meta_info.get("apply_chat_template_kwargs") or {})
        return kwargs

    def _divide_pixels_across_images(self) -> bool:
        return bool(self.config.get("divide_max_pixels_by_image_count", False))

    @staticmethod
    def _count_images(messages: list) -> int:
        count = 0
        for message in messages:
            content = message.get("content", "") if isinstance(message, dict) else ""
            if isinstance(content, list):
                count += sum(1 for item in content if isinstance(item, dict) and item.get("type") in ("image", "image_url"))
        return count

    def _maybe_divide_qwen_image_pixels(self, messages: list) -> list:
        if not self._divide_pixels_across_images():
            return messages
        image_count = self._count_images(messages)
        if image_count <= 1:
            return messages
        image_processor = getattr(self.processor, "image_processor", None)
        size = getattr(image_processor, "size", None) or {}
        max_pixels = int(size.get("longest_edge") or size.get("max_pixels") or 0)
        min_pixels = int(size.get("shortest_edge") or size.get("min_pixels") or 0)
        if max_pixels <= 0:
            return messages
        per_image_max = max(1, max_pixels // image_count)
        per_image_min = min(min_pixels, per_image_max) if min_pixels > 0 else per_image_max
        scaled = []
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                scaled.append(message)
                continue
            new_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") in ("image", "image_url"):
                    item = dict(item)
                    item["max_pixels"] = per_image_max
                    item["min_pixels"] = per_image_min
                new_content.append(item)
            message = dict(message)
            message["content"] = new_content
            scaled.append(message)
        return scaled

    def _tokenize_messages(self, messages: list[dict], chat_template_kwargs: dict | None = None):
        """Tokenize a single sample's chat messages into model inputs (mirrors AgentLoop)."""
        template_kwargs = dict(chat_template_kwargs or {})
        if self.processor is not None and getattr(self.processor, "image_processor", None) is not None:
            from qwen_vl_utils import process_vision_info

            template_kwargs = self._filter_chat_template_kwargs(self.processor, template_kwargs)
            image_patch_size = getattr(getattr(self.processor, "image_processor", None), "patch_size", None)
            vision_info_kwargs = {"return_video_metadata": True}
            if image_patch_size is not None:
                vision_info_kwargs["image_patch_size"] = image_patch_size
            messages = self._maybe_divide_qwen_image_pixels(messages)
            images, videos = process_vision_info(messages, **vision_info_kwargs)
            videos_kwargs = {}
            if videos:
                videos, video_metadata = zip(*videos, strict=True)
                videos = list(videos)
                videos_kwargs = {"video_metadata": list(video_metadata), "do_sample_frames": False}
            text = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **template_kwargs,
            )
            model_inputs = self.processor(
                text=[text],
                images=images,
                videos=videos,
                return_tensors="pt",
                videos_kwargs=videos_kwargs,
            )
            input_ids = model_inputs.pop("input_ids")[0]
            model_inputs.pop("attention_mask", None)
            multi_modal_inputs = dict(model_inputs.convert_to_tensors("pt"))
            multi_modal_inputs = self._ensure_images_seqlens(multi_modal_inputs, input_ids)
            return input_ids, multi_modal_inputs

        template_kwargs = self._filter_chat_template_kwargs(self.tokenizer, template_kwargs)
        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            **template_kwargs,
        )
        return torch.tensor(prompt_ids, dtype=torch.long), {}

    def _prompt_snippet(messages: list, limit: int = 160) -> str:
        parts = []
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else ""
            if isinstance(content, str):
                parts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
        text = " ".join(" ".join(parts).replace("<image>", " ").split())
        return text[:limit]

    def _maybe_build_vlm_inputs(self, prompts: DataProto) -> DataProto:
        """Build (left-padded) input_ids/attention_mask/position_ids/multi_modal_inputs from raw_prompt.

        In the agent-loop pipeline the dataset only emits `raw_prompt` messages, so the sync
        HF-rollout path must do the tokenization + multimodal preprocessing itself.
        """
        if prompts.batch is not None and "input_ids" in prompts.batch.keys():
            return prompts

        raw_prompts = prompts.non_tensor_batch.get("raw_prompt")
        if raw_prompts is None:
            raise ValueError(
                "HFRollout received neither pre-tokenized `input_ids` nor `raw_prompt` messages. "
                "Ensure the dataset/gen_batch carries `raw_prompt` for the HF rollout backend."
            )

        device = get_torch_device().current_device()
        _, pad_token_id = self._resolve_eos_pad_token_ids(getattr(prompts, "meta_info", {}) or {})

        chat_template_kwargs = self._resolve_chat_template_kwargs(prompts)
        max_prompt_length = int(self.config.get("prompt_length", 0) or 0)
        placeholder_id = pad_token_id if pad_token_id is not None else 0

        per_sample_ids = []
        per_sample_pos = []
        multi_modal_inputs_list = []
        overlong_flags = []
        for messages in raw_prompts:
            messages = list(messages)
            input_ids, mm_inputs = self._tokenize_messages(messages, chat_template_kwargs)
            prompt_len = int(input_ids.shape[-1])
            if max_prompt_length > 0 and prompt_len > max_prompt_length:
                print(
                    f"[HFRollout] skip overlong prompt: len={prompt_len} > "
                    f"max_prompt_length={max_prompt_length} text={self._prompt_snippet(messages)!r}",
                    flush=True,
                )
                input_ids = torch.tensor([placeholder_id], dtype=torch.long)
                mm_inputs = {}
                overlong_flags.append(True)
            else:
                overlong_flags.append(False)
            input_ids = input_ids.to(device)
            attn = torch.ones((1, input_ids.shape[-1]), dtype=torch.long, device=device)
            pos = self._compute_position_ids(input_ids.unsqueeze(0), attn, mm_inputs)  # (1,seq) or (1,4,seq)
            per_sample_ids.append(input_ids)
            per_sample_pos.append(pos.squeeze(0))  # (seq) or (4,seq)
            mm_inputs.pop("mm_token_type_ids", None)
            multi_modal_inputs_list.append({k: v for k, v in mm_inputs.items()})

        max_len = max(ids.shape[-1] for ids in per_sample_ids)
        batch_size = len(per_sample_ids)
        is_mrope = per_sample_pos[0].dim() == 2

        input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
        if is_mrope:
            rope_dims = per_sample_pos[0].shape[0]
            position_ids = torch.zeros((batch_size, rope_dims, max_len), dtype=torch.long, device=device)
        else:
            position_ids = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)

        for i, (ids, pos) in enumerate(zip(per_sample_ids, per_sample_pos)):
            seq_len = ids.shape[-1]
            input_ids[i, max_len - seq_len :] = ids  # left padding for generation
            attention_mask[i, max_len - seq_len :] = 1
            if is_mrope:
                position_ids[i, :, max_len - seq_len :] = pos
            else:
                position_ids[i, max_len - seq_len :] = pos

        new_batch = TensorDict(
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "overlong_prompt_mask": torch.tensor(overlong_flags, dtype=torch.bool, device=device),
            },
            batch_size=batch_size,
        )
        non_tensor_batch = dict(prompts.non_tensor_batch)
        non_tensor_batch["multi_modal_inputs"] = np.array(multi_modal_inputs_list, dtype=object)
        return DataProto(batch=new_batch, non_tensor_batch=non_tensor_batch, meta_info=dict(prompts.meta_info))

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        prompts = self._maybe_build_vlm_inputs(prompts)
        if bool(self.config.get("use_sft_response", False)):
            return self._pack_sft_responses(prompts)
        batch_size = prompts.batch.batch_size[0]
        num_chunks = max(batch_size // self.config.get("micro_batch_size", batch_size), 1)
        batch_prompts = prompts.chunk(chunks=num_chunks)
        output = []
        for batch_prompt in batch_prompts:
            output.append(self._generate_minibatch(batch_prompt))
        output = DataProto.concat(output)
        return output

    @torch.no_grad()
    def _generate_minibatch(self, prompts: DataProto) -> DataProto:
        # make sampling args can be overridden by inputs
        do_sample = prompts.meta_info.get("do_sample", self.config.do_sample)
        is_validate = prompts.meta_info.get("validate", False)

        temperature = prompts.meta_info.get("temperature", self.config.temperature)
        response_length = prompts.meta_info.get("response_length", self.config.response_length)
        top_p = prompts.meta_info.get("top_p", self.config.get("top_p", 1.0))
        top_k = max(0, prompts.meta_info.get("top_k", self.config.get("top_k", 0)))  # to be compatible with vllm

        if not do_sample:
            # do_sample==False -> greedy decoding
            kwargs = {
                "do_sample": False,
                "num_beams": 1,
            }
        elif is_validate:
            # do validate and do sample -> use val_kwargs
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_k": max(0, self.config.val_kwargs.top_k),  # to be compatible with vllm
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "num_return_sequences": 1,  # if validate, already repeat in ray_trainer
            }
        else:
            # do_sample -> use rollout config
            kwargs = {
                "do_sample": True,
                "num_beams": 1,
                "top_p": top_p,
                "top_k": top_k,
                "temperature": temperature,
                # already repeat in ray_trainer
                # https://github.com/volcengine/verl/blob/2fdfbdcba6f2e076f64bc47922d8fe6cf7dc7da5/verl/trainer/ppo/ray_trainer.py#L1117
                "num_return_sequences": 1,
            }

        idx = prompts.batch["input_ids"]  # (bs, prompt_length)
        prompt_length = idx.size(1)
        attention_mask = prompts.batch["attention_mask"]  # left-padded attention_mask
        position_ids = prompts.batch["position_ids"]
        # mrope position ids are 3D (bs, n_rope, seq); plain text position ids are 2D (bs, seq).
        is_mrope = position_ids.dim() == 3
        multi_modal_kwargs = self._build_multi_modal_kwargs(prompts, idx.device)

        # Rebuild mm_token_type_ids against the (left-padded) input_ids so the model can derive
        # multimodal positions internally during generation.
        # Qwen3.5 uses mm_token_type_ids for multimodal rope; Qwen3-VL derives mrope internally.
        if self._supports_mm_token_type_ids() and self.processor is not None and "pixel_values" in multi_modal_kwargs:
            image_token_id = getattr(self.processor, "image_token_id", None)
            if image_token_id is not None and "mm_token_type_ids" not in multi_modal_kwargs:
                mm_token_type_ids = torch.zeros_like(idx)
                mm_token_type_ids[idx == image_token_id] = 1
                video_token_id = getattr(self.processor, "video_token_id", None)
                if video_token_id is not None:
                    mm_token_type_ids[idx == video_token_id] = 2
                multi_modal_kwargs["mm_token_type_ids"] = mm_token_type_ids

        # used to construct attention_mask / response padding
        eos_token_id, pad_token_id = self._resolve_eos_pad_token_ids(prompts.meta_info)
        fallback_token_id = eos_token_id
        if isinstance(fallback_token_id, (list, tuple)):
            fallback_token_id = fallback_token_id[0] if fallback_token_id else pad_token_id
        if fallback_token_id is None:
            fallback_token_id = pad_token_id
        logits_processors = [_FiniteLogitsProcessor(int(fallback_token_id))]
        visual_special_token_ids = self._visual_special_token_ids()
        if visual_special_token_ids:
            logits_processors.append(_SuppressTokenIdsProcessor(visual_special_token_ids))
        invalid_text_token_ids = self._invalid_text_token_ids()
        if invalid_text_token_ids:
            logits_processors.append(_SuppressTokenIdsProcessor(invalid_text_token_ids))
        logits_processor = LogitsProcessorList(logits_processors)
        # When True, keep per-step scores so we can recover π_rollout(token) logprobs
        # from the same HF generate path (SAMoD sparse + cache), for rollout IS.
        calculate_log_probs = bool(self.config.get("calculate_log_probs", False))
        generation_config = GenerationConfig(
            **kwargs,
            max_new_tokens=response_length,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
            output_scores=calculate_log_probs,
            return_dict_in_generate=True,
            use_cache=True,
            remove_invalid_values=True,
            renormalize_logits=True,
        )

        self.module.eval()
        # Rollout generate stays on the normal cache forward. Do not pass
        # return_samod_trajectory / output_hidden_states; those belong to actor update.
        param_ctx = contextlib.nullcontext()

        # For mrope (VLM) we let the model derive position ids internally from input_ids +
        # image_grid_thw + mm_token_type_ids; passing 3D position ids to generate is unsupported.
        generate_position_kwargs = {} if is_mrope else {"position_ids": position_ids}

        if isinstance(self.module, FSDP):
            # recurse need to set to False according to https://github.com/pytorch/pytorch/issues/100069
            param_ctx = FSDP.summon_full_params(self.module, writeback=False, recurse=False)
        saved_cpu_rng = None
        saved_device_rng = None
        if kwargs.get("do_sample", False):
            rollout_seeds = prompts.non_tensor_batch.get("rollout_seed")
            if rollout_seeds is not None:
                saved_cpu_rng = torch.get_rng_state()
                saved_device_rng = get_torch_device().get_rng_state()
                sample_seed = _seed_from_rollout_seeds(rollout_seeds)
                torch.manual_seed(sample_seed)
                get_torch_device().manual_seed(sample_seed)

        try:
            with param_ctx, torch.autocast(device_type=get_device_name(), dtype=torch.bfloat16):
                output = self.module.generate(
                    input_ids=idx,
                    attention_mask=attention_mask,
                    **generate_position_kwargs,
                    **multi_modal_kwargs,
                    generation_config=generation_config,
                    logits_processor=logits_processor,
                )
        finally:
            if saved_cpu_rng is not None:
                torch.set_rng_state(saved_cpu_rng)
                get_torch_device().set_rng_state(saved_device_rng)

        # TODO: filter out the seq with no answers like ds-chat
        seq = output.sequences
        generated_batch_size = seq.size(0)  # bs * num_return_sequences

        # huggingface generate will stop generating when all the batch reaches [EOS].
        # We have to pad to response_length
        sequence_length = prompt_length + response_length
        delta_length = sequence_length - seq.shape[1]

        if delta_length > 0:
            delta_tokens = torch.ones(size=(generated_batch_size, delta_length), device=seq.device, dtype=seq.dtype)
            delta_tokens = pad_token_id * delta_tokens
            seq = torch.cat((seq, delta_tokens), dim=1)
        assert seq.shape[1] == sequence_length

        # make necessary reputations if num_return_sequences > 1
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        if num_return_sequences > 1:
            position_ids = position_ids.repeat_interleave(num_return_sequences, dim=0)
            attention_mask = attention_mask.repeat_interleave(num_return_sequences, dim=0)

        prompt = seq[:, :prompt_length]  # (generated_batch_size, prompt_length)
        response = seq[:, prompt_length:]  # (generated_batch_size, response_length)

        response_length = response.size(1)
        rollout_log_probs = None
        if calculate_log_probs:
            # output.scores: tuple[len=actual_new_tokens] of (bs, vocab). These are the
            # logits *after* logits processors / warpers (incl. temperature when do_sample),
            # i.e. the distribution that multinomial actually sampled from = π_rollout.
            if not getattr(output, "scores", None):
                raise RuntimeError(
                    "calculate_log_probs=True but HF generate returned no scores. "
                    "Ensure GenerationConfig.output_scores=True and return_dict_in_generate=True."
                )
            score_tensors = torch.stack(output.scores, dim=1)  # (bs, gen_len, vocab)
            # Greedy path skips TemperatureLogitsWarper; match actor logprob scaling.
            sample_temperature = float(kwargs.get("temperature", temperature) or 1.0)
            if (not kwargs.get("do_sample", True)) and sample_temperature > 0 and sample_temperature != 1.0:
                score_tensors = score_tensors / sample_temperature
            gen_len = score_tensors.size(1)
            if gen_len > response_length:
                score_tensors = score_tensors[:, :response_length, :]
                gen_len = response_length
            rollout_log_probs = logprobs_from_logits(score_tensors, response[:, :gen_len])
            if gen_len < response_length:
                pad = rollout_log_probs.new_zeros((generated_batch_size, response_length - gen_len))
                rollout_log_probs = torch.cat((rollout_log_probs, pad), dim=1)

        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)

        # Response tokens are text, so all rope components advance by 1 per step.
        if is_mrope:  # (bs, n_rope, seq)
            delta = delta_position_id.view(1, 1, -1).expand(generated_batch_size, position_ids.shape[1], -1)
            response_position_ids = position_ids[:, :, -1:] + delta
        else:  # (bs, seq)
            delta = delta_position_id.unsqueeze(0).expand(generated_batch_size, -1)
            response_position_ids = position_ids[:, -1:] + delta
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        if "overlong_prompt_mask" in prompts.batch.keys():
            skipped = prompts.batch["overlong_prompt_mask"].to(device=response_attention_mask.device, dtype=torch.bool)
            if skipped.shape[0] == response_attention_mask.shape[0] and bool(skipped.any()):
                response_attention_mask = response_attention_mask.clone()
                response_attention_mask[skipped] = 0
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # Different HF-rollout workers can receive different local max prompt
        # lengths. DataProto.concat requires identical non-batch dimensions, so
        # normalize the returned prompt side to the global rollout prompt_length.
        target_prompt_length = int(getattr(self.config, "prompt_length", prompt_length) or prompt_length)
        if prompt_length > target_prompt_length:
            raise ValueError(
                f"HFRollout prompt length {prompt_length} exceeds configured prompt_length={target_prompt_length}. "
                "Increase data.max_prompt_length / actor_rollout_ref.rollout.prompt_length."
            )
        if prompt_length < target_prompt_length:
            pad_len = target_prompt_length - prompt_length
            prompt_pad = prompt.new_full((generated_batch_size, pad_len), pad_token_id)
            prompt = torch.cat((prompt_pad, prompt), dim=1)
            seq = torch.cat((prompt_pad, seq), dim=1)
            attention_pad = attention_mask.new_zeros((generated_batch_size, pad_len))
            attention_mask = torch.cat((attention_pad, attention_mask), dim=1)
            if is_mrope:
                pos_pad = position_ids.new_zeros((generated_batch_size, position_ids.shape[1], pad_len))
            else:
                pos_pad = position_ids.new_zeros((generated_batch_size, pad_len))
            position_ids = torch.cat((pos_pad, position_ids), dim=-1)

        batch_tensors = {
            "prompts": prompt,
            "responses": response,
            "input_ids": seq,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            # Responses are physically right-appended after the common
            # left-padded prompt width.
            "response_start_idx": torch.full(
                (generated_batch_size,),
                target_prompt_length,
                dtype=torch.long,
                device=seq.device,
            ),
        }
        if rollout_log_probs is not None:
            batch_tensors["rollout_log_probs"] = rollout_log_probs
        if "overlong_prompt_mask" in prompts.batch.keys():
            batch_tensors["overlong_prompt_mask"] = prompts.batch["overlong_prompt_mask"].to(device=prompt.device)
        batch = TensorDict(batch_tensors, batch_size=generated_batch_size)

        # empty cache before compute old_log_prob
        get_torch_device().empty_cache()

        self.module.train()
        out = DataProto(batch=batch)
        # Carry multi_modal_inputs forward so downstream VLM log-prob/loss steps keep them.
        if "multi_modal_inputs" in prompts.non_tensor_batch:
            out.non_tensor_batch["multi_modal_inputs"] = prompts.non_tensor_batch["multi_modal_inputs"]
        return out

    @torch.no_grad()
    def _pack_sft_responses(self, prompts: DataProto) -> DataProto:
        """Right-append the dataset assistant answer. The actor weights are not sampled."""
        texts = prompts.non_tensor_batch.get("sft_response")
        if texts is None:
            raise ValueError("use_sft_response=True but the batch has no sft_response")
        if self.tokenizer is None:
            raise ValueError("use_sft_response requires a tokenizer on HFRollout")
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        is_mrope = position_ids.dim() == 3
        prompt_length = idx.size(1)
        batch_size = idx.size(0)
        eos_token_id, pad_token_id = self._resolve_eos_pad_token_ids(prompts.meta_info)
        device = idx.device
        eos_text = self.tokenizer.eos_token or ""
        overlong = prompts.batch.get("overlong_prompt_mask")
        encoded = []
        for i, text in enumerate(texts):
            if overlong is not None and bool(overlong[i].item()):
                encoded.append(idx.new_empty(0))
                continue
            token_ids = self.tokenizer(
                str(text) + eos_text,
                add_special_tokens=False,
                return_tensors="pt",
            )["input_ids"][0]
            if token_ids.numel() == 0:
                raise ValueError("sft_response tokenized to an empty sequence")
            encoded.append(token_ids)
        response_length = max((int(token_ids.numel()) for token_ids in encoded), default=0)
        if response_length == 0:
            response_length = 1
        response = torch.full((batch_size, response_length), pad_token_id, dtype=idx.dtype, device=device)
        for i, token_ids in enumerate(encoded):
            if token_ids.numel() == 0:
                continue
            response[i, : token_ids.numel()] = token_ids.to(device=device)

        seq = torch.cat((idx, response), dim=1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        if is_mrope:
            delta = delta_position_id.view(1, 1, -1).expand(batch_size, position_ids.shape[1], -1)
            response_position_ids = position_ids[:, :, -1:] + delta
        else:
            delta = delta_position_id.unsqueeze(0).expand(batch_size, -1)
            response_position_ids = position_ids[:, -1:] + delta
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(
            response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype
        )
        if overlong is not None and bool(overlong.any()):
            response_attention_mask = response_attention_mask.clone()
            response_attention_mask[overlong.to(dtype=torch.bool)] = 0
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        target_prompt_length = int(getattr(self.config, "prompt_length", prompt_length) or prompt_length)
        if prompt_length > target_prompt_length:
            raise ValueError(
                f"HFRollout prompt length {prompt_length} exceeds configured prompt_length={target_prompt_length}."
            )
        if prompt_length < target_prompt_length:
            pad_len = target_prompt_length - prompt_length
            prompt_pad = idx.new_full((batch_size, pad_len), pad_token_id)
            idx = torch.cat((prompt_pad, idx), dim=1)
            seq = torch.cat((prompt_pad, seq), dim=1)
            attention_mask = torch.cat((attention_mask.new_zeros((batch_size, pad_len)), attention_mask), dim=1)
            if is_mrope:
                pos_pad = position_ids.new_zeros((batch_size, position_ids.shape[1], pad_len))
            else:
                pos_pad = position_ids.new_zeros((batch_size, pad_len))
            position_ids = torch.cat((pos_pad, position_ids), dim=-1)

        batch_tensors = {
            "prompts": idx,
            "responses": response,
            "input_ids": seq,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "response_start_idx": torch.full(
                (batch_size,),
                target_prompt_length,
                dtype=torch.long,
                device=seq.device,
            ),
        }
        if overlong is not None:
            batch_tensors["overlong_prompt_mask"] = overlong
        batch = TensorDict(batch_tensors, batch_size=batch_size)
        out = DataProto(batch=batch)
        if "multi_modal_inputs" in prompts.non_tensor_batch:
            out.non_tensor_batch["multi_modal_inputs"] = prompts.non_tensor_batch["multi_modal_inputs"]
        return out
