# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2025 ModelBest Inc. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Multi-turn SFT dataset that supports training on conversation data with multiple turns
"""

import json
import logging
import os
import re
import copy
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.models.transformers.qwen3_vl import get_rope_index as get_rope_index_qwen3_vl
from verl.utils import hf_tokenizer
from verl.utils.chat_template import extract_system_prompt_and_generation
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.dataset.vision_utils import process_image
from verl.utils.fs import copy_local_path_from_hdfs

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def get_rope_index_qwen2_vl(
    processor,
    input_ids: torch.Tensor,
    image_grid_thw: Optional[torch.Tensor] = None,
    video_grid_thw: Optional[torch.Tensor] = None,
    second_per_grid_ts: Optional[torch.Tensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """MRoPE ids for Qwen2-VL-style processors, including Qwen3.5.

    Batch dim is already removed; ``input_ids`` is one example.
    """
    spatial_merge_size = processor.image_processor.merge_size
    tokens_per_second = 2
    image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    video_token_id = processor.tokenizer.convert_tokens_to_ids("<|video_pad|>")
    vision_start_token_id = processor.tokenizer.convert_tokens_to_ids("<|vision_start|>")
    if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        position_ids = torch.ones(3, input_ids.size(0), dtype=input_ids.dtype, device=input_ids.device)
        image_index, video_index = 0, 0
        input_ids = input_ids[attention_mask == 1]
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
                second_per_grid_t = 0
                image_index += 1
                remain_images -= 1
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                second_per_grid_t = second_per_grid_ts[video_index] if second_per_grid_ts is not None else 1.0
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
            t_index = torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w)
            t_index = (t_index * second_per_grid_t * tokens_per_second).long().flatten()
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
            position_ids = position_ids.unsqueeze(0).expand(3, -1).to(input_ids.device)
        else:
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).view(1, -1).expand(3, -1)
    return position_ids


def _read_parquet_dataframe(parquet_file: str) -> pd.DataFrame:
    """Read a parquet file into pandas, robust to nested chunked columns.

    Some sliced nested parquets fail in ``pd.read_parquet`` / dataset scanner with
    ``ArrowNotImplementedError: Nested data conversions not implemented for
    chunked array outputs``. Prefer a normal read; on failure, read row-group by
    row-group and concatenate.
    """
    try:
        return pd.read_parquet(parquet_file)
    except Exception as first_exc:
        try:
            import pyarrow.parquet as pq

            pf = pq.ParquetFile(parquet_file)
            frames = []
            for rg_idx in range(pf.num_row_groups):
                table = pf.read_row_group(rg_idx)
                try:
                    table = table.combine_chunks()
                except Exception:
                    pass
                frames.append(table.to_pandas())
            if not frames:
                raise first_exc
            logger.warning(
                "pd.read_parquet failed for %s (%s); fell back to row-group reads.",
                parquet_file,
                first_exc,
            )
            return pd.concat(frames, ignore_index=True)
        except Exception as second_exc:
            raise RuntimeError(
                f"Failed to read parquet {parquet_file}: {first_exc}"
            ) from second_exc


def _read_jsonl_dataframe(jsonl_file: str) -> pd.DataFrame:
    """Read a JSONL / JSON file into a pandas DataFrame."""
    path = Path(jsonl_file)
    if path.suffix.lower() == ".json" and not path.name.endswith(".jsonl"):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            # Common HF-style {"data": [...]} wrapper; otherwise treat as one row.
            if "data" in payload and isinstance(payload["data"], list):
                payload = payload["data"]
            else:
                payload = [payload]
        if not isinstance(payload, list):
            raise ValueError(f"JSON file must contain a list or dict, got {type(payload)}: {jsonl_file}")
        return pd.DataFrame(payload)

    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {jsonl_file}:{line_idx}") from exc
    if not rows:
        raise ValueError(f"JSONL file is empty: {jsonl_file}")
    return pd.DataFrame(rows)


def _read_sft_dataframe(data_file: str) -> pd.DataFrame:
    """Dispatch parquet / json / jsonl readers for MultiTurnSFTDataset."""
    suffix = Path(data_file).suffix.lower()
    if suffix == ".parquet":
        return _read_parquet_dataframe(data_file)
    if suffix in {".jsonl", ".json"}:
        return _read_jsonl_dataframe(data_file)
    raise ValueError(
        f"Unsupported SFT data file format '{suffix}' for {data_file}. "
        "Expected .parquet, .jsonl, or .json."
    )


def _resolve_media_path(path_like: str, media_root: Optional[str]) -> str:
    """Join relative media paths with ``media_root`` without touching the filesystem.

    Avoid ``Path.is_file()`` / ``exists()`` here: SFT datasets can have 100k+ rows
    and every rank would otherwise hammer shared storage during init.
    """
    path = str(path_like)
    if path.startswith(("http://", "https://", "file://")):
        return path
    candidate = Path(path)
    if candidate.is_absolute():
        return path
    if media_root:
        return str(Path(media_root) / path)
    return path


def _as_media_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        value = value.tolist()
    if isinstance(value, list):
        return list(value)
    return [value]


def _normalize_video_entry(
    entry: Any,
    *,
    media_root: Optional[str],
    video_sampling: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Normalize a video field into the dict expected by ``process_video``."""
    sampling = dict(video_sampling or {})

    if isinstance(entry, dict):
        video = dict(entry)
        raw = video.get("video", video.get("path", video.get("url")))
        if raw is None:
            raise ValueError(f"Video dict must contain video/path/url, got keys={sorted(video.keys())}")
        if isinstance(raw, str):
            video["video"] = _resolve_media_path(raw, media_root)
        else:
            # Frame-list style videos: leave as-is, only resolve string frames.
            if isinstance(raw, list):
                video["video"] = [
                    _resolve_media_path(frame, media_root) if isinstance(frame, str) else frame for frame in raw
                ]
            else:
                video["video"] = raw
        for key, val in sampling.items():
            video.setdefault(key, val)
        return video

    if isinstance(entry, str):
        video = {"video": _resolve_media_path(entry, media_root)}
        video.update(sampling)
        return video

    raise TypeError(f"Unsupported video entry type: {type(entry)}")


_OPD_THINKING_RE = re.compile(r"^<think>\n.*?</think>\n\n?", re.DOTALL)


def _maybe_convert_opd_prompt_to_messages(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Turn an OPD parquet (``prompt`` + ``reward_model.ground_truth``) into SFT messages.

    ``train_opd_integrated_25k`` stores the user turn in ``prompt`` and the answer in
    ``reward_model.ground_truth``. SFT training reads ``messages``. The conversion
    matches ``scripts/data_prepare/convert_opd_parquet_to_sft.py``, including stripping
    a leading ``<think>`` block from the answer.
    """
    if "messages" in dataframe.columns or "prompt" not in dataframe.columns or "reward_model" not in dataframe.columns:
        return dataframe

    messages = []
    for prompt, reward in zip(dataframe["prompt"], dataframe["reward_model"], strict=True):
        prompt = convert_nested_value_to_list_recursive(prompt)
        reward = convert_nested_value_to_list_recursive(reward)
        if not isinstance(prompt, list) or not prompt or not isinstance(prompt[0], dict):
            raise ValueError(f"OPD prompt must be a list of role/content dicts, got {type(prompt)}")
        user = prompt[0]
        content = user.get("content")
        if not isinstance(content, str):
            raise ValueError(f"OPD prompt content must be a string, got {type(content)}")
        answer = reward.get("ground_truth", "") if isinstance(reward, dict) else ""
        if not isinstance(answer, str):
            answer = "" if answer is None else str(answer)
        answer = _OPD_THINKING_RE.sub("", answer, count=1)
        messages.append(
            [
                {"role": user.get("role", "user"), "content": content},
                {"role": "assistant", "content": answer},
            ]
        )
    converted = dataframe.copy()
    converted["messages"] = messages
    logger.info("Converted %s OPD prompt rows into SFT messages.", len(messages))
    return converted


def _normalize_image_entry(entry: Any, *, media_root: Optional[str]) -> Any:
    """Normalize path-like image entries; leave bytes/PIL/dict payloads unchanged."""
    if isinstance(entry, str):
        return {"image": _resolve_media_path(entry, media_root)}
    if isinstance(entry, dict):
        image = dict(entry)
        if "image" not in image and "path" in image and isinstance(image["path"], str):
            image["image"] = _resolve_media_path(image["path"], media_root)
        elif isinstance(image.get("image"), str):
            image["image"] = _resolve_media_path(image["image"], media_root)
        return image
    return entry


def _normalize_media_columns(
    dataframe: pd.DataFrame,
    *,
    image_key: str,
    video_key: str,
    media_root: Optional[str] = None,
    video_sampling: Optional[dict[str, Any]] = None,
) -> pd.DataFrame:
    """Map singular image/video columns and path strings into SFT media lists.

    Existing parquet rows that already use ``images`` / ``videos`` lists are left
    intact aside from optional path resolution for string entries.
    """
    df = dataframe.copy()

    # Singular aliases used by VideoNSA-style jsonl (``video`` / ``image``).
    if video_key not in df.columns and "video" in df.columns:
        df[video_key] = df["video"]
    if image_key not in df.columns and "image" in df.columns:
        df[image_key] = df["image"]

    if video_key in df.columns:
        df[video_key] = df[video_key].apply(
            lambda value: [
                _normalize_video_entry(item, media_root=media_root, video_sampling=video_sampling)
                for item in _as_media_list(value)
            ]
            if value is not None and not (isinstance(value, float) and pd.isna(value))
            else []
        )

    if image_key in df.columns:
        df[image_key] = df[image_key].apply(
            lambda value: [
                _normalize_image_entry(item, media_root=media_root) for item in _as_media_list(value)
            ]
            if value is not None and not (isinstance(value, float) and pd.isna(value))
            else []
        )

    return df


def convert_nested_value_to_list_recursive(data_item):
    if isinstance(data_item, dict):
        return {k: convert_nested_value_to_list_recursive(v) for k, v in data_item.items()}
    elif isinstance(data_item, list):
        return [convert_nested_value_to_list_recursive(elem) for elem in data_item]
    elif isinstance(data_item, np.ndarray):
        # Convert to list, then recursively process the elements of the new list
        return convert_nested_value_to_list_recursive(data_item.tolist())
    else:
        # Base case: item is already a primitive type (int, str, float, bool, etc.)
        return data_item


class MultiTurnSFTDataset(Dataset):
    """
    Dataset for multi-turn conversations where each assistant response should be trained

    Args:
        data_files (str or list): Path(s) to Parquet / JSONL / JSON file(s).
        tokenizer (PreTrainedTokenizer): For the tokenization of text to token IDs.
        config (DictConfig): Options like cache_dir, prompt_key, max_prompt_length, truncation, etc.
        processor (ProcessorMixin, optional): Multimodal preprocessor for images/videos.
        max_samples (int, optional): Limit the number of samples. Defaults to -1 (use all).
    """

    def __init__(
        self,
        parquet_files: str | list[str],
        tokenizer: PreTrainedTokenizer,
        config: DictConfig,
        processor: Optional[ProcessorMixin] = None,
        max_samples: int = -1,
    ):
        # Set defaults and extract parameters from config if provided
        config = config or {}
        self.pad_mode = config.get("pad_mode", "right")
        assert self.pad_mode in ["right", "no_padding"], (
            f"Expect pad_mode to be 'right' or 'no_padding'. Got {self.pad_mode}"
        )
        self.truncation = config.get("truncation", "error")
        # for right padding
        self.max_length = config.get("max_length", 1024)
        # Get messages_key from the new multiturn config structure
        self.messages_key = config.get("messages_key", "messages")
        self.image_key = config.get("image_key", "images")
        self.video_key = config.get("video_key", "videos")
        self.image_patch_size = config.get(
            "image_patch_size", processor.image_processor.patch_size if processor else None
        )
        self.tools_key = config.get("tools_key", "tools")
        self.enable_thinking_key = config.get("enable_thinking_key", "enable_thinking")
        self.apply_chat_template_kwargs = config.get("apply_chat_template_kwargs", {})
        self.shuffle = config.get("shuffle", False)
        self.seed = config.get("seed")
        self.max_samples = max_samples
        self.ignore_input_ids_mismatch = config.get("ignore_input_ids_mismatch", False)
        # Optional root for relative video/image paths in jsonl rows.
        self.media_root = config.get("media_root", None) or config.get("video_root", None)
        # Optional default sampling overrides applied to path-style video entries.
        # NOTE: HF ``apply_chat_template`` ignores fps/nframes keys inside the message
        # content dict; they must be applied via ``processor_kwargs`` and/or by syncing
        # ``processor.video_processor`` defaults (see ``_sync_video_processor_sampling``).
        self.video_sampling = {}
        if config.get("video_nframes", None) is not None:
            self.video_sampling["nframes"] = int(config.get("video_nframes"))
        if config.get("video_fps", None) is not None:
            self.video_sampling["fps"] = float(config.get("video_fps"))
        if config.get("video_min_frames", None) is not None:
            self.video_sampling["min_frames"] = int(config.get("video_min_frames"))
        if config.get("video_max_frames", None) is not None:
            self.video_sampling["max_frames"] = int(config.get("video_max_frames"))
        if "nframes" in self.video_sampling and "fps" in self.video_sampling:
            raise ValueError(
                "data.video_nframes and data.video_fps are mutually exclusive; set only one."
            )
        # Optional per-frame pixel cap. HF Qwen3-VL video processor uses
        # size.longest_edge as a *total* budget over frames, so we convert:
        #   longest_edge ≈ video_max_pixels_per_frame * planned_num_frames
        self.video_max_pixels_per_frame = config.get("video_max_pixels_per_frame", None)
        if self.video_max_pixels_per_frame is not None:
            self.video_max_pixels_per_frame = int(self.video_max_pixels_per_frame)
        assert self.truncation in ["error", "left", "right"]

        if not isinstance(parquet_files, list | ListConfig):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer
        self.processor = processor
        self._sync_video_processor_sampling()

        self._download()
        self._read_files_and_process()

    def _sync_video_processor_sampling(self) -> None:
        """Make ``data.video_*`` sampling config actually control HF video decoding."""
        if self.processor is None or not hasattr(self.processor, "video_processor"):
            return
        vp = self.processor.video_processor
        if "nframes" in self.video_sampling:
            # Fixed-frame mode: disable fps default so sample_frames uses num_frames.
            if hasattr(vp, "fps"):
                vp.fps = None
        elif "fps" in self.video_sampling and hasattr(vp, "fps"):
            vp.fps = self.video_sampling["fps"]
        if "min_frames" in self.video_sampling and hasattr(vp, "min_frames"):
            vp.min_frames = self.video_sampling["min_frames"]
        if "max_frames" in self.video_sampling and hasattr(vp, "max_frames"):
            vp.max_frames = self.video_sampling["max_frames"]

    @staticmethod
    def _estimate_video_num_frames(
        video_path: str,
        *,
        fps: Optional[float],
        nframes: Optional[int],
        min_frames: int,
        max_frames: int,
    ) -> int:
        """Estimate frames HF will sample (used to convert per-frame pixel cap → total budget)."""
        if nframes is not None:
            return max(min_frames, min(int(nframes), max_frames))
        if fps is None or not isinstance(video_path, str) or not os.path.exists(video_path):
            return max_frames
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            total = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            raw_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0) or 1.0
            cap.release()
            duration_s = total / raw_fps
            planned = int(duration_s * float(fps))
            return max(min_frames, min(planned, max_frames, int(total) if total > 0 else max_frames))
        except Exception:
            return max_frames

    def _build_video_processor_kwargs(self, message: dict[str, Any]) -> dict[str, Any]:
        """Map message video dict fields → HF ``processor_kwargs`` (videos_kwargs).

        Message content may contain ``fps`` / ``nframes`` / ``min_frames`` / ``max_frames``
        (written by ``_build_messages`` from ``data.video_*``), but Transformers only honors
        these when passed as ``processor_kwargs``, not as content keys.
        """
        content = message.get("content")
        if not isinstance(content, list):
            return {}

        video_items = [c for c in content if isinstance(c, dict) and c.get("type") == "video"]
        if not video_items:
            return {}

        # One video per message is the SFT convention; use the first video's sampling.
        video = video_items[0]
        pk: dict[str, Any] = {"do_sample_frames": True}

        nframes = video.get("nframes", self.video_sampling.get("nframes"))
        fps = video.get("fps", self.video_sampling.get("fps"))
        min_frames = int(video.get("min_frames", self.video_sampling.get("min_frames", 4)))
        max_frames = int(video.get("max_frames", self.video_sampling.get("max_frames", 768)))

        if nframes is not None:
            pk["num_frames"] = int(nframes)
            pk["fps"] = None
        elif fps is not None:
            pk["fps"] = float(fps)

        pk["min_frames"] = min_frames
        pk["max_frames"] = max_frames

        if self.video_max_pixels_per_frame is not None:
            video_path = video.get("video")
            planned = self._estimate_video_num_frames(
                video_path if isinstance(video_path, str) else "",
                fps=None if nframes is not None else (float(fps) if fps is not None else None),
                nframes=int(nframes) if nframes is not None else None,
                min_frames=min_frames,
                max_frames=max_frames,
            )
            # Align to even frame count used by temporal_patch_size=2.
            planned = max(min_frames, planned - (planned % 2))
            longest_edge = int(self.video_max_pixels_per_frame) * int(planned)
            shortest_edge = None
            if self.processor is not None and hasattr(self.processor, "video_processor"):
                size = getattr(self.processor.video_processor, "size", None)
                shortest_edge = getattr(size, "shortest_edge", None) if size is not None else None
            if shortest_edge is None:
                shortest_edge = min(longest_edge, 128 * 32 * 32)
            pk["size"] = {"shortest_edge": int(shortest_edge), "longest_edge": longest_edge}

        return pk

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_local_path_from_hdfs(parquet_file, verbose=True)

    def _read_files_and_process(self):
        def series_to_item(ls):
            import numpy
            import pandas

            while isinstance(ls, pandas.core.series.Series | numpy.ndarray) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            dataframe = _read_sft_dataframe(parquet_file)
            dataframe = _maybe_convert_opd_prompt_to_messages(dataframe)
            dataframe = _normalize_media_columns(
                dataframe,
                image_key=self.image_key,
                video_key=self.video_key,
                media_root=self.media_root,
                video_sampling=self.video_sampling or None,
            )
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes, ignore_index=True)

        total = len(self.dataframe)
        print(f"dataset len: {len(self.dataframe)}")

        if self.max_samples > 0 and self.max_samples < total:
            if self.shuffle:
                rngs_args = (self.seed,) if self.seed is not None else ()
                rng = np.random.default_rng(*rngs_args)
                indices = rng.choice(total, size=self.max_samples, replace=False)
            else:
                indices = np.arange(self.max_samples)
            self.dataframe = self.dataframe.iloc[indices.tolist()]
            print(f"selected {self.max_samples} random samples out of {total}")

        # Extract messages list from dataframe
        self.messages = self.dataframe[self.messages_key].apply(convert_nested_value_to_list_recursive).tolist()

        # Extract tools list from dataframe
        if self.tools_key in self.dataframe.columns:
            self.tools = self.dataframe[self.tools_key].apply(convert_nested_value_to_list_recursive).tolist()
        else:
            self.tools = None
        # Extract enable_thinking list from dataframe
        if self.enable_thinking_key in self.dataframe.columns:
            self.enable_thinking = self.dataframe[self.enable_thinking_key].tolist()
        else:
            self.enable_thinking = None

        # system prompt: <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n
        # generation prompt: <|im_start|>assistant\n
        self.system_prompt, self.generation_prompt = extract_system_prompt_and_generation(self.tokenizer)

    def __len__(self):
        return len(self.messages)

    def _process_single_message(
        self,
        index: int,
        message: dict[str, Any],
        tools: Optional[list[dict[str, Any]]] = None,
        enable_thinking: Optional[bool] = None,
    ) -> tuple[list[int], list[int], list[int]]:
        """
        Process a single message and return its tokenized representation.

        Args:
            index: turn index in the conversation
            message: A single message dictionary
            images: List of images to be used
            videos: List of videos to be used
            tools: List of tools to be used
            enable_thinking: Whether to enable thinking mode

        Returns:
            Tuple of (input_ids, loss_mask, attention_mask, dict[str, torch.Tensor])
        """
        processor = self.processor if self.processor is not None else self.tokenizer
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking

        # HF ignores fps/nframes inside chat content; forward them via processor_kwargs.
        video_pk = self._build_video_processor_kwargs(message)
        if video_pk:
            existing_pk = dict(apply_chat_template_kwargs.get("processor_kwargs") or {})
            existing_pk.update(video_pk)
            apply_chat_template_kwargs["processor_kwargs"] = existing_pk

        inputs = processor.apply_chat_template(
            [message],
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **apply_chat_template_kwargs,
        )

        inputs = dict(inputs)
        input_ids = inputs.pop("input_ids")[0]
        attention_mask = inputs.pop("attention_mask")[0]

        # remove system prompt if exists
        if index != 0 and message["role"] != "system":
            input_ids = input_ids[len(self.system_prompt) :]
            attention_mask = attention_mask[len(self.system_prompt) :]

        if message["role"] == "assistant":
            loss_mask = torch.ones_like(attention_mask)
            # mask out generation prompt if assistant message
            loss_mask[: len(self.generation_prompt)] = 0
        else:
            loss_mask = torch.zeros_like(attention_mask)

        return input_ids, loss_mask, attention_mask, inputs

    def _build_messages(self, example: dict):
        """Replace <image> and <video> placeholder in messages with corresponding image and video
        which is required by processor.apply_chat_template.
        - <image>: {"type": "image", "image": image}
        - <video>: {"type": "video", **video_dict}  # one placeholder => one whole video
          Pass the path/sampling dict through (same as RL dataset). Do NOT pre-decode
          with ``process_video``: Qwen3-VL needs the path + fps metadata to treat the
          clip as a single video (timestamp-split frames). Prefetched tensors are
          mis-parsed as raw frame stacks and ignore VIDEO_FPS.

        Args:
            example: Row dictionary from dataframe.

        Returns:
            messages: List of messages with replaced placeholder.
        """
        # Deepcopy so in-place content rewrites do not corrupt the cached dataframe row
        # (otherwise a second fetch of the same index skips <video> placeholders).
        messages: list = copy.deepcopy(example[self.messages_key])
        images = example[self.image_key] if self.image_key in example else []
        videos = example[self.video_key] if self.video_key in example else []

        image_offset, video_offset = 0, 0
        for message in messages:
            if self.image_key not in example and self.video_key not in example:
                continue
            assert self.processor is not None, "processor is needed to process image and video"

            content = message["content"]
            if not isinstance(content, str):
                continue

            content_list = []
            segments = re.split("(<image>|<video>)", content)
            segments = [item for item in segments if item != ""]
            for segment in segments:
                if segment == "<image>":
                    assert image_offset < len(images), (
                        f"image_offset {image_offset} >= len(images) {len(images)}"
                    )
                    image = process_image(images[image_offset], image_patch_size=self.image_patch_size)
                    content_list.append({"type": "image", "image": image})
                    image_offset += 1
                elif segment == "<video>":
                    assert video_offset < len(videos), (
                        f"video_offset {video_offset} >= len(videos) {len(videos)}"
                    )
                    video_entry = videos[video_offset]
                    if not isinstance(video_entry, dict) or "video" not in video_entry:
                        raise TypeError(
                            "Each <video> placeholder requires a dict with key 'video' "
                            f"(path or frames), got: {type(video_entry)}"
                        )
                    # One <video> tag <-> one whole video clip (path + optional fps/nframes).
                    content_list.append({"type": "video", **dict(video_entry)})
                    video_offset += 1
                else:
                    content_list.append({"type": "text", "text": segment})
            message["content"] = content_list

        assert image_offset == len(images), f"image_offset {image_offset} != len(images) {len(images)}"
        assert video_offset == len(videos), f"video_offset {video_offset} != len(videos) {len(videos)}"
        return messages

    def __getitem__(self, item):
        row_dict: dict = self.dataframe.iloc[item].to_dict()
        messages = self._build_messages(row_dict)
        tools = self.tools[item] if self.tools is not None else None
        enable_thinking = self.enable_thinking[item] if self.enable_thinking is not None else None

        # 1. tokenize each message
        input_ids, loss_mask, attention_mask, multi_modal_inputs = [], [], [], {}
        for i, message in enumerate(messages):
            _input_ids, _loss_mask, _attention_mask, _inputs = self._process_single_message(
                index=i,
                message=message,
                tools=tools if i == 0 else None,
                enable_thinking=enable_thinking,
            )
            input_ids.append(_input_ids)
            loss_mask.append(_loss_mask)
            attention_mask.append(_attention_mask)
            for k, v in _inputs.items():
                multi_modal_inputs.setdefault(k, []).append(v)

        input_ids = torch.cat(input_ids, dim=0)
        loss_mask = torch.cat(loss_mask, dim=0)
        attention_mask = torch.cat(attention_mask, dim=0)
        assert input_ids.shape == loss_mask.shape == attention_mask.shape, (
            f"Shape mismatch: {input_ids.shape}, {loss_mask.shape}, {attention_mask.shape}"
        )
        self.sanity_check(input_ids, messages, tools, enable_thinking)

        # Since the tokenizer may return user-customized results, we need to filter out inconsistent tensor shapes
        keys_to_remove = []
        for k, v in multi_modal_inputs.items():
            if len(v) > 0 and v[0] is not None and isinstance(v[0], torch.Tensor):
                # Check if all tensors in the list have the same shape
                first_shape = v[0].shape[1:]
                if not all(tensor.shape[1:] == first_shape for tensor in v):
                    keys_to_remove.append(k)

        for k in keys_to_remove:
            del multi_modal_inputs[k]

        for k, v in multi_modal_inputs.items():
            multi_modal_inputs[k] = torch.concat(v, dim=0)

        # 2. handle position_ids for Qwen-VL series models
        # Qwen3-VL reuses Qwen2VLImageProcessor*, but video tokens are timestamp-
        # split and need qwen3_vl.get_rope_index (expands video_grid_thw by T).
        if self.processor is not None and "Qwen2VLImageProcessor" in self.processor.image_processor.__class__.__name__:
            image_grid_thw = multi_modal_inputs.get("image_grid_thw", None)
            video_grid_thw = multi_modal_inputs.get("video_grid_thw", None)
            second_per_grid_ts = multi_modal_inputs.get("second_per_grid_ts", None)

            processor_name = self.processor.__class__.__name__
            if "Qwen3VL" in processor_name:
                vision_position_ids = get_rope_index_qwen3_vl(
                    self.processor,
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    attention_mask=attention_mask,
                )  # (3, seq_len)
            else:
                vision_position_ids = get_rope_index_qwen2_vl(
                    self.processor,
                    input_ids=input_ids,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                    second_per_grid_ts=second_per_grid_ts,
                    attention_mask=attention_mask,
                )  # (3, seq_len)
            text_position_ids = torch.arange(input_ids.shape[0], dtype=torch.long).unsqueeze(0)  # (1, seq_len)
            position_ids = torch.cat((text_position_ids, vision_position_ids), dim=0)  # (4, seq_length)
        else:
            position_ids = torch.arange(input_ids.shape[0], dtype=torch.long)  # (seq_len,)

        # 3. handle padding
        sequence_length = input_ids.shape[0]
        # Handle sequence length
        if self.pad_mode == DatasetPadMode.RIGHT:
            if sequence_length < self.max_length:
                # Pad sequences
                pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
                padded_input_ids = torch.full((self.max_length - sequence_length,), pad_token_id, dtype=input_ids.dtype)
                padded_attention_mask = torch.zeros((self.max_length - sequence_length,), dtype=attention_mask.dtype)
                padded_loss_mask = torch.zeros((self.max_length - sequence_length,), dtype=loss_mask.dtype)

                input_ids = torch.cat((input_ids, padded_input_ids))
                attention_mask = torch.cat((attention_mask, padded_attention_mask))
                loss_mask = torch.cat((loss_mask, padded_loss_mask))
                position_ids = F.pad(position_ids, (0, self.max_length - sequence_length), value=0)
            elif sequence_length > self.max_length:
                if self.truncation == "left":
                    input_ids = input_ids[-self.max_length :]
                    attention_mask = attention_mask[-self.max_length :]
                    loss_mask = loss_mask[-self.max_length :]
                    position_ids = position_ids[..., -self.max_length :]
                elif self.truncation == "right":
                    input_ids = input_ids[: self.max_length]
                    attention_mask = attention_mask[: self.max_length]
                    loss_mask = loss_mask[: self.max_length]
                    position_ids = position_ids[..., : self.max_length]
                elif self.truncation == "error":
                    raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
                else:
                    raise ValueError(f"Unknown truncation method {self.truncation}")

            res = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
            if len(multi_modal_inputs) > 0:
                res["multi_modal_inputs"] = multi_modal_inputs
            return res
        elif self.pad_mode == DatasetPadMode.NO_PADDING:
            # Keep no-padding behavior consistent with the right-padding path:
            # silently truncating multimodal sequences can desynchronize image
            # placeholder tokens from pixel features.
            if len(input_ids) > self.max_length:
                if self.truncation == "left":
                    input_ids = input_ids[-self.max_length :]
                    loss_mask = loss_mask[-self.max_length :]
                    position_ids = position_ids[..., -self.max_length :]
                elif self.truncation == "right":
                    input_ids = input_ids[: self.max_length]
                    loss_mask = loss_mask[: self.max_length]
                    position_ids = position_ids[..., : self.max_length]
                elif self.truncation == "error":
                    raise ValueError(f"{sequence_length=} is larger than {self.max_length=}")
                else:
                    raise ValueError(f"Unknown truncation method {self.truncation}")

            # return nested tensor with out padding
            res = {
                "input_ids": input_ids,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            }
            if len(multi_modal_inputs) > 0:
                res["multi_modal_inputs"] = multi_modal_inputs
            return res
        else:
            raise ValueError(f"Unknown pad mode {self.pad_mode}")

    def sanity_check(self, input_ids: torch.Tensor, messages: list[dict], tools: list[dict], enable_thinking: bool):
        """Check concatenated input_ids of apply_chat_template to each turn equals
        apply_chat_template to whole messages.
        """
        processor = self.processor if self.processor is not None else self.tokenizer
        apply_chat_template_kwargs = {**self.apply_chat_template_kwargs}
        if enable_thinking is not None:
            apply_chat_template_kwargs["enable_thinking"] = enable_thinking
        inputs = processor.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            **apply_chat_template_kwargs,
        )

        error_message = (
            "MultiTurnSFTDataset apply_chat_template to each turn separately and concat `input_ids` "
            "as a whole sequence, which may not equal to apply_chat_template to whole messages at once.\n"
            "For example, Qwen Thinking series models add <think></think> tags to last turn, please check "
            "your tokenizer chat template settings.\n"
            "Set `ignore_input_ids_mismatch=True` to ignore input_ids mismatch and use the concatenated "
            "input_ids as the final input_ids. "
        )

        if not torch.equal(input_ids, inputs["input_ids"].squeeze(0)):
            if self.ignore_input_ids_mismatch:
                logger.warning_once(error_message)
            else:
                raise AssertionError(error_message)
