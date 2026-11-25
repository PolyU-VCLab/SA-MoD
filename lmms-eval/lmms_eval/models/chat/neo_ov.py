import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
import torchvision.transforms as T
from accelerate import Accelerator
from loguru import logger as eval_logger
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.models.model_utils.gen_metrics import log_metrics
from lmms_eval.protocol import ChatMessages

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
_NON_GENERATE_KEYS = {
    "add_special_tokens",
    "downsample_ratio",
    "enable_thinking",
    "max_pixels",
    "min_pixels",
    "patch_size",
    "remove_think",
    "system_prompt",
    "until",
    "upscale",
    "_text_tokens",
}


def _compute_default_rope_parameters(config, device=None, seq_len=None, **kwargs):
    """Original default RoPE. Prefer config.rope_theta, which NEO writes onto the H/W copy."""
    base = getattr(config, "rope_theta", None)
    if base is None:
        rope_parameters = getattr(config, "rope_parameters", None) or config.rope_scaling
        base = rope_parameters["rope_theta"]
    partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0) or 1.0
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial_rotary_factor)
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim))
    return inv_freq, 1.0


def _install_legacy_rope_default(pretrained: str) -> None:
    """Current transformers dropped ROPE_INIT_FUNCTIONS['default'] and, when
    finishing from_pretrained, calls Qwen3RotaryEmbedding.compute_default_rope_parameters.
    NEO's bundled class has neither."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    ROPE_INIT_FUNCTIONS.setdefault("default", _compute_default_rope_parameters)
    rotary_cls = get_class_from_dynamic_module("modeling_qwen3.Qwen3RotaryEmbedding", pretrained)
    if not hasattr(rotary_cls, "compute_default_rope_parameters"):
        rotary_cls.compute_default_rope_parameters = staticmethod(_compute_default_rope_parameters)


def _install_legacy_tied_weights_keys() -> None:
    """NEOChatModel.__init__ never calls post_init, so newer transformers crash
    in from_pretrained while marking tied weights."""
    from transformers.modeling_utils import PreTrainedModel

    if getattr(PreTrainedModel.mark_tied_weights_as_initialized, "_neo_ov_patched", False):
        return
    original = PreTrainedModel.mark_tied_weights_as_initialized

    def _mark_tied_weights_as_initialized(self, loading_info):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return original(self, loading_info)

    _mark_tied_weights_as_initialized._neo_ov_patched = True
    PreTrainedModel.mark_tied_weights_as_initialized = _mark_tied_weights_as_initialized


def _patch_qwen3_prefill(model) -> None:
    """Current transformers no longer builds cache_position for this nested
    Qwen3, while NEO's forward asserts it, position_ids and past_key_values
    are all present before its own fallback code."""
    qwen_model = model.language_model.model
    if getattr(qwen_model.forward, "_neo_ov_patched", False):
        return
    original = qwen_model.__class__.forward

    def forward(
        self,
        input_ids=None,
        indexes=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        cache_position=None,
        **kwargs,
    ):
        if inputs_embeds is not None:
            seqlen = inputs_embeds.shape[1]
            device = inputs_embeds.device
        elif input_ids is not None:
            seqlen = input_ids.shape[1]
            device = input_ids.device
        else:
            raise ValueError("NEO Qwen3 forward needs input_ids or inputs_embeds")
        if cache_position is None:
            past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(past_len, past_len + seqlen, device=device)
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        if past_key_values is None:
            from transformers.cache_utils import DynamicCache

            past_key_values = DynamicCache(config=self.config)
        return original(
            self,
            input_ids=input_ids,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

    forward._neo_ov_patched = True
    qwen_model.__class__.forward = forward


def _as_int(value: Any) -> int:
    return int(float(value))


def _as_float(value: Any) -> float:
    return float(value)


def _round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: Union[int, float], factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: Union[int, float], factor: int) -> int:
    return math.floor(number / factor) * factor


def _smart_resize(
    height: int,
    width: int,
    factor: int = 32,
    min_pixels: int = 65536,
    max_pixels: int = 4194304,
) -> Tuple[int, int]:
    if max(height, width) / min(height, width) > 200:
        raise ValueError(f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}")

    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, _floor_by_factor(height / beta, factor))
        w_bar = max(factor, _floor_by_factor(width / beta, factor))
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def _contrasting_background(image: Image.Image) -> Optional[Tuple[int, int, int]]:
    image_alpha = image.getchannel("A")
    if not image_alpha.getextrema()[0] == 0:
        return None

    pixels = image.getdata()
    non_transparent = [pixel[:3] for pixel in pixels if pixel[3] > 0]
    if not non_transparent:
        return None
    pixel_mean = sum(sum(pixel) for pixel in non_transparent) / (len(non_transparent) * 3)
    return (0, 0, 0) if pixel_mean > 382.5 else (255, 255, 255)


def _ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGBA":
        bg_color = _contrasting_background(image)
        if bg_color:
            background = Image.new("RGB", image.size, bg_color)
            background.paste(image, mask=image.split()[3])
            return background.convert("RGB")
    return image.convert("RGB")


def _fmt_mmss(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


def _fmt_mmss_dot(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    return f"{int(seconds // 60):02d}:{seconds % 60:05.2f}"


def _doc_window(doc: Any) -> Tuple[Optional[float], Optional[float]]:
    """MVBench stores the question span on the full video, not in the filename."""
    if not isinstance(doc, dict) or "start" not in doc or "end" not in doc:
        return None, None
    try:
        start, end = float(doc["start"]), float(doc["end"])
    except (TypeError, ValueError):
        return None, None
    if end <= start:
        return None, None
    return start, end


# VLMEvalKit NEOov-*-video user-message prefixes. The chat-template system
# message stays "Reason step by step."; these strings are part of the user turn.
_MVBENCH_SYS = (
    "Carefully watch the video and pay attention to the cause and sequence of events, "
    "the detail and movement of objects, and the action and pose of persons. "
    "Based on your observations, select the best option that accurately addresses the question.\n"
)
_MLVU_SYS = (
    "Carefully watch this video and pay attention to every detail. "
    "Based on your observations, select the best option that accurately addresses the question."
)
_VIDEOME_FRAMES = (
    "\nThese are the frames of a video. "
    "Select the best answer to the following multiple-choice question based on the video. "
    "Respond with only the letter (A, B, C, or D) of the correct option.\n"
)
# Tasks whose VLMEvalKit reader clips with the annotation start/end.
_MVBENCH_BOUND_TASKS = {
    "mvbench_action_sequence",
    "mvbench_action_prediction",
    "mvbench_object_interaction",
    "mvbench_action_localization",
    "mvbench_episodic_reasoning",
}


def _task_kind(task: Optional[str]) -> str:
    name = task or ""
    if name.startswith("mvbench"):
        return "mvbench"
    if "videomme" in name:
        return "videomme"
    if name.startswith("mlvu"):
        return "mlvu"
    if "longvideobench" in name:
        return "longvideobench"
    if "egoschema" in name:
        return "egoschema"
    return "generic"


def _as_list(value: Any) -> Optional[List[Any]]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, str):
        import ast

        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            return None
        if isinstance(parsed, (list, tuple)):
            return list(parsed)
        return None
    if value is None or isinstance(value, (bytes, dict)):
        return None
    try:
        return list(value)
    except TypeError:
        return None


def _sample_indices(total_frames: int, video_fps: float, max_frames: int, sample_fps: Optional[float]) -> List[int]:
    """VLMEvalKit Video-MME / MLVU / LongVideoBench sampling.

    Fixed frames: step = n / (nframe + 1), indices = 1..nframe.
    FPS: sample at sample_fps and cap at max_frames, indices start at 0.
    """
    total_frames = int(total_frames)
    if total_frames <= 0:
        return []
    if sample_fps is None:
        n = min(int(max_frames), total_frames)
        if n >= total_frames:
            return list(range(total_frames))
        step = total_frames / (n + 1)
        return [min(int(i * step), total_frames - 1) for i in range(1, n + 1)]

    duration = total_frames / video_fps
    required = int(duration * sample_fps)
    if required <= max_frames:
        step = video_fps / sample_fps
        count = max(required, 1)
    else:
        count = max_frames
        step = total_frames / (max_frames + 1)
    count = min(count, total_frames)
    return [min(int(i * step), total_frames - 1) for i in range(count)]


def _mvbench_indices(
    total_frames: int,
    video_fps: float,
    max_frames: int,
    start_time: Optional[float],
    end_time: Optional[float],
) -> List[int]:
    """VLMEvalKit MVBench.get_index: one frame at the center of each segment.

    start/end are applied only when the file is longer than that span. A clip
    that is already the annotated segment is sampled in full.
    """
    total_frames = int(total_frames)
    if total_frames <= 0:
        return []
    max_frame = total_frames - 1
    use_bound = start_time is not None and end_time is not None
    if use_bound:
        span = float(end_time) - float(start_time)
        duration = total_frames / video_fps
        if span <= 0 or duration <= span + 1.0:
            use_bound = False
    if use_bound:
        start_idx = max(0, int(round(float(start_time) * video_fps)))
        end_idx = min(int(round(float(end_time) * video_fps)), max_frame)
    else:
        start_idx, end_idx = 0, max_frame
    if end_idx < start_idx:
        end_idx = start_idx
    n = min(int(max_frames), end_idx - start_idx + 1)
    if n <= 0:
        return []
    if n >= end_idx - start_idx + 1:
        return list(range(start_idx, end_idx + 1))
    seg_size = float(end_idx - start_idx) / n
    indices = []
    for idx in range(n):
        frame = int(start_idx + (seg_size / 2.0) + round(seg_size * idx))
        indices.append(min(max(frame, 0), max_frame))
    return indices


def _timestamp_to_seconds(timestamp: Any) -> float:
    hour, minute, second = str(timestamp).split(":")
    return int(hour) * 3600 + int(minute) * 60 + float(second)


def _interleave_lvb_blocks(
    timestamps: List[float],
    subtitles: List[dict],
    starting: float,
    duration: float,
) -> List[Tuple[str, Any]]:
    """VLMEvalKit LongVideoBench insert_subtitles_into_frames, as prompt blocks.

    Each block is ("frame", frame_index) or ("text", subtitle_line).
    """
    blocks: List[Tuple[str, Any]] = []
    cursor = 0
    starting = float(starting or 0.0)
    duration = float(duration)
    for subtitle in subtitles:
        if "timestamp" in subtitle:
            start, end = subtitle["timestamp"]
            if not isinstance(end, float):
                end = duration
            start = float(start) - starting
            end = float(end) - starting
            subtitle_text = subtitle["text"]
        else:
            start = _timestamp_to_seconds(subtitle["start"]) - starting
            end = _timestamp_to_seconds(subtitle["end"]) - starting
            subtitle_text = subtitle["line"]
        midpoint = (start + end) / 2.0
        while cursor < len(timestamps) and timestamps[cursor] <= midpoint:
            blocks.append(("frame", cursor))
            cursor += 1
        if end - start < 1:
            end = midpoint + 0.5
            start = midpoint - 0.5
        if any(start < stamp < end for stamp in timestamps):
            blocks.append(("text", str(subtitle_text) + "\n"))
    while cursor < len(timestamps):
        blocks.append(("frame", cursor))
        cursor += 1
    return blocks


def _mvbench_question(doc: dict) -> str:
    candidates = _as_list(doc.get("candidates")) or []
    lines = [f"({chr(ord('A') + index)}) {candidate}" for index, candidate in enumerate(candidates)]
    return f"Question: {doc['question']}\nOptions:\n" + "\n".join(lines) + "\nOnly give the best option."


def _videomme_question(doc: dict) -> str:
    options = _as_list(doc.get("options")) or []
    body = str(doc["question"])
    if options:
        body = body + "\n" + "\n".join(str(option) for option in options)
    return (
        _VIDEOME_FRAMES
        + f"Question: {body}\nAnswer with the option's letter from the given choices directly."
    )


def _mlvu_question(text: str) -> str:
    """VLMEvalKit MLVU_MCQ.qa_template plus the trailing instruction.

    lmms stores the stem and the ``(A) ...`` lines in one string, and the task
    yaml appends ``Best option: (``. Official text is ``Question:`` / ``Options:``
    and stops at ``Only give the best option.``
    """
    for suffix in ("\nBest option: (", "\nBest option:(", "Best option:(", "\nBest option: "):
        text = text.replace(suffix, "")
    question_lines: List[str] = []
    option_lines: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line in {"Only give the best option.", "Only give the best option"}:
            continue
        if line.startswith("Best option"):
            continue
        if re.match(r"^\([A-Z]\) ", line):
            option_lines.append(line)
        elif option_lines:
            option_lines[-1] = option_lines[-1] + " " + line
        else:
            question_lines.append(line)
    if not option_lines:
        body = "\n".join(question_lines).rstrip()
        if "Only give the best option" not in body:
            body = body + "\nOnly give the best option."
        return body
    question = " ".join(question_lines).strip()
    return f"Question: {question}\nOptions:\n" + "\n".join(option_lines) + "\nOnly give the best option."


def _load_image(image_input: Any) -> Image.Image:
    if isinstance(image_input, Image.Image):
        return image_input.copy()
    if isinstance(image_input, (str, Path, os.PathLike)):
        return Image.open(image_input)
    raise TypeError(f"Unsupported image type for neo_ov: {type(image_input)}")


def _preprocess_pixel_values(pixel_values: torch.Tensor, patch_size: int = 16) -> Tuple[torch.Tensor, torch.Tensor]:
    c, h, w = pixel_values.shape
    grid_h = h // patch_size
    grid_w = w // patch_size
    flat_pixel_values = pixel_values.view(c, grid_h, patch_size, grid_w, patch_size).permute(1, 3, 0, 2, 4).reshape(grid_h * grid_w, c * patch_size**2)
    return flat_pixel_values, torch.tensor([[grid_h, grid_w]], device=pixel_values.device)


def _load_image_native(
    image_input: Any,
    patch_size: int = 16,
    downsample_ratio: float = 0.5,
    min_pixels: int = 65536,
    max_pixels: int = 4194304,
    upscale: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    image = _ensure_rgb(_load_image(image_input))
    if upscale:
        image = image.resize((image.width * 2, image.height * 2), Image.BILINEAR)

    resized_height, resized_width = _smart_resize(
        image.height,
        image.width,
        factor=int(patch_size // downsample_ratio),
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    image = image.resize((resized_width, resized_height))

    transform = T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ],
    )
    return _preprocess_pixel_values(transform(image).to(torch.float32), patch_size=patch_size)


@register_model("neo_ov")
class NeoOV(lmms):
    is_simple = False

    def __init__(
        self,
        pretrained: str = "Paranioar/NEO1_5-2B-SFT",
        device: Optional[str] = "cuda",
        device_map: Optional[str] = "auto",
        batch_size: Optional[Union[int, str]] = 1,
        load_in_8bit: bool = False,
        trust_remote_code: bool = True,
        torch_dtype: str = "bfloat16",
        patch_size: int = 16,
        min_pixels: int = 65536,
        max_pixels: int = 4194304,
        total_pixels: Optional[int] = None,
        downsample_ratio: float = 0.5,
        system_prompt: Optional[str] = "Reason step by step.",
        max_new_tokens: int = 16384,
        max_num_frames: int = 64,
        fps: Optional[float] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        assert kwargs == {}, f"Unexpected kwargs: {kwargs}"

        accelerator = Accelerator()
        self.accelerator = accelerator
        if accelerator.num_processes > 1:
            self._device = torch.device(f"cuda:{accelerator.local_process_index}")
            self.device_map = f"cuda:{accelerator.local_process_index}"
        else:
            self._device = torch.device(device)
            self.device_map = device_map if device_map else device

        dtype = getattr(torch, torch_dtype) if isinstance(torch_dtype, str) else torch_dtype
        self._tokenizer = AutoTokenizer.from_pretrained(pretrained, trust_remote_code=trust_remote_code, use_fast=False)
        _install_legacy_rope_default(pretrained)
        _install_legacy_tied_weights_keys()
        load_kwargs = {}
        if load_in_8bit:
            load_kwargs["load_in_8bit"] = True
            load_kwargs["device_map"] = self.device_map
        self._model = AutoModel.from_pretrained(
            pretrained,
            torch_dtype=dtype,
            trust_remote_code=trust_remote_code,
            low_cpu_mem_usage=not load_in_8bit,
            **load_kwargs,
        ).eval()
        if not load_in_8bit:
            self._model = self._model.to(self._device)
        _patch_qwen3_prefill(self._model)
        self.device_map = str(self._device)

        self.batch_size_per_gpu = int(batch_size)
        self.patch_size = _as_int(patch_size)
        self.min_pixels = _as_int(min_pixels)
        self.max_pixels = _as_int(max_pixels)
        self.total_pixels = None if total_pixels in (None, "None", "") else _as_int(total_pixels)
        self.downsample_ratio = _as_float(downsample_ratio)
        self.system_prompt = system_prompt
        self.max_new_tokens = _as_int(max_new_tokens)
        self.max_num_frames = _as_int(max_num_frames)
        self.fps = None if fps in (None, "None", "") else _as_float(fps)
        self._pixel_cap_warned = False
        self._short_gen_warned = False
        self._lvb_subtitle_warned = False
        self._config = self.model.config
        self._max_length = getattr(self._config, "max_position_embeddings", 2048)

        if accelerator.num_processes > 1:
            self._rank = accelerator.local_process_index
            self._world_size = accelerator.num_processes
            if accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
        else:
            self._rank = 0
            self._world_size = 1

    @property
    def config(self):
        return self._config

    @property
    def tokenizer(self):
        return self._tokenizer

    @property
    def model(self):
        return self.accelerator.unwrap_model(self._model) if hasattr(self, "accelerator") else self._model

    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        raise NotImplementedError("Loglikelihood is not implemented for neo_ov")

    def generate_until_multi_round(self, requests: List[Instance]) -> List[GenerationResult]:
        raise NotImplementedError("Multi-round generation is not implemented for neo_ov")

    def _temporal_header(self, frame_count: int, duration: float) -> str:
        sample_fps = frame_count / duration
        return (
            f"The time range of this video is [00:00-{_fmt_mmss(duration)}], "
            f"and the following is a series of {frame_count} frames sampled at {round(sample_fps, 1)} FPS.\n"
        )

    def _append_timestamped_frames(
        self,
        prompt_parts: List[str],
        images: List[Any],
        frames: List[Image.Image],
        timestamps: List[float],
        duration: float,
    ) -> None:
        """Video-MME / MLVU / LongVideoBench: time-range line, then [mm:ss.xx]:<image>."""
        if frames and duration > 0 and len(timestamps) == len(frames):
            prompt_parts.append(self._temporal_header(len(frames), duration))
            for frame, timestamp in zip(frames, timestamps):
                prompt_parts.append(f"[{_fmt_mmss_dot(timestamp)}]:<image>\n")
                images.append(frame)
            return
        for frame in frames:
            prompt_parts.append("<image>\n")
            images.append(frame)

    def _load_lvb_subtitles(self, doc: Any) -> Optional[List[dict]]:
        if not isinstance(doc, dict) or not doc.get("subtitle_path"):
            return None
        relative = str(doc["subtitle_path"])
        candidates = []
        try:
            from lmms_eval.tasks.longvideobench.utils import _resolve_dataset_dir

            cache_dir, _ = _resolve_dataset_dir("longvideobench_val_v.yaml", "subtitle_subdir", "subtitles")
            candidates.append(os.path.join(cache_dir, relative))
        except Exception as error:
            eval_logger.debug(f"LongVideoBench subtitle cache lookup failed: {error}")
        hf_home = os.environ.get("HF_HOME", "")
        if hf_home:
            candidates.append(os.path.join(hf_home, "datasets", "longvideobench", "subtitles", relative))
        for path in candidates:
            if path and os.path.exists(path):
                with open(path, "r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                if isinstance(payload, list):
                    return payload
        if not getattr(self, "_lvb_subtitle_warned", False):
            eval_logger.warning(
                "LongVideoBench subtitle file was not found; frames are sent without interleaved subtitles"
            )
            self._lvb_subtitle_warned = True
        return None

    def _append_lvb_frames(
        self,
        prompt_parts: List[str],
        images: List[Any],
        frames: List[Image.Image],
        timestamps: List[float],
        duration: float,
        doc: Any,
    ) -> None:
        """Official LongVideoBench_256frame inserts subtitles between timestamped frames."""
        if not frames or duration <= 0 or len(timestamps) != len(frames):
            self._append_timestamped_frames(prompt_parts, images, frames, timestamps, duration)
            return
        prompt_parts.append(self._temporal_header(len(frames), duration))
        subtitles = self._load_lvb_subtitles(doc)
        if not subtitles:
            for frame, timestamp in zip(frames, timestamps):
                prompt_parts.append(f"[{_fmt_mmss_dot(timestamp)}]:<image>\n")
                images.append(frame)
            return
        starting = 0.0
        clip_duration = duration
        if isinstance(doc, dict):
            if doc.get("starting_timestamp_for_subtitles") is not None:
                starting = float(doc["starting_timestamp_for_subtitles"])
            if doc.get("duration") is not None:
                clip_duration = float(doc["duration"])
        for kind, payload in _interleave_lvb_blocks(timestamps, subtitles, starting, clip_duration):
            if kind == "frame":
                index = int(payload)
                prompt_parts.append(f"[{_fmt_mmss_dot(timestamps[index])}]:<image>\n")
                images.append(frames[index])
            else:
                prompt_parts.append(str(payload))

    def _question_text(self, kind: str, doc: Any, fallback: str) -> str:
        if kind == "mvbench" and isinstance(doc, dict) and doc.get("question") and _as_list(doc.get("candidates")):
            return _mvbench_question(doc)
        if kind == "videomme" and isinstance(doc, dict) and doc.get("question"):
            return _videomme_question(doc)
        if kind == "mlvu":
            return _mlvu_question(fallback)
        return fallback

    def _serialize_messages(self, messages: ChatMessages, doc: Any = None, task: Optional[str] = None) -> Tuple[str, List[Any], Optional[str]]:
        kind = _task_kind(task)
        prompt_parts = []
        images = []
        system_prompt = None
        doc_start, doc_end = _doc_window(doc) if kind == "mvbench" and task in _MVBENCH_BOUND_TASKS else (None, None)
        question_parts: List[str] = []

        for message in messages.messages:
            if message.role == "system":
                system_text = "\n".join(content.text for content in message.content if content.type == "text").strip()
                if system_text:
                    system_prompt = system_text
                continue

            for content in message.content:
                if content.type == "text":
                    question_parts.append(content.text)
                elif content.type == "image":
                    prompt_parts.append("<image>\n")
                    images.append(content.url)
                elif content.type == "video":
                    start_time = content.start_time if content.start_time is not None else doc_start
                    end_time = content.end_time if content.end_time is not None else doc_end
                    frames, timestamps, duration = self._read_video(content.url, start_time, end_time, kind)
                    self._append_video(kind, prompt_parts, images, frames, timestamps, duration, doc)
                elif content.type == "audio":
                    try:
                        frames, timestamps, duration = self._read_video(getattr(content, "url", None), doc_start, doc_end, kind)
                    except Exception:
                        frames, timestamps, duration = [], [], 0.0
                    if frames:
                        self._append_video(kind, prompt_parts, images, frames, timestamps, duration, doc)
                    else:
                        eval_logger.warning("Skipping an audio input NEO-OV cannot encode")

        question = self._question_text(kind, doc, "".join(question_parts))
        if kind == "mvbench":
            prompt_parts.insert(0, _MVBENCH_SYS)
        elif kind == "mlvu" and prompt_parts:
            prompt_parts[0] = _MLVU_SYS + prompt_parts[0]
        elif kind == "mlvu":
            prompt_parts.append(_MLVU_SYS)
        prompt_parts.append(question)
        return "".join(prompt_parts).strip(), images, system_prompt

    def _append_video(
        self,
        kind: str,
        prompt_parts: List[str],
        images: List[Any],
        frames: List[Image.Image],
        timestamps: List[float],
        duration: float,
        doc: Any,
    ) -> None:
        if kind == "mvbench":
            for frame in frames:
                prompt_parts.append("<image>\n")
                images.append(frame)
            return
        if kind == "longvideobench":
            self._append_lvb_frames(prompt_parts, images, frames, timestamps, duration, doc)
            return
        self._append_timestamped_frames(prompt_parts, images, frames, timestamps, duration)

    def _window_seconds(self, total: int, video_fps: float, start_time: Optional[float], end_time: Optional[float]) -> Tuple[int, int, float]:
        """Clip to doc start/end. Skip the clip when the file is already that segment."""
        duration = total / video_fps
        if start_time is not None and end_time is not None and duration <= (float(end_time) - float(start_time)) + 1.0:
            start_time, end_time = None, None
        start = 0 if start_time is None else min(total - 1, max(0, int(float(start_time) * video_fps)))
        end = total if end_time is None else min(total, max(start + 1, int(float(end_time) * video_fps)))
        window = max((end - start) / video_fps, 1.0 / video_fps)
        return start, end, window

    def _select_indices(
        self,
        total: int,
        video_fps: float,
        start_time: Optional[float],
        end_time: Optional[float],
        kind: str,
    ) -> Tuple[List[int], float]:
        if kind == "mvbench":
            indices = _mvbench_indices(total, video_fps, self.max_num_frames, start_time, end_time)
            if start_time is not None and end_time is not None and indices:
                span = float(end_time) - float(start_time)
                duration = total / video_fps
                if span > 0 and duration > span + 1.0:
                    return indices, max(span, 1.0 / video_fps)
            return indices, total / video_fps
        start, end, duration = self._window_seconds(total, video_fps, start_time, end_time)
        local_indices = _sample_indices(end - start, video_fps, self.max_num_frames, self.fps)
        return [start + index for index in local_indices], duration

    def _read_video(
        self,
        path: Any,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        kind: str = "generic",
    ) -> Tuple[List[Image.Image], List[float], float]:
        if os.path.splitext(str(path))[1].lower() == ".webm":
            frames, timestamps, duration = self._read_webm_cv2(path, start_time, end_time, kind)
            if frames:
                return frames, timestamps, duration
            eval_logger.warning(f"OpenCV decoded no frames from {path}; using one blank frame")
            return [Image.new("RGB", (64, 64), (0, 0, 0))], [0.0], 0.0

        from decord import VideoReader, cpu

        # num_threads=1 avoids decord's threaded ffmpeg decoder returning EAGAIN (-11).
        reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
        total = len(reader)
        if total <= 0:
            return [], [], 0.0
        video_fps = float(reader.get_avg_fps()) or 1.0
        indices, duration = self._select_indices(total, video_fps, start_time, end_time, kind)
        if not indices:
            return [], [], 0.0
        frames = self._decode_frames(path, reader, indices)
        if not frames:
            eval_logger.warning(f"No frames decoded from {path}; using one blank frame")
            return [Image.new("RGB", (64, 64), (0, 0, 0))], [0.0], 0.0
        timestamps = [index / video_fps for index in indices[: len(frames)]]
        return frames, timestamps, duration

    def _decode_frames(self, path: Any, reader, indices: List[int]) -> List[Image.Image]:
        try:
            batch = reader.get_batch(indices).asnumpy()
            return [Image.fromarray(frame) for frame in batch]
        except Exception as batch_error:
            eval_logger.warning(f"decord get_batch failed on {path}: {batch_error}")

        frames: List[Image.Image] = []
        for index in indices:
            frame = self._decode_one_frame(path, index)
            if frame is None:
                frame = frames[-1] if frames else None
            if frame is not None:
                frames.append(frame)
        if frames:
            return frames
        return self._decode_with_cv2(path, indices)

    def _decode_one_frame(self, path: Any, index: int) -> Optional[Image.Image]:
        from decord import VideoReader, cpu

        try:
            reader = VideoReader(str(path), ctx=cpu(0), num_threads=1)
            return Image.fromarray(reader[int(index)].asnumpy())
        except Exception:
            return None

    def _read_webm_cv2(
        self,
        path: Any,
        start_time: Optional[float],
        end_time: Optional[float],
        kind: str = "generic",
    ) -> Tuple[List[Image.Image], List[float], float]:
        """Read .webm sequentially. Random seek via decord raises EAGAIN on these clips."""
        import cv2

        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return [], [], 0.0
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        video_fps = float(capture.get(cv2.CAP_PROP_FPS)) or 1.0
        if total <= 0:
            capture.release()
            return [], [], 0.0
        indices, duration = self._select_indices(total, video_fps, start_time, end_time, kind)
        if not indices:
            capture.release()
            return [], [], 0.0
        wanted = set(indices)
        last = max(indices)
        grabbed = {}
        index = 0
        while index <= last:
            ok, frame = capture.read()
            if not ok:
                break
            if index in wanted:
                grabbed[index] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            index += 1
        capture.release()
        frames: List[Image.Image] = []
        kept_indices: List[int] = []
        for index in indices:
            frame = grabbed.get(index)
            if frame is None and frames:
                frame = frames[-1]
                index = kept_indices[-1]
            if frame is not None:
                frames.append(frame)
                kept_indices.append(index)
        timestamps = [index / video_fps for index in kept_indices]
        return frames, timestamps, duration

    def _decode_with_cv2(self, path: Any, indices: List[int]) -> List[Image.Image]:
        try:
            import cv2
        except ImportError:
            return []
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return []
        frames: List[Image.Image] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(index))
            ok, frame = capture.read()
            if not ok:
                if frames:
                    frames.append(frames[-1])
                continue
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        capture.release()
        return frames

    def _cap_max_pixels(
        self,
        num_images: int,
        max_pixels: int,
        patch_size: int,
        downsample_ratio: float,
        text_tokens: int = 768,
        total_pixels: Optional[int] = None,
        min_pixels: Optional[int] = None,
    ) -> int:
        """Keep every frame's visual tokens inside the LLM context.

        Official video max_pixels is 360*640. That fits 64 and 128 frames.
        256 frames at that resolution, plus timestamp and subtitle text, exceed
        NEO's 40960 positions and the RoPE kernel device-asserts, so only that
        case is reduced. ``text_tokens`` is the non-image prompt already built.
        """
        if num_images <= 0:
            return max_pixels
        if total_pixels is not None:
            required_min = self.min_pixels if min_pixels is None else int(min_pixels)
            size_factor = int(patch_size / downsample_ratio)
            quantum = size_factor * size_factor
            total_cap = (int(total_pixels) // num_images // quantum) * quantum
            if total_cap < required_min:
                raise ValueError(
                    f"total_pixels={total_pixels} cannot satisfy {num_images} frames "
                    f"* min_pixels={required_min}"
                )
            max_pixels = min(max_pixels, total_cap)
        llm_config = getattr(self._config, "llm_config", self._config)
        max_pos = int(getattr(llm_config, "max_position_embeddings", 40960))
        # Two extra tokens per frame: <img> and </img>, which are not in text_tokens.
        budget = max_pos - max(int(text_tokens), 0) - 2 * num_images
        tokens_each = max(1, budget // num_images)
        pixel_cap = int(tokens_each * (patch_size ** 2) / (downsample_ratio ** 2))
        pixel_cap = max(pixel_cap, patch_size * patch_size)
        if max_pixels > pixel_cap and not self._pixel_cap_warned:
            eval_logger.warning(
                f"Clamping max_pixels {max_pixels} -> {pixel_cap} so {num_images} frames "
                f"and {text_tokens} text tokens fit in {max_pos} positions"
            )
            self._pixel_cap_warned = True
        return min(max_pixels, pixel_cap)

    def _build_image_inputs(
        self,
        images: List[Any],
        gen_kwargs: dict,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if not images:
            return None, None

        patch_size = _as_int(gen_kwargs.get("patch_size", self.patch_size))
        min_pixels = _as_int(gen_kwargs.get("min_pixels", self.min_pixels))
        max_pixels = _as_int(gen_kwargs.get("max_pixels", self.max_pixels))
        total_pixels_value = gen_kwargs.get("total_pixels", self.total_pixels)
        total_pixels = None if total_pixels_value in (None, "None", "") else _as_int(total_pixels_value)
        downsample_ratio = _as_float(gen_kwargs.get("downsample_ratio", self.downsample_ratio))
        text_tokens = int(gen_kwargs.get("_text_tokens", 768))
        max_pixels = self._cap_max_pixels(
            len(images),
            max_pixels,
            patch_size,
            downsample_ratio,
            text_tokens,
            total_pixels,
            min_pixels,
        )
        if min_pixels > max_pixels:
            min_pixels = max_pixels
        upscale_value = gen_kwargs.get("upscale", False)
        upscale = str(upscale_value).lower() == "true" if isinstance(upscale_value, str) else bool(upscale_value)

        pixel_values_list = []
        grid_hw_list = []
        for image in images:
            pixel_values, grid_hw = _load_image_native(
                image,
                patch_size=patch_size,
                downsample_ratio=downsample_ratio,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                upscale=upscale,
            )
            target_device = "cuda" if self.device_map == "auto" else self.device
            pixel_values_list.append(pixel_values.to(target_device).to(torch.bfloat16))
            grid_hw_list.append(grid_hw.to(target_device))

        return torch.cat(pixel_values_list, dim=0), torch.cat(grid_hw_list, dim=0)

    def _build_generation_config(self, gen_kwargs: dict) -> dict:
        generation_config = {
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "repetition_penalty": 1.05,
            "max_new_tokens": self.max_new_tokens,
        }
        generation_config.update({k: v for k, v in gen_kwargs.items() if k not in _NON_GENERATE_KEYS})

        # lmms-eval task yamls request 16-32 greedy tokens for short-answer models.
        # NEO's official video config samples a long trace (max_new_tokens=16384);
        # cutting it at 32 drops the option letter and the score collapses.
        max_new = generation_config.get("max_new_tokens", self.max_new_tokens)
        try:
            max_new = int(max_new)
        except (TypeError, ValueError):
            max_new = self.max_new_tokens
        if max_new <= 32:
            if not self._short_gen_warned:
                eval_logger.warning(
                    f"Raising max_new_tokens {max_new} -> 16384 to match NEO VLMEvalKit video decoding"
                )
                self._short_gen_warned = True
            generation_config["do_sample"] = True
            generation_config["temperature"] = 0.6
            generation_config["top_p"] = 0.95
            generation_config["top_k"] = 20
            generation_config["repetition_penalty"] = 1.05
            generation_config["max_new_tokens"] = 16384

        if generation_config.get("temperature") in (0, 0.0):
            generation_config["do_sample"] = False
        if generation_config.get("do_sample") is False:
            generation_config["temperature"] = None
            if generation_config.get("top_p", None) == 1.0:
                generation_config["top_p"] = None
            generation_config["top_k"] = None
        else:
            generation_config.pop("num_beams", None)
        return {k: v for k, v in generation_config.items() if v is not None}

    @staticmethod
    def _strip_thinking(response: str) -> str:
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE)
        response = re.sub(r"<thinking>.*?</thinking>", "", response, flags=re.DOTALL | re.IGNORECASE)
        return response.strip()

    @torch.no_grad()
    def generate_until(self, requests: List[Instance]) -> List[GenerationResult]:
        res = []
        pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
        total_elapsed_time = 0
        total_tokens = 0

        for request in requests:
            ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.args
            doc = self.task_dict[task][split][doc_id]
            messages = ChatMessages(**{"messages": doc_to_messages(doc)})
            prompt, images, message_system_prompt = self._serialize_messages(messages, doc, task)
            request_system_prompt = gen_kwargs.get("system_prompt", message_system_prompt if message_system_prompt is not None else self.system_prompt)
            if request_system_prompt is not None:
                self.model.system_message = str(request_system_prompt).replace("\\n", "\n")

            text_tokens = len(self.tokenizer.encode(prompt.replace("<image>", ""), add_special_tokens=True)) + 32
            gen_kwargs = dict(gen_kwargs)
            gen_kwargs["_text_tokens"] = text_tokens
            pixel_values, grid_hw = self._build_image_inputs(images, gen_kwargs)
            generation_config = self._build_generation_config(gen_kwargs)

            start_time = time.time()
            response = self.model.chat(
                self.tokenizer,
                pixel_values=pixel_values,
                grid_hw=grid_hw,
                question=prompt,
                generation_config=generation_config,
                verbose=False,
            )
            end_time = time.time()

            if gen_kwargs.get("remove_think", False):
                response = self._strip_thinking(response)

            output_tokens = len(self.tokenizer.encode(response, add_special_tokens=False))
            total_elapsed_time += end_time - start_time
            total_tokens += output_tokens
            res.append(GenerationResult(text=response, token_counts=TokenCounts(output_tokens=output_tokens)))
            self.cache_hook.add_partial("generate_until", (ctx, gen_kwargs), response)
            eval_logger.debug(f"Question: {prompt}")
            eval_logger.debug(f"Model Response: {response}")
            pbar.update(1)

        avg_speed = total_tokens / total_elapsed_time if total_elapsed_time > 0 else 0
        log_metrics(
            total_gen_tokens=total_tokens,
            total_elapsed_time=total_elapsed_time,
            avg_speed=avg_speed,
            additional_metrics={"rank": self.rank},
        )

        pbar.close()
        return res
