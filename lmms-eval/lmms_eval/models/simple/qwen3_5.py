import math
import os
from typing import Optional

from loguru import logger as eval_logger

from lmms_eval.api.registry import register_model
from lmms_eval.models.simple.qwen3_vl import Qwen3_VL

# fetch_video uses image_patch_size 16 and merge 2, so one video token is 32x32 pixels.
_VIDEO_PIXEL_FACTOR = 32


def _raise_video_max_token_num(max_pixels: int) -> None:
    """Align qwen_vl_utils per-frame token cap with configured max_pixels.

    fetch_video clamps each frame to VIDEO_MAX_TOKEN_NUM * 32 * 32. The
    library default is 768, which silently replaces a larger max_pixels.
    Env ``VIDEO_MAX_TOKEN_NUM`` / ``MAX_TOKENS_PER_FRAME`` win when set.
    """
    import qwen_vl_utils.vision_process as vision_process

    needed = math.ceil(max_pixels / (_VIDEO_PIXEL_FACTOR * _VIDEO_PIXEL_FACTOR))
    for key in ("VIDEO_MAX_TOKEN_NUM", "MAX_TOKENS_PER_FRAME"):
        raw = os.environ.get(key)
        if raw:
            # Honor an explicit per-frame token budget (and never go below max_pixels).
            needed = max(needed, int(raw))
            break
    if vision_process.VIDEO_MAX_TOKEN_NUM != needed:
        eval_logger.info(
            f"Set VIDEO_MAX_TOKEN_NUM {vision_process.VIDEO_MAX_TOKEN_NUM} → {needed} "
            f"(per-frame max_pixels={max_pixels})"
        )
        vision_process.VIDEO_MAX_TOKEN_NUM = needed


@register_model("qwen3_5")
class Qwen3_5(Qwen3_VL):
    """
    Qwen3.5 model -- thin wrapper over Qwen3_VL with Qwen3.5-specific defaults.
    https://huggingface.co/Qwen/Qwen3.5-4B
    """

    DEFAULT_GEN_KWARGS = {
        "max_new_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
    }

    def __init__(
        self,
        pretrained: str = "Qwen/Qwen3.5-4B",
        min_pixels: int = 64 * 32 * 32,
        max_pixels: int = 128 * 32 * 32,
        total_pixels: int = 224 * 1024 * 32 * 32,
        max_num_frames: int = 768,
        max_frames: Optional[int] = None,
        enable_thinking: Optional[bool] = True,
        force_direct_answer: Optional[bool] = None,
        **kwargs,
    ):
        # Accept max_frames as backward-compat alias for max_num_frames
        if max_frames is not None:
            max_num_frames = max_frames
        super().__init__(
            pretrained=pretrained,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
            total_pixels=total_pixels,
            max_num_frames=max_num_frames,
            enable_thinking=enable_thinking,
            force_direct_answer=force_direct_answer,
            **kwargs,
        )
        _raise_video_max_token_num(self.max_pixels)

    def _build_video_kwargs(self):
        """Video kwargs that keep both the per-frame cap and the whole-video budget.

        Qwen3_VL drops ``max_pixels`` once ``total_pixels`` is set, and then
        samples at the qwen_vl_utils default of 2 fps. With ``fps`` unset this
        uses a fixed ``nframes`` count, and still forwards both pixel budgets.
        """
        video_kwargs = {
            "min_pixels": self.min_pixels,
            "max_pixels": self.max_pixels,
        }
        if self.fps is not None:
            video_kwargs["fps"] = self.fps
            video_kwargs["max_frames"] = self.max_num_frames
        else:
            video_kwargs["nframes"] = self.max_num_frames
        if self.total_pixels is not None:
            video_kwargs["total_pixels"] = self.total_pixels
        return video_kwargs
